"""Own serving resources independently of trainers and rollout managers."""

import multiprocessing
from dataclasses import dataclass

import ray

from slime.ray.training_recovery import TrainingRecovery
from slime.ray.utils import Lock
from slime.utils.health_monitor import RolloutHealthMonitor


@dataclass
class ServingDeployment:
    placements: dict
    servers: dict
    controller: object
    restore_plan: object
    routers: dict
    reused: bool = False


@ray.remote
class ServingCluster:
    """A named, detached owner for routers, engines and their GPU placements."""

    def __init__(self, args, restore_plan):
        from slime.backends.sglang_utils.deployment import start_rollout_servers
        from slime.observability.logging_utils import configure_logger
        from slime.ray.placement_group import create_placement_groups

        configure_logger()
        self.args = args
        self.restore_plan = restore_plan
        self.configuration = TrainingRecovery._configuration(args)
        self.placements = create_placement_groups(args)
        existing_children = {process.pid for process in multiprocessing.active_children()}
        self.servers, handles = (
            ({}, []) if args.debug_train_only else start_rollout_servers(args, self.placements["rollout"])
        )
        self.router_processes = [
            process for process in multiprocessing.active_children() if process.pid not in existing_children
        ]
        ray.get(handles)
        self.controller = None
        self.lock = Lock.options(num_cpus=0, num_gpus=0).remote()
        self.driver_job_id = None
        self.training_actors = {}
        self.attachments = 0
        self._health_monitors = []
        self._ci_fault_injection_pending = args.ci_test

    def get_queue_controller(self):
        """Create only when the built-in source or batch builder needs a queue."""
        if self.controller is None:
            from slime.data.queue_data_source import create_queue_controller

            self.controller = create_queue_controller(self.args, restore_plan=self.restore_plan)
        return self.controller

    def deployment(self, reused=False):
        routers = {
            name: getattr(self.args, name)
            for name in ("sglang_router_ip", "sglang_router_port", "sglang_model_routers")
            if hasattr(self.args, name)
        }
        return ServingDeployment(self.placements, self.servers, self.controller, self.restore_plan, routers, reused)

    def validate_attachment(self, args):
        configuration = TrainingRecovery._configuration(args)
        changed = [name for name, value in self.configuration.items() if configuration.get(name) != value]
        if changed:
            raise ValueError("Retained serving requires unchanged rollout/model configuration: " + ", ".join(changed))
        if self.driver_job_id is not None:
            from ray._private.state import jobs

            previous = next((job for job in jobs() if job["JobID"] == self.driver_job_id), None)
            if previous is None or not previous["IsDead"]:
                raise RuntimeError(f"Serving session is still owned by training job {self.driver_job_id}")

    def attach_training(self, args, job_id):
        from ray.util.placement_group import remove_placement_group

        from slime.ray.placement_group import _create_placement_group

        self.validate_attachment(args)
        self.health_monitoring_pause()
        self.release_trainers()
        reused = self.attachments > 0
        count = args.actor_num_nodes * args.actor_num_gpus_per_node
        actor_pg = self.placements["actor"]
        if args.colocate:
            if count > len(actor_pg[1]):
                raise ValueError("Restarted colocated trainer exceeds the retained GPU placement")
        elif not args.debug_rollout_only and count != len(actor_pg[1]):
            if actor_pg[0] is not None:
                remove_placement_group(actor_pg[0])
            self.placements["actor"] = _create_placement_group(count)
            self.placements["critic"] = self.placements["actor"] if args.use_critic else None
        if reused:
            engines = [engine for server in self.servers.values() for engine in server.engines if engine is not None]
            ray.get([engine.reset_weights_update_groups.remote() for engine in engines])
            ray.kill(self.lock, no_restart=True)
            self.lock = Lock.options(num_cpus=0, num_gpus=0).remote()
            server = self._updatable_server()
            if server:
                for group in server.server_groups:
                    group.num_new_engines = len([engine for engine in group.engines if engine is not None])
        # Internal serving is always monitored, independently of the legacy flag.
        for monitor in self._health_monitors:
            monitor.stop()
        self._health_monitors = []
        for server in self.servers.values():
            for group in server.server_groups:
                monitor = RolloutHealthMonitor(group, args)
                monitor.start()
                monitor.pause()
                self._health_monitors.append(monitor)
        for name, value in self.deployment().routers.items():
            setattr(args, name, value)
        self.args = args
        self.driver_job_id = job_id
        self.attachments += 1
        return self.deployment(reused)

    def try_ci_fault_injection(self):
        server = self._updatable_server()
        if not self._ci_fault_injection_pending or server is None:
            return self.deployment()
        self._ci_fault_injection_pending = False
        engines = [engine for engine in server.all_engines if engine is not None]
        if engines:
            ray.get(engines[0].simulate_crash.remote(), timeout=self.args.rollout_health_check_timeout)
            for monitor in self._health_monitors:
                monitor.check_once()
        return self.deployment()

    def register_trainers(self, role, actors):
        self.training_actors[role] = actors

    def release_trainers(self):
        for actors in self.training_actors.values():
            for actor in actors:
                ray.kill(actor, no_restart=True)
        self.training_actors.clear()

    def detach_training(self, job_id):
        if self.driver_job_id != job_id:
            raise RuntimeError("Only the owning driver can detach its serving session")
        self.health_monitoring_pause()
        self.release_trainers()
        self.driver_job_id = None

    def _updatable_server(self):
        return next((server for server in self.servers.values() if server.update_weights), None)

    def get_updatable_engines_and_lock(self):
        server = self._updatable_server()
        if server is None:
            return [], self.lock, 0, [], [], []
        return (
            server.engines,
            self.lock,
            server.num_new_engines,
            server.engine_gpu_counts,
            server.engine_gpu_offsets,
            server.engine_parallel_configs,
        )

    def get_weight_version(self, *, allow_inconsistent=False):
        server = self._updatable_server()
        engines = [engine for engine in server.engines if engine is not None] if server else []
        versions = ray.get([engine.get_weight_version.remote() for engine in engines])
        if not versions:
            return None
        if allow_inconsistent:
            # A partial update or a replacement engine may have no version yet.
            return max((int(version) for version in versions if str(version).isdigit()), default=0)
        if any(not str(version).isdigit() for version in versions):
            raise RuntimeError(f"Cannot resume nonnumeric serving weight versions: {versions}")
        if len(set(versions)) != 1:
            raise RuntimeError(f"Cannot checkpoint inconsistent serving weight versions: {versions}")
        return int(versions[0])

    def recover_updatable_engines(self):
        self.health_monitoring_pause()
        server = self._updatable_server()
        if server:
            server.recover()
        return self.deployment()

    def clear_updatable_num_new_engines(self):
        server = self._updatable_server()
        if server:
            server.num_new_engines = 0

    def health_monitoring_pause(self, *, check=False):
        for monitor in self._health_monitors:
            monitor.pause()
        if check:
            for monitor in self._health_monitors:
                monitor.check_once()
            return self.deployment()

    def health_monitoring_resume(self):
        for monitor in self._health_monitors:
            monitor.resume()
        return self.deployment()

    def offload(self):
        self.health_monitoring_pause()
        for server in self.servers.values():
            server.offload()

    def onload(self, tags=None):
        for server in self.servers.values():
            server.onload(tags)

    def onload_weights(self):
        for server in self.servers.values():
            server.onload_weights()

    def onload_kv(self):
        for server in self.servers.values():
            server.onload_kv()

    def check_weights(self, action):
        engines = [engine for server in self.servers.values() for engine in server.engines]
        return ray.get([engine.check_weights.remote(action=action) for engine in engines])

    def dispose(self):
        from ray.util.placement_group import remove_placement_group

        self.release_trainers()
        for monitor in self._health_monitors:
            monitor.stop()
        engines = [engine for server in self.servers.values() for engine in server.all_engines if engine is not None]
        for result in [engine.shutdown.remote() for engine in engines]:
            try:
                ray.get(result)
            except ray.exceptions.RayActorError:
                pass
        for engine in engines:
            ray.kill(engine, no_restart=True)
        for router in self.router_processes:
            router.terminate()
            router.join(timeout=10)
            if router.is_alive():
                router.kill()
                router.join()
        if self.controller is not None:
            ray.get(self.controller.close.remote())
            ray.kill(self.controller, no_restart=True)
        ray.kill(self.lock, no_restart=True)
        for group in {placement[0] for placement in self.placements.values() if placement and placement[0]}:
            remove_placement_group(group)
