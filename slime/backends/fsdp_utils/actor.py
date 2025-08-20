from slime.ray.ppo_actor import TrainRayActor


class FSDPTrainRayActor(TrainRayActor):
    def init(self, args, role, with_ref=False):
        super().init(args, role, with_ref)

        raise NotImplementedError

    def sleep(self, tags):
        raise NotImplementedError

    def wake_up(self, tags):
        raise NotImplementedError

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        raise NotImplementedError

    def train(self, rollout_id, with_data_fetching=True):
        raise NotImplementedError

    def eval(self, rollout_id):
        raise NotImplementedError

    def save_model(self, iteration, with_optimizer=True):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError
