import ray


@ray.remote
class Registry:
    def __init__(self):
        self.actors = {}

    def set(self, role, key, actor):
        if role not in self.actors:
            self.actors[role] = {}
        self.actors[role][key] = actor

    def get(self, role: str, key=None):
        actors = self.actors[role]
        if key is None:
            return list(actors.values())
        return actors[key]


REGISTRY = None


def register_actor(role, key, actor):
    try:
        registry = ray.get_actor("slime_actor_registry")
    except ValueError:
        global REGISTRY
        REGISTRY = Registry.options(name="slime_actor_registry").remote()
        registry = REGISTRY
    registry.set.remote(role, key, actor)


def get_actors(role, key=None):
    registry = ray.get_actor("slime_actor_registry")
    return ray.get(registry.get.remote(role, key))
