class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner
