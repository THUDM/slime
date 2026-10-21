"""Optional straw tensor references used by rollout and training."""

try:
    from straw.tensor import TensorRef
except ModuleNotFoundError as error:
    if error.name != "straw":
        raise

    class TensorRef:
        """Keep in-memory paths importable when straw is not installed."""

        def __new__(cls, *args, **kwargs):
            raise ModuleNotFoundError("Install straw with: pip install straw-queue", name="straw")


def materialize_tensor_refs(value):
    """Make debug dumps self-contained, independent of queue ownership and GC."""
    if isinstance(value, TensorRef):
        return value.load()
    if isinstance(value, dict):
        return {key: materialize_tensor_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize_tensor_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_tensor_refs(item) for item in value)
    return value
