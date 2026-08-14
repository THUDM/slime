from abc import ABC, abstractmethod
from collections.abc import Sequence

from slime.utils.types import ParamInfo


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        return HfWeightIteratorDirect(args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config, transform_ue8m0=True):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.transform_ue8m0 = transform_ue8m0

    @abstractmethod
    def get_hf_weight_chunks(
        self,
        megatron_local_weights,
        progress_desc: str = "Update weights",
        param_info_buckets: Sequence[Sequence[ParamInfo]] | None = None,
    ):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
