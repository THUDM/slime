import re

from .common import strip_mcore_wrappers
from .deepseek import deepseek_hf_tensor


class _LanguageModelReader:
    """Translate the conditional-generation checkpoint's text prefix."""

    def __init__(self, reader):
        self.reader = reader

    def __contains__(self, name):
        return name in self.reader or self._name(name) in self.reader

    def _name(self, name):
        return name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name

    def get_tensor(self, name):
        return self.reader.get_tensor(name if name in self.reader else self._name(name))


def glm5_next_hf_tensor(name, reader, config):
    name = strip_mcore_wrappers(name)
    if name.startswith("model.visual."):
        return reader.get_tensor(name)
    reader = _LanguageModelReader(reader)
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer, rest = match.groups()
        if rest.startswith("self_attention."):
            return reader.get_tensor(f"model.layers.{layer}.self_attn.{rest.removeprefix('self_attention.')}")
        if rest.startswith("hc_"):
            return reader.get_tensor(f"model.layers.{layer}.{rest}")
    return deepseek_hf_tensor(name, reader, config.text_config)
