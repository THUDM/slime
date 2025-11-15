from abc import ABC
from collections import defaultdict
from typing import Callable, Dict, Iterable, Tuple

import torch

_SourceGetter = Callable[[], Iterable[Tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag):
        return _TensorBackuperNormal(source_getter=source_getter)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    def backup_tags(self):
        raise NotImplementedError

    def get(self, tag: str):
        raise NotImplementedError

    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    def restore(self, tag: str):
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()
