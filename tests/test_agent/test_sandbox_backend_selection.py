"""CPU tests for ``slime.agent.sandbox.make_sandbox`` backend selection.

``SLIME_AGENT_SANDBOX_BACKEND`` names a registered backend (default ``e2b``) or
the dotted import path of an out-of-tree class; these tests pin both forms and
the errors a misconfiguration raises.
"""

from __future__ import annotations

import sys
import types

import pytest

from slime.agent.sandbox import SANDBOX_BACKEND_ENV, E2BSandbox, make_sandbox, sandbox_backend

NUM_GPUS = 0


class _ExternalSandbox:
    def __init__(self, image: str) -> None:
        self.image = image
        self.sandbox_id = ""


@pytest.fixture
def external_module(monkeypatch):
    mod = types.ModuleType("fake_vendor.sandboxes")
    mod.ExternalSandbox = _ExternalSandbox
    monkeypatch.setitem(sys.modules, "fake_vendor", types.ModuleType("fake_vendor"))
    monkeypatch.setitem(sys.modules, "fake_vendor.sandboxes", mod)
    return mod


@pytest.mark.unit
def test_default_is_e2b(monkeypatch):
    monkeypatch.delenv(SANDBOX_BACKEND_ENV, raising=False)
    sb = make_sandbox("img")
    assert isinstance(sb, E2BSandbox)
    assert sb.image == "img"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["e2b", "E2B", " e2b "])
def test_registered_name_is_case_and_space_insensitive(monkeypatch, value):
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, value)
    assert sandbox_backend() is E2BSandbox


@pytest.mark.unit
def test_dotted_path_loads_out_of_tree_class(monkeypatch, external_module):
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, "fake_vendor.sandboxes.ExternalSandbox")
    sb = make_sandbox("registry/swe:tag")
    assert isinstance(sb, _ExternalSandbox)
    assert sb.image == "registry/swe:tag"


@pytest.mark.unit
def test_unknown_bare_name_raises_value_error(monkeypatch):
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, "docker")
    with pytest.raises(ValueError, match="docker"):
        sandbox_backend()


@pytest.mark.unit
def test_missing_module_or_attribute_raises(monkeypatch, external_module):
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, "no_such_pkg_for_slime_tests.Sandbox")
    with pytest.raises(ModuleNotFoundError):
        sandbox_backend()
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, "fake_vendor.sandboxes.Missing")
    with pytest.raises(AttributeError):
        sandbox_backend()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
