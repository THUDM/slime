from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
@pytest.mark.parametrize("patch_version", ["latest", "v0.5.15.post1"])
def test_post_process_weights_endpoint_reads_json_body(patch_version):
    patch = (REPO_ROOT / "docker" / "patch" / patch_version / "sglang.patch").read_text()

    assert "+async def post_process_weights(" in patch
    assert "req: Annotated[PostProcessWeightsReqInput, Body()]" in patch
    assert "req: PostProcessWeightsReqInput, request: Request" not in patch
