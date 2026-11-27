import sys

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


@pytest.fixture(autouse=True)
def _clean_failures():
    nodes.CUSTOM_NODE_LOAD_FAILURES.clear()
    yield
    nodes.CUSTOM_NODE_LOAD_FAILURES.clear()
    for name in ("broken_pack", "empty_pack"):
        sys.modules.pop(name, None)


@pytest.mark.asyncio
async def test_failed_loads_record_pack_and_reason(tmp_path, monkeypatch):
    broken = tmp_path / "broken_pack.py"
    broken.write_text("import module_that_does_not_exist_xyz\n")
    empty = tmp_path / "empty_pack.py"
    empty.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert not await nodes.load_custom_node(str(broken), module_parent="tests")
    assert not await nodes.load_custom_node(str(empty), module_parent="tests")

    failures = nodes.CUSTOM_NODE_LOAD_FAILURES
    assert "ModuleNotFoundError" in failures[str(broken)]
    assert "module_that_does_not_exist_xyz" in failures[str(broken)]
    assert failures[str(empty)] == "no NODE_CLASS_MAPPINGS or comfy_entrypoint"


@pytest.mark.asyncio
async def test_successful_load_records_nothing(tmp_path, monkeypatch):
    ok = tmp_path / "ok_pack.py"
    ok.write_text("NODE_CLASS_MAPPINGS = {}\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert await nodes.load_custom_node(str(ok), module_parent="tests")
    assert nodes.CUSTOM_NODE_LOAD_FAILURES == {}
    sys.modules.pop("ok_pack", None)


def test_summary_lists_pack_name_and_reason():
    lines = nodes.custom_node_failure_summary({
        "/custom_nodes/zeta-pack": "ImportError: no torchaudio",
        "/custom_nodes/alpha_pack.py": "no NODE_CLASS_MAPPINGS or comfy_entrypoint",
    })
    assert lines[0] == "2 custom node pack(s) failed to load:"
    assert lines[1] == "  alpha_pack: no NODE_CLASS_MAPPINGS or comfy_entrypoint"
    assert lines[2] == "  zeta-pack: ImportError: no torchaudio"


def test_summary_empty_when_nothing_failed():
    assert nodes.custom_node_failure_summary({}) == []
