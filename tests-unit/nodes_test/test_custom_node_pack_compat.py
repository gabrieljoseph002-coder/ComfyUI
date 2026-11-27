"""Compatibility checks for the two custom node pack interfaces.

V1: a module-level NODE_CLASS_MAPPINGS (+ optional NODE_DISPLAY_NAME_MAPPINGS).
V3: a comfy_entrypoint() returning a ComfyExtension whose get_node_list()
returns node classes with GET_SCHEMA().

Covers registration, display names, duplicate IDs and the ignore list
(built-in names a pack must not override).
"""
import sys
import textwrap

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


PACK_NAMES = ("compat_v1_pack", "compat_v1_pack_b", "compat_v3_pack", "compat_v3_sync_pack", "compat_bad_entrypoint")


@pytest.fixture(autouse=True)
def _restore_node_mappings():
    class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    try:
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
        for name in PACK_NAMES:
            sys.modules.pop(name, None)


def _write(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source))
    return str(path)


def _v1_pack(tmp_path, name, node_id, display_name):
    return _write(tmp_path, name, f"""
        class Node:
            TAG = {name!r}

        NODE_CLASS_MAPPINGS = {{{node_id!r}: Node}}
        NODE_DISPLAY_NAME_MAPPINGS = {{{node_id!r}: {display_name!r}}}
    """)


def _v3_pack(tmp_path, name, node_ids, sync=False):
    node_defs = "\n".join(
        f"""
        class Node{i}:
            @classmethod
            def GET_SCHEMA(cls):
                class Schema:
                    node_id = {node_id!r}
                    display_name = {node_id + ' Display'!r}
                return Schema()
        """
        for i, node_id in enumerate(node_ids)
    )
    node_list = ", ".join(f"Node{i}" for i in range(len(node_ids)))
    entry = "def" if sync else "async def"
    return _write(tmp_path, name, f"""
        from comfy_api.latest import ComfyExtension
        {node_defs}

        class Extension(ComfyExtension):
            async def get_node_list(self):
                return [{node_list}]

        {entry} comfy_entrypoint():
            return Extension()
    """)


@pytest.fixture
def packs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_v1_pack_registers_node_and_display_name(packs):
    path = _v1_pack(packs, "compat_v1_pack", "CompatV1Node", "Compat V1")

    assert await nodes.load_custom_node(path, module_parent="tests")

    node_cls = nodes.NODE_CLASS_MAPPINGS["CompatV1Node"]
    assert node_cls.TAG == "compat_v1_pack"
    assert node_cls.RELATIVE_PYTHON_MODULE == "tests.compat_v1_pack"
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["CompatV1Node"] == "Compat V1"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_v3_pack_registers_every_node_with_display_name(packs, sync):
    name = "compat_v3_sync_pack" if sync else "compat_v3_pack"
    path = _v3_pack(packs, name, ["CompatV3A", "CompatV3B"], sync=sync)

    assert await nodes.load_custom_node(path, module_parent="tests")

    for node_id in ("CompatV3A", "CompatV3B"):
        assert nodes.NODE_CLASS_MAPPINGS[node_id].RELATIVE_PYTHON_MODULE == f"tests.{name}"
        assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == f"{node_id} Display"


@pytest.mark.asyncio
async def test_duplicate_id_last_loaded_pack_wins(packs):
    first = _v1_pack(packs, "compat_v1_pack", "CompatDup", "First")
    second = _v1_pack(packs, "compat_v1_pack_b", "CompatDup", "Second")

    assert await nodes.load_custom_node(first, module_parent="tests")
    assert await nodes.load_custom_node(second, module_parent="tests")

    assert nodes.NODE_CLASS_MAPPINGS["CompatDup"].TAG == "compat_v1_pack_b"
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["CompatDup"] == "Second"


@pytest.mark.asyncio
async def test_ignored_ids_are_not_overridden_v1_and_v3(packs):
    builtin = object()
    nodes.NODE_CLASS_MAPPINGS["CompatBuiltin"] = builtin
    v1 = _v1_pack(packs, "compat_v1_pack", "CompatBuiltin", "Hijack")
    v3 = _v3_pack(packs, "compat_v3_pack", ["CompatBuiltin", "CompatV3Fresh"])

    assert await nodes.load_custom_node(v1, ignore={"CompatBuiltin"}, module_parent="tests")
    assert await nodes.load_custom_node(v3, ignore={"CompatBuiltin"}, module_parent="tests")

    assert nodes.NODE_CLASS_MAPPINGS["CompatBuiltin"] is builtin
    assert "CompatBuiltin" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "CompatV3Fresh" in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_same_id_twice_in_a_v3_pack_registers_once(packs):
    path = _v3_pack(packs, "compat_v3_pack", ["CompatSame", "CompatSame"])

    assert await nodes.load_custom_node(path, module_parent="tests")

    ids = [k for k in nodes.NODE_CLASS_MAPPINGS if k == "CompatSame"]
    assert ids == ["CompatSame"]


@pytest.mark.asyncio
async def test_entrypoint_not_returning_extension_is_rejected(packs):
    path = _write(packs, "compat_bad_entrypoint", """
        def comfy_entrypoint():
            return object()
    """)

    assert not await nodes.load_custom_node(path, module_parent="tests")
