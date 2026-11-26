import logging

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


def test_warns_when_node_id_is_overridden(monkeypatch, caplog):
    class Existing:
        RELATIVE_PYTHON_MODULE = "pack_a"

    class Replacement:
        pass

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "DupNode", Existing)
    with caplog.at_level(logging.WARNING):
        nodes._warn_duplicate_node_id("DupNode", Replacement, "/custom_nodes/pack_b")
    assert "DupNode" in caplog.text
    assert "pack_a" in caplog.text


def test_no_warning_for_new_or_same_node(monkeypatch, caplog):
    class Node:
        pass

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SameNode", Node)
    with caplog.at_level(logging.WARNING):
        nodes._warn_duplicate_node_id("SameNode", Node, "/custom_nodes/pack")
        nodes._warn_duplicate_node_id("BrandNewNode", Node, "/custom_nodes/pack")
    assert caplog.text == ""
