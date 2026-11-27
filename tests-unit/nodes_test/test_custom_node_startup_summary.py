import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _import_nodes():
    try:
        import nodes
        return nodes
    except ModuleNotFoundError:
        for mod in (
            "comfy.diffusers_load",
            "comfy.samplers",
            "comfy.sample",
            "comfy.sd",
            "comfy.utils",
            "comfy.latent_formats",
            "comfy.controlnet",
            "comfy.clip_vision",
            "comfy.model_management",
            "comfy.memory_management",
            "comfy.quant_ops",
            "latent_preview",
            "node_helpers",
        ):
            if mod not in sys.modules:
                stub = MagicMock()
                stub.__path__ = []
                stub.__spec__ = None
                sys.modules[mod] = stub

        for _ in range(40):
            try:
                import nodes
                return nodes
            except ModuleNotFoundError as exc:
                missing = exc.name
                if not missing:
                    raise
                parts = missing.split(".")
                for i in range(1, len(parts) + 1):
                    pkg = ".".join(parts[:i])
                    if pkg not in sys.modules:
                        stub = MagicMock()
                        stub.__path__ = []
                        stub.__spec__ = None
                        sys.modules[pkg] = stub
        import nodes
        return nodes


nodes = _import_nodes()


class CustomNodeStartupSummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
        self._orig_display_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self._orig_loaded_dirs = dict(nodes.LOADED_MODULE_DIRS)
        self._orig_web_dirs = dict(nodes.EXTENSION_WEB_DIRS)
        self._orig_sys_path = list(sys.path)
        self._orig_modules = set(sys.modules.keys())
        self._temp_dir = tempfile.TemporaryDirectory()
        self.custom_nodes_dir = Path(self._temp_dir.name)

    def tearDown(self):
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(self._orig_class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(self._orig_display_mappings)
        nodes.LOADED_MODULE_DIRS.clear()
        nodes.LOADED_MODULE_DIRS.update(self._orig_loaded_dirs)
        nodes.EXTENSION_WEB_DIRS.clear()
        nodes.EXTENSION_WEB_DIRS.update(self._orig_web_dirs)
        sys.path[:] = self._orig_sys_path
        for mod_name in set(sys.modules.keys()) - self._orig_modules:
            sys.modules.pop(mod_name, None)
        self._temp_dir.cleanup()

    async def test_startup_summary_logs_failed_packs_without_crashing_healthy_packs(self):
        healthy_pack = self.custom_nodes_dir / "healthy_pack.py"
        healthy_pack.write_text(
            "class HealthyNode:\n"
            "    pass\n\n"
            "NODE_CLASS_MAPPINGS = {'HealthySummaryNode': HealthyNode}\n",
            encoding="utf-8",
        )

        missing_dep_pack = self.custom_nodes_dir / "broken_import_pack.py"
        missing_dep_pack.write_text(
            "import nonexistent_dependency_xyz_123\n"
            "NODE_CLASS_MAPPINGS = {}\n",
            encoding="utf-8",
        )

        empty_pack = self.custom_nodes_dir / "empty_pack.py"
        empty_pack.write_text("VALUE = 42\n", encoding="utf-8")

        v3_error_pack = self.custom_nodes_dir / "broken_v3_pack"
        v3_error_pack.mkdir()
        (v3_error_pack / "__init__.py").write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            "class BrokenExt(ComfyExtension):\n"
            "    async def on_load(self):\n"
            "        raise RuntimeError('init boom')\n"
            "    async def get_node_list(self):\n"
            "        return []\n\n"
            "async def comfy_entrypoint():\n"
            "    return BrokenExt()\n",
            encoding="utf-8",
        )

        with (
            patch.object(nodes.folder_paths, "get_folder_paths", return_value=[str(self.custom_nodes_dir)]),
            patch.object(nodes.governance, "pack_refusal", return_value=None),
            patch.object(nodes.args, "disable_all_custom_nodes", False),
            patch.object(nodes.args, "enable_manager", False),
            self.assertLogs(level=logging.WARNING) as log_ctx,
        ):
            await nodes.init_external_custom_nodes()

        self.assertIn("HealthySummaryNode", nodes.NODE_CLASS_MAPPINGS)

        joined = "\n".join(log_ctx.output)
        self.assertIn("Startup summary: 3 custom node pack(s) failed to load:", joined)
        self.assertIn("broken_import_pack.py", joined)
        self.assertIn("ModuleNotFoundError", joined)
        self.assertIn("empty_pack.py", joined)
        self.assertIn("missing NODE_CLASS_MAPPINGS or comfy_entrypoint", joined)
        self.assertIn("broken_v3_pack", joined)
        self.assertIn("RuntimeError: init boom", joined)

    async def test_startup_summary_omitted_when_all_packs_succeed(self):
        healthy_pack = self.custom_nodes_dir / "only_healthy_pack.py"
        healthy_pack.write_text(
            "class OnlyHealthyNode:\n"
            "    pass\n\n"
            "NODE_CLASS_MAPPINGS = {'OnlyHealthySummaryNode': OnlyHealthyNode}\n",
            encoding="utf-8",
        )

        with (
            patch.object(nodes.folder_paths, "get_folder_paths", return_value=[str(self.custom_nodes_dir)]),
            patch.object(nodes.governance, "pack_refusal", return_value=None),
            patch.object(nodes.args, "disable_all_custom_nodes", False),
            patch.object(nodes.args, "enable_manager", False),
            patch("nodes._log_failed_custom_nodes_summary") as summary_mock,
        ):
            await nodes.init_external_custom_nodes()

        self.assertIn("OnlyHealthySummaryNode", nodes.NODE_CLASS_MAPPINGS)
        summary_mock.assert_called_once_with([])

    def test_log_failed_custom_nodes_summary_empty_is_noop(self):
        with patch.object(nodes.logging, "warning") as warn_mock:
            nodes._log_failed_custom_nodes_summary([])
        warn_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
