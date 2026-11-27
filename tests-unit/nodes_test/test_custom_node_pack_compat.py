import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


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


class CustomNodePackCompatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
        self._orig_display_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self._orig_loaded_dirs = dict(nodes.LOADED_MODULE_DIRS)
        self._orig_web_dirs = dict(nodes.EXTENSION_WEB_DIRS)
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)
        self._added_modules = []

    def tearDown(self):
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(self._orig_class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(self._orig_display_mappings)
        nodes.LOADED_MODULE_DIRS.clear()
        nodes.LOADED_MODULE_DIRS.update(self._orig_loaded_dirs)
        nodes.EXTENSION_WEB_DIRS.clear()
        nodes.EXTENSION_WEB_DIRS.update(self._orig_web_dirs)
        for mod_name in self._added_modules:
            sys.modules.pop(mod_name, None)
        self._temp_dir.cleanup()

    def _track_module(self, path: Path) -> str:
        mod_name = str(path) if path.is_dir() else str(path.with_suffix(""))
        sys_mod_name = mod_name.replace(".", "_x_") if path.is_dir() else mod_name
        self._added_modules.append(sys_mod_name)
        return str(path)

    async def test_v1_single_file_pack_registration_and_display_names(self):
        pack = self.tmp_path / "compat_v1_single.py"
        pack.write_text(
            "class AlphaNode:\n"
            "    pass\n\n"
            "class BetaNode:\n"
            "    pass\n\n"
            "NODE_CLASS_MAPPINGS = {'CompatAlpha': AlphaNode, 'CompatBeta': BetaNode}\n"
            "NODE_DISPLAY_NAME_MAPPINGS = {'CompatAlpha': 'Compat Alpha Display'}\n",
            encoding="utf-8",
        )

        loaded = await nodes.load_custom_node(self._track_module(pack))
        self.assertTrue(loaded)
        self.assertIs(nodes.NODE_CLASS_MAPPINGS["CompatAlpha"].__name__, "AlphaNode")
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["CompatAlpha"].RELATIVE_PYTHON_MODULE, "custom_nodes.compat_v1_single")
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["CompatAlpha"], "Compat Alpha Display")
        self.assertIn("CompatBeta", nodes.NODE_CLASS_MAPPINGS)
        self.assertNotIn("CompatBeta", nodes.NODE_DISPLAY_NAME_MAPPINGS)

    async def test_v1_package_directory_without_display_names(self):
        pack_dir = self.tmp_path / "compat_v1_pkg"
        pack_dir.mkdir()
        (pack_dir / "__init__.py").write_text(
            "class PkgNode:\n"
            "    pass\n\n"
            "NODE_CLASS_MAPPINGS = {'CompatPkgNode': PkgNode}\n",
            encoding="utf-8",
        )

        loaded = await nodes.load_custom_node(self._track_module(pack_dir))
        self.assertTrue(loaded)
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["CompatPkgNode"].RELATIVE_PYTHON_MODULE, "custom_nodes.compat_v1_pkg")
        self.assertNotIn("CompatPkgNode", nodes.NODE_DISPLAY_NAME_MAPPINGS)

    async def test_v3_async_and_sync_entrypoint_registration_and_display_names(self):
        v3_async = self.tmp_path / "compat_v3_async.py"
        v3_async.write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            "class V3AsyncNode:\n"
            "    @classmethod\n"
            "    def GET_SCHEMA(cls):\n"
            "        class Schema:\n"
            "            node_id = 'CompatV3Async'\n"
            "            display_name = 'Compat V3 Async'\n"
            "        return Schema()\n\n"
            "class V3NoDisplayNode:\n"
            "    @classmethod\n"
            "    def GET_SCHEMA(cls):\n"
            "        class Schema:\n"
            "            node_id = 'CompatV3NoDisplay'\n"
            "            display_name = None\n"
            "        return Schema()\n\n"
            "class AsyncExt(ComfyExtension):\n"
            "    loaded = False\n"
            "    async def on_load(self):\n"
            "        AsyncExt.loaded = True\n"
            "    async def get_node_list(self):\n"
            "        assert AsyncExt.loaded\n"
            "        return [V3AsyncNode, V3NoDisplayNode]\n\n"
            "async def comfy_entrypoint():\n"
            "    return AsyncExt()\n",
            encoding="utf-8",
        )

        v3_sync = self.tmp_path / "compat_v3_sync.py"
        v3_sync.write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            "class V3SyncNode:\n"
            "    @classmethod\n"
            "    def GET_SCHEMA(cls):\n"
            "        class Schema:\n"
            "            node_id = 'CompatV3Sync'\n"
            "            display_name = 'Compat V3 Sync'\n"
            "        return Schema()\n\n"
            "class SyncExt(ComfyExtension):\n"
            "    async def get_node_list(self):\n"
            "        return [V3SyncNode]\n\n"
            "def comfy_entrypoint():\n"
            "    return SyncExt()\n",
            encoding="utf-8",
        )

        self.assertTrue(await nodes.load_custom_node(self._track_module(v3_async)))
        self.assertTrue(await nodes.load_custom_node(self._track_module(v3_sync)))

        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["CompatV3Async"].RELATIVE_PYTHON_MODULE, "custom_nodes.compat_v3_async")
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["CompatV3Async"], "Compat V3 Async")
        self.assertIn("CompatV3NoDisplay", nodes.NODE_CLASS_MAPPINGS)
        self.assertNotIn("CompatV3NoDisplay", nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["CompatV3Sync"].RELATIVE_PYTHON_MODULE, "custom_nodes.compat_v3_sync")
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["CompatV3Sync"], "Compat V3 Sync")

    async def test_duplicate_node_ids_and_ignore_set_across_v1_and_v3(self):
        pack_v1 = self.tmp_path / "compat_dup_v1.py"
        pack_v1.write_text(
            "class FirstNode:\n"
            "    pass\n\n"
            "NODE_CLASS_MAPPINGS = {'SharedCompatNode': FirstNode, 'IgnoredCompatNode': FirstNode}\n"
            "NODE_DISPLAY_NAME_MAPPINGS = {'SharedCompatNode': 'First Display', 'IgnoredCompatNode': 'Ignored Display'}\n",
            encoding="utf-8",
        )

        pack_v3 = self.tmp_path / "compat_dup_v3.py"
        pack_v3.write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            "class OverrideNode:\n"
            "    @classmethod\n"
            "    def GET_SCHEMA(cls):\n"
            "        class Schema:\n"
            "            node_id = 'SharedCompatNode'\n"
            "            display_name = 'Second V3 Display'\n"
            "        return Schema()\n\n"
            "class Ext(ComfyExtension):\n"
            "    async def get_node_list(self):\n"
            "        return [OverrideNode]\n\n"
            "async def comfy_entrypoint():\n"
            "    return Ext()\n",
            encoding="utf-8",
        )

        self.assertTrue(await nodes.load_custom_node(self._track_module(pack_v1), ignore={"IgnoredCompatNode"}))
        self.assertNotIn("IgnoredCompatNode", nodes.NODE_CLASS_MAPPINGS)
        self.assertNotIn("IgnoredCompatNode", nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["SharedCompatNode"], "First Display")

        self.assertTrue(await nodes.load_custom_node(self._track_module(pack_v3), ignore={"IgnoredCompatNode"}))
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["SharedCompatNode"].__name__, "OverrideNode")
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS["SharedCompatNode"].RELATIVE_PYTHON_MODULE, "custom_nodes.compat_dup_v3")
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["SharedCompatNode"], "Second V3 Display")

    async def test_invalid_v3_entrypoints_return_false(self):
        non_callable = self.tmp_path / "compat_v3_non_callable.py"
        non_callable.write_text("comfy_entrypoint = 123\n", encoding="utf-8")

        bad_return = self.tmp_path / "compat_v3_bad_return.py"
        bad_return.write_text("async def comfy_entrypoint():\n    return object()\n", encoding="utf-8")

        bad_node_list = self.tmp_path / "compat_v3_bad_list.py"
        bad_node_list.write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            "class BadListExt(ComfyExtension):\n"
            "    async def get_node_list(self):\n"
            "        return 'not-a-list'\n\n"
            "async def comfy_entrypoint():\n"
            "    return BadListExt()\n",
            encoding="utf-8",
        )

        self.assertFalse(await nodes.load_custom_node(self._track_module(non_callable)))
        self.assertFalse(await nodes.load_custom_node(self._track_module(bad_return)))
        self.assertFalse(await nodes.load_custom_node(self._track_module(bad_node_list)))


if __name__ == "__main__":
    unittest.main()
