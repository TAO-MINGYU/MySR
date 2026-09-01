import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
JULIAPKG_JSON = REPO_ROOT / "mysr" / "juliapkg.json"
DEV_CONFIG_SCRIPT = REPO_ROOT / "mysr" / "test" / "generate_dev_juliapkg.py"
MYSRCORE_UUID = "65629769-74ff-4293-a07c-ae7483a03f6e"
MYSRCORE_URL = "https://github.com/TAO-MINGYU/MySRCore.jl"


class TestJuliaPkgConfig(unittest.TestCase):
    def test_release_config_pins_mysrcore_github_tag(self):
        config = json.loads(JULIAPKG_JSON.read_text(encoding="utf-8"))
        package = config["packages"]["MySRCore"]

        self.assertEqual(package["uuid"], MYSRCORE_UUID)
        self.assertEqual(package["url"], MYSRCORE_URL)
        self.assertEqual(package["rev"], "v1.0.0")
        self.assertEqual(package["preferences"], {"precompile_float64": False})
        self.assertNotIn("path", package)
        self.assertNotIn("dev", package)

    def test_dev_script_replaces_release_source_with_local_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "juliapkg.json"
            config_path.write_text(
                JULIAPKG_JSON.read_text(encoding="utf-8"), encoding="utf-8"
            )
            backend_path = "/workspace/MySRCore.jl"

            subprocess.run(
                [
                    sys.executable,
                    str(DEV_CONFIG_SCRIPT),
                    str(config_path),
                    backend_path,
                ],
                check=True,
            )

            package = json.loads(config_path.read_text(encoding="utf-8"))["packages"][
                "MySRCore"
            ]

        self.assertEqual(package["uuid"], MYSRCORE_UUID)
        self.assertEqual(package["path"], backend_path)
        self.assertIs(package["dev"], True)
        self.assertEqual(package["preferences"], {"precompile_float64": False})
        self.assertNotIn("url", package)
        self.assertNotIn("rev", package)


if __name__ == "__main__":
    unittest.main()
