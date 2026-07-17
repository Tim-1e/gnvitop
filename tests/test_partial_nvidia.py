import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from gnvitop.server import COMBINED_CMD, _build_gpus, _parse_combined_output


class PartialNvidiaFailureTest(unittest.TestCase):
    def _run_with_query_outputs(self, index_script, gpu_script=""):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_bytes(
                ("""#!/usr/bin/env bash
case "$1" in
  --query-gpu=index)
""" + index_script + """
    ;;
  --query-gpu=*)
""" + gpu_script + """
    ;;
  pmon)
    echo '# gpu pid type sm mem enc dec command'
    echo '# Idx # # # # # # name'
    ;;
esac
""").encode("utf-8")
            )
            nvidia_smi.chmod(0o755)
            mx_smi = fake_bin / "mx-smi"
            mx_smi.write_bytes(
                b"#!/usr/bin/env bash\n"
                b"echo '| 0       MetaX C500  Off | 0000:0e:00.0 | 0%  Native |'\n"
                b"echo '| 36C  56W / 350W  P0     | 858/65536 MiB | Available  |'\n"
            )
            mx_smi.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

            completed = subprocess.run(
                ["bash", "-c", COMBINED_CMD],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            return _parse_combined_output(completed.stdout.strip())

    def test_keeps_healthy_gpus_when_one_nvidia_gpu_is_lost(self):
        gpu_part, _, vendor = self._run_with_query_outputs(
            """    for i in 0 1 2 3 4 5 7; do echo "$i"; done
    echo 'Unable to determine the device handle for GPU6' >&2""",
            """    for i in 0 1 2 3 4 5 7; do
      echo "$i, NVIDIA GeForce RTX 4090, 24564, 5, 24559, 0, 27"
    done
    echo 'Unable to determine the device handle for GPU6' >&2"""
        )
        gpus = _build_gpus(gpu_part)

        self.assertEqual(vendor, "nvidia")
        self.assertEqual([gpu["index"] for gpu in gpus], [0, 1, 2, 3, 4, 5, 7])

    def test_falls_back_to_mx_for_malformed_nvidia_output(self):
        gpu_part, _, vendor = self._run_with_query_outputs(
            "    echo 'No devices were found'"
        )

        self.assertEqual(vendor, "mx")
        self.assertIn("MetaX C500", gpu_part)


if __name__ == "__main__":
    unittest.main()
