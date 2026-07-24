import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gnvitop.server import (
    COMBINED_CMD,
    _attach_processes,
    _build_mthreads_gpus,
    _parse_combined_output,
    _run_openssh_query,
)


MTHREADS_OVERVIEW = """\
Fri Jul 24 05:58:43 2026
---------------------------------------------------------------------
    mthreads-gmi:2.3.2           Driver Version:3.3.5-server
---------------------------------------------------------------------
ID   Name                 |PCIe                |%GPU  Mem
     Device Type   Perf   |Pcie Lane Width     |Temp  MPC Capable
+-------------------------------------------------------------------+
0    MTT S5000            |00000000:2a:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |40C   YES
+-------------------------------------------------------------------+
1    MTT S5000            |00000000:3a:00.0    |75%   8192MiB(81920MiB)
     Physical      P0     |16x(16x)            |39C   YES
+-------------------------------------------------------------------+
Processes:
ID   PID           Process name                           GPU Memory
                                                               Usage
+-------------------------------------------------------------------+
1 4242 python worker.py 2048MiB
---------------------------------------------------------------------
"""


class MthreadsProviderTest(unittest.TestCase):
    def test_parses_s5000_overview_into_existing_gpu_contract(self):
        gpus = _build_mthreads_gpus(MTHREADS_OVERVIEW)

        self.assertEqual([gpu["index"] for gpu in gpus], [0, 1])
        self.assertEqual(gpus[0]["name"], "MTT S5000")
        self.assertEqual(gpus[0]["temperature_c"], 40.0)
        self.assertEqual(gpus[1]["memory_total_mb"], 81920.0)
        self.assertEqual(gpus[1]["memory_used_mb"], 8192.0)
        self.assertEqual(gpus[1]["memory_free_mb"], 73728.0)
        self.assertEqual(gpus[1]["memory_usage_pct"], 10.0)
        self.assertEqual(gpus[1]["gpu_utilization_pct"], 75.0)

    def test_detects_mthreads_and_normalizes_process_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text(
                "#!/usr/bin/env bash\necho 'No devices were found'\n",
                encoding="utf-8",
            )
            nvidia_smi.chmod(0o755)

            mthreads_gmi = fake_bin / "mthreads-gmi"
            mthreads_gmi.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<EOF\n"
                + MTHREADS_OVERVIEW.replace(
                    "1 4242 python worker.py 2048MiB",
                    f"1 {os.getpid()} python worker.py 2048 MiB",
                )
                + "EOF\n",
                encoding="utf-8",
            )
            mthreads_gmi.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            completed = subprocess.run(
                ["bash", "-c", COMBINED_CMD],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

        gpu_part, proc_part, vendor = _parse_combined_output(completed.stdout.strip())
        gpus = _build_mthreads_gpus(gpu_part)
        _attach_processes(gpus, proc_part)

        self.assertEqual(vendor, "mthreads")
        self.assertEqual(len(gpus), 2)
        self.assertEqual(len(gpus[1]["processes"]), 1)
        self.assertEqual(gpus[1]["processes"][0]["gpu_memory_mb"], 2048.0)
        self.assertNotEqual(gpus[1]["processes"][0]["user"], "unknown")

    @patch("gnvitop.server.subprocess.run")
    def test_openssh_fallback_uses_argument_list_without_shell(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="MTHREADS\nstats", stderr=""
        )

        output = _run_openssh_query("lab_S5000_28")

        command = run.call_args.args[0]
        self.assertEqual(output, "MTHREADS\nstats")
        self.assertEqual(command[0], "ssh")
        self.assertIn("lab_S5000_28", command)
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
