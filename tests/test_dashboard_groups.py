import json
import shutil
import subprocess
import unittest

from gnvitop.dashboard import DASHBOARD_HTML


class DashboardGroupsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is required for the embedded JS check")
    def test_embedded_script_parses(self):
        script = DASHBOARD_HTML.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        subprocess.run(
            [
                shutil.which("node"),
                "-e",
                "new Function(require('fs').readFileSync(0, 'utf8'))",
            ],
            input=script,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_group_tabs_keep_requested_order(self):
        positions = [
            DASHBOARD_HTML.index(f'data-group="{group}"')
            for group in ("all", "wsl", "ubuntu", "nvidia", "s5000", "offline")
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("gnvitop-groups", DASHBOARD_HTML)
        self.assertIn("resetGroups()", DASHBOARD_HTML)

    @unittest.skipUnless(shutil.which("node"), "node is required for the embedded JS check")
    def test_default_custom_and_offline_group_priority(self):
        start = DASHBOARD_HTML.index("// GROUP_LOGIC_START")
        end = DASHBOARD_HTML.index("// GROUP_LOGIC_END")
        logic = DASHBOARD_HTML[start:end]
        probe = r"""
const local = {alias:'localhost', is_local:true, status:'ok',
  gpus:[{name:'NVIDIA GeForce RTX 4090'}]};
const s5000 = {alias:'lab_S5000_28', is_local:false, status:'ok',
  gpus:[{name:'MTT S5000'}]};
const offline = {alias:'lab_A100', is_local:false, status:'error', gpus:[]};
const later = {alias:'lab_8x4090', is_local:false, status:'ok',
  gpus:[{name:'NVIDIA GeForce RTX 4090'}]};
const custom = {lab_A100:['wsl']};
console.log(JSON.stringify({
  local: groupsForHost(local, null),
  s5000: groupsForHost(s5000, null),
  custom: groupsForHost({...offline, status:'ok'}, custom),
  later: groupsForHost(later, custom),
  offline: groupsForHost(offline, null),
  offlinePrimary: primaryGroup(offline, groupsForHost(offline, null)),
  localPrimary: primaryGroup(local, groupsForHost(local, null)),
}));
"""
        completed = subprocess.run(
            [shutil.which("node"), "-e", logic + probe],
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["local"], ["wsl", "nvidia"])
        self.assertEqual(result["s5000"], ["ubuntu", "s5000"])
        self.assertEqual(result["custom"], ["wsl"])
        self.assertEqual(result["later"], ["ubuntu", "nvidia"])
        self.assertEqual(result["offline"], ["ubuntu", "nvidia", "offline"])
        self.assertEqual(result["offlinePrimary"], "offline")
        self.assertEqual(result["localPrimary"], "wsl")


if __name__ == "__main__":
    unittest.main()
