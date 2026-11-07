import json
import os
import stat
import subprocess
import sys


def test_cli_configures_loopback_and_rejects_insecure_endpoint_without_leaking_token(tmp_path):
    secret = "test-only-credential-not-a-real-secret"
    home = tmp_path / "cli-home"

    def create(endpoint):
        return subprocess.run(
            [sys.executable, "-m", "mlzo", "cluster", "create", "local", "--json",
             "--home", str(home), "--endpoint", endpoint],
            env={**os.environ, "MLZO_TOKEN": secret}, capture_output=True, text=True, timeout=15,
        )

    success = create("http://127.0.0.1:9")
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)["result"]["configured"] is True
    config = home / "clusters" / "local.json"
    original = config.read_bytes()
    assert json.loads(original)["token"] == secret
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    failure = create("http://192.0.2.1:9")
    assert failure.returncode == 2
    assert json.loads(failure.stdout)["ok"] is False
    assert config.read_bytes() == original
    for result in (success, failure):
        assert secret not in result.stdout + result.stderr
