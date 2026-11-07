"""Execute complete documented CPU examples in isolated subprocesses."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

DOCS = Path(__file__).resolve().parents[1] / "src/content/docs"


def main():
    for name in ("start/quickstart.md", "start/tensors-and-data.md", "guides/custom-training.md", "guides/transforms.md", "guides/data.md"):
        # Optional image inputs and multiprocessing scripts use python no-exec.
        code = "\n\n".join(re.findall(r"^```python\n(.*?)^```", (DOCS / name).read_text(), re.M | re.S))
        if not code:
            raise SystemExit(f"No Python example found in {name}")
        with tempfile.TemporaryDirectory(prefix="mlzo-doc-example-") as temporary:
            environment = {**os.environ, "MLZO_HOME": temporary + "/home", "TMPDIR": temporary,
                           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            for secret in ("MLZO_TOKEN", "MLZO_SUBMITTER_TOKEN", "MLZO_INVITE_TOKEN"):
                environment.pop(secret, None)
            result = subprocess.run([sys.executable, "-c", code], cwd=temporary, env=environment,
                                    capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise SystemExit(f"{name} failed:\n{result.stdout}\n{result.stderr}")
            print(f"Passed {name}")


if __name__ == "__main__":
    main()
