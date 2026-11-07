"""Submit only under a main guard; importing this file never starts a job."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlzo
from toy_project.data import samples
from toy_project.models import Classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", default="demo", help="saved cluster configuration name")
    parser.add_argument("--home", help="configuration root; otherwise MLZO_HOME or ~/.mlzo")
    parser.add_argument("--wait", action="store_true", help="wait for the remote result")
    args = parser.parse_args()

    cluster = mlzo.cluster(args.cluster, home=args.home)
    run = mlzo.train(
        mlzo.Lazy(Classifier), mlzo.data(samples).shuffle(seed=17).batch(16),
        loss="cross_entropy", optimizer=mlzo.opt("sgd", lr=0.1),
        device="cpu", precision="fp32", seed=17,
    )
    print(run.plan(on=cluster).explain())
    run.fit(epochs=2, on=cluster, code=str(Path(__file__).resolve().parent), wait=False)
    print("Run ID:", run.id)
    if args.wait:
        run.wait()
        print(run.status().to_dict())


if __name__ == "__main__":
    main()
