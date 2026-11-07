# Synthetic remote project

Install MLZO on the submitter and worker first. Configure a coordinator and enroll a CPU worker using [the usage guide](../../docs/usage.md#start-a-coordinator-and-worker). Keep both services running. From the MLZO repository root, submit:

```sh
python examples/remote_project/submit.py --cluster demo --wait
```

The script snapshots this directory as `code=`, so `toy_project.models.Classifier` and `toy_project.data.samples` have the same import paths on the worker. The worker's job environment imports them from the extracted project directory. This example has no extra dependency declarations; it uses the worker's installed MLZO and PyTorch, generates 64 synthetic CPU samples there, and requests two epochs. It does not download data or provision cloud resources.

Omit `--wait` to return after durable submission. Use the printed ID with `mlzo run status RUN_ID --cluster demo` or reconnect in Python with `mlzo.run(RUN_ID, on=mlzo.cluster("demo"))`.

For your own project, keep constructors and data factories in importable modules rather than `__main__`, closures, or notebook cells. Use a zero-argument importable training factory for custom `Training` algorithms. Remote submission rejects eager models and custom `Training` instances. Project dependencies, when declared, are prepared on the worker; host virtual environments are never shipped.
