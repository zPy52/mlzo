# Preparing an alpha release

The release version is read from `src/mlzo/_version.py`, currently `0.1.0a1`. Hatchling reads that file as version metadata without importing MLZO or PyTorch. The wheel uses the `src/mlzo` layout, includes `py.typed`, and installs the `mlzo` console command. The source distribution also includes documentation and Python examples.

## Owner decisions before publication

- The repository's `LICENSE` contains MIT terms. Package metadata declares MIT and includes that file in both distributions. Preserve the license and copyright notice when publishing.
- Check the intended `mlzo` name on TestPyPI and PyPI and confirm ownership or availability. This repository does not establish that the name is available.
- Confirm `0.1.0a1` is the desired, unused pre-release version on each target index. Change `_version.py` for a new release; do not replace an already uploaded version's files. Update the changelog and version-specific docs with that decision.
- Add author/contact/project URLs only if the owner supplies and approves them. None are inferred here.

These decisions precede any upload. They do not require installing or publishing the package to review the implementation.

## Build and inspect

Use a Python 3.12+ virtual environment. Install release tools when ready, then build from a clean checkout containing the intended release files:

```sh
python -m pip install '.[dev]'
python -m build
python -m twine check dist/*
```

Use an empty `dist` directory so the wildcard cannot include older releases. Inspect both the wheel ZIP and source tarball before upload:

```sh
python -m zipfile -l dist/mlzo-0.1.0a1-py3-none-any.whl
python -m tarfile -l dist/mlzo-0.1.0a1.tar.gz
```

Confirm the wheel has `mlzo/py.typed`, runtime modules, and console entrypoint metadata. Confirm the source archive has useful examples, usage/support/publishing docs, the contract, and the original future plan. Build configuration excludes common secret, cache, virtualenv, token, checkpoint, and temporary-state paths, and limits source-distribution additions to declared paths. Review archive contents yourself; exclusion patterns are not a secret scanner. No training factory, device discovery, network client, or optional adapter should run during metadata/build.

In a separate clean virtual environment, do only basic installation and smoke checks:

```sh
python -m pip install dist/mlzo-0.1.0a1-py3-none-any.whl
python -m mlzo --version
mlzo --help
python -c 'import mlzo; print(mlzo.__version__)'
python examples/local_classifier.py --output /tmp/mlzo-alpha-example
```

Run the example from the unpacked source tree after wheel installation. Choose a fresh output directory. These commands exercise a tiny CPU path; they do not validate GPUs, TLS, remote execution, every dependency version, or optional services. This project deliberately adds no comprehensive test suite or CI workflow.

## Optional index uploads by the owner

No upload has been performed. Once the decisions and checks above are complete, the owner may upload the reviewed files to TestPyPI first:

```sh
python -m twine upload --repository testpypi dist/mlzo-0.1.0a1-py3-none-any.whl dist/mlzo-0.1.0a1.tar.gz
```

TestPyPI may not have all runtime dependencies. Prepare those separately from a trusted source, then check the candidate in another clean environment without resolving dependencies from TestPyPI:

```sh
python -m pip install --pre --no-deps --index-url https://test.pypi.org/simple/ mlzo==0.1.0a1
```

Repeat the version/help/import and tiny CPU checks. After the owner's release decision, upload the same reviewed files to PyPI:

```sh
python -m twine upload dist/mlzo-0.1.0a1-py3-none-any.whl dist/mlzo-0.1.0a1.tar.gz
```

Use the index's credential prompt or protected credential configuration. Never put tokens in repository files, shell examples, build artifacts, or logs. Installing from an index is a later owner action and is not assumed by the local installation instructions.
