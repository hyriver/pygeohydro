import shutil
from pathlib import Path

import nox


@nox.session(python="3.10")
def tests(session):
    session.install(".[test]")
    hr_deps = ["async_retriever", "pygeoogc", "pygeoutils", "pynhd"]
    for p in hr_deps:
        session.install(f"git+https://github.com/cheginit/{p}.git")
    session.run("pytest")
    session.run("coverage", "report")
    session.run("coverage", "html")

    dirs = [".pytest_cache", "build", "dist", ".eggs"]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    patters = ["*.egg-info", "*.egg", "*.pyc", "*~", "**/__pycache__"]
    for p in patters:
        for f in Path.cwd().rglob(p):
            shutil.rmtree(f, ignore_errors=True)
