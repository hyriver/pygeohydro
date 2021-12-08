import shutil
from pathlib import Path

import nox

HR_DEPS = ["async_retriever", "pygeoogc", "pygeoutils", "pynhd"]


@nox.session(python="3.9")
def tests(session):
    install_deps(session)

    session.run("pytest")
    session.run("coverage", "report")
    session.run("coverage", "html")


def install_deps(session):
    deps = [".[test]"] + [f"git+https://github.com/cheginit/{p}.git" for p in HR_DEPS]
    session.install(*deps)
    dirs = [".pytest_cache", "build", "dist", ".eggs"]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    patters = ["*.egg-info", "*.egg", "*.pyc", "*~", "**/__pycache__"]
    for p in patters:
        for f in Path.cwd().rglob(p):
            shutil.rmtree(f, ignore_errors=True)
