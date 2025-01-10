"""Nox sessions."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import nox

try:
    import tomllib as tomli
except ImportError:
    import tomli


def get_package_name() -> str:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        return tomli.load(f)["project"]["name"]


def get_extras() -> list[str]:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        extras = tomli.load(f)["project"]["optional-dependencies"]
    return [e for e in extras if e not in ("test", "typeguard")]


def get_deps() -> list[str]:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        return tomli.load(f)["project"]["dependencies"]


py39 = ["3.9"]
py312 = ["3.12"]
py313 = ["3.13"]
package = get_package_name()
gh_deps = {
    "async-retriever": [],
    "hydrosignatures": [],
    "pygeoogc": ["async-retriever"],
    "pygeoutils": ["async-retriever", "pygeoogc"],
    "pynhd": ["async-retriever", "pygeoogc", "pygeoutils"],
    "py3dep": ["async-retriever", "pygeoogc", "pygeoutils"],
    "pygeohydro": ["async-retriever", "pygeoogc", "pygeoutils", "pynhd", "hydrosignatures"],
    "pydaymet": ["async-retriever", "pygeoogc", "pygeoutils", "py3dep"],
    "pygridmet": ["async-retriever", "pygeoogc", "pygeoutils"],
    "pynldas2": ["async-retriever", "pygeoutils"],
}
nox.options.sessions = (
    "pc-update",
    "pre-commit",
    "type-check",
    "test39",
    "test312",
)


def install_deps(
    session: nox.Session, extra: str | None = None, version_limit: list[str] | None = None
) -> None:
    """Install package dependencies."""
    deps = [f".[{extra}]"] if extra else ["."]
    deps += [f"git+https://github.com/hyriver/{p}.git" for p in gh_deps[package]]
    if version_limit:
        deps += list(version_limit)
    session.install(*deps)
    dirs = [".pytest_cache", "build", "dist", ".eggs"]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    patterns = ["*.egg-info", "*.egg", "*.pyc", "*~", "**/__pycache__"]
    for p in patterns:
        for f in Path.cwd().rglob(p):
            shutil.rmtree(f, ignore_errors=True)


@nox.session(name="pre-commit", python=py313, venv_backend="micromamba")
def pre_commit(session: nox.Session) -> None:
    """Lint using pre-commit."""
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage=manual",
        *session.posargs,
    )


@nox.session(name="pc-update", python=py313, venv_backend="micromamba")
def pc_update(session: nox.Session) -> None:
    """Lint using pre-commit."""
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "autoupdate",
        *session.posargs,
    )


@nox.session(name="type-check", python=py312, venv_backend="micromamba")
def type_check(session: nox.Session) -> None:
    """Run Pyright."""
    extras = get_extras()
    install_deps(session, ",".join(extras))
    session.install("pyright")
    session.run("pyright")


def setup_session(session: nox.Session) -> bool:
    """Set up session environment with conda and dependencies."""
    session.conda_install("gdal", channel="conda-forge")
    extras = get_extras()
    jit_dep = "jit" in extras
    if jit_dep:
        extras.remove("jit")
    install_deps(session, ",".join(["test", *extras]))
    return jit_dep


def run_tests(
    session: nox.Session,
    jit: bool,
    py_version: Literal[39, 312],
    extra_args: list[str] | None = None,
) -> None:
    """Run the test suite with optional jit and extra arguments."""
    session.run("pytest", *(extra_args or []), *session.posargs)
    session.notify("cover")
    if jit:
        session.notify(f"jit{py_version}")


@nox.session(python="3.9", venv_backend="micromamba")
def test39(session: nox.Session) -> None:
    """Run the test suite for Python 3.9."""
    jit_dep = setup_session(session)
    run_tests(session, jit_dep, 39)


@nox.session(python="3.12", venv_backend="micromamba")
def test312(session: nox.Session) -> None:
    """Run the test suite for Python 3.12."""
    jit_dep = setup_session(session)
    run_tests(session, jit_dep, 312)


@nox.session(python="3.9", venv_backend="micromamba")
def jit39(session: nox.Session) -> None:
    """Run tests that require jit dependencies for Python 3.9."""
    setup_session(session)
    session.run("pytest", "-m", "jit", *session.posargs)


@nox.session(python="3.12", venv_backend="micromamba")
def jit312(session: nox.Session) -> None:
    """Run tests that require jit dependencies for Python 3.12."""
    setup_session(session)
    session.run("pytest", "-m", "jit", *session.posargs)


@nox.session(python=py313, venv_backend="micromamba")
def cover(session: nox.Session) -> None:
    """Coverage analysis."""
    session.install("coverage[toml]")
    session.run("coverage", "report")
    session.run("coverage", "html")
