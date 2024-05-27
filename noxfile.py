"""Nox sessions."""

from __future__ import annotations

import shutil
from pathlib import Path

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


python_versions = ["3.9"]
lint_versions = ["3.11"]
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
    "pre-commit",
    "type-check",
    "tests",
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


@nox.session(name="pre-commit", python=lint_versions)
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


@nox.session(name="type-check", python=python_versions)
def type_check(session: nox.Session) -> None:
    """Run Pyright."""
    extras = get_extras()
    install_deps(session, ",".join(extras))
    session.install("pyright")
    session.run("pyright")


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    extras = get_extras()
    speedup_dep = True
    try:
        extras.remove("speedup")
    except ValueError:
        speedup_dep = False

    install_deps(session, ",".join(["test", *extras]))
    session.run(
        "pytest",
        "--doctest-modules",
        f"--cov={package.replace('-', '_')}",
        "--cov-report",
        "xml",
        *session.posargs,
    )
    session.notify("cover")
    if speedup_dep:
        session.notify("speedup")


@nox.session(python=python_versions)
def speedup(session: nox.Session) -> None:
    """Run tests that require speedup deps."""
    extras = get_extras()
    install_deps(session, ",".join(["test", *extras]))
    session.run("pytest", "--doctest-modules", "-m", "speedup", *session.posargs)


@nox.session
def cover(session: nox.Session) -> None:
    """Coverage analysis."""
    session.install("coverage[toml]")
    session.run("coverage", "report")
    session.run("coverage", "erase")
