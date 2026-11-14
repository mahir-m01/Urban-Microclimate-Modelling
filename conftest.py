"""Shared pytest set-up for the whole repository.

On macOS, PyTorch and the tree libraries (LightGBM, XGBoost) each load their own copy of the
OpenMP runtime, and once a process has imported PyTorch, LightGBM crashes or hangs in it. So on
macOS, when a run would mix the two, the PyTorch tests (all of them live in src/urbanwind/dl/tests)
are taken out of the main run and run afterwards in a second pytest process with the same options.
On Linux, including CI, everything runs in one process as usual.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import shlex
import subprocess
import sys

import pytest

DL_TESTS = pathlib.Path(__file__).resolve().parent / "src" / "urbanwind" / "dl" / "tests"
CHILD = "URBANWIND_DL_TESTS_ONLY"
DEFERRED = pytest.StashKey[list]()  # targets for the second process; set only when the run is split
OPTIONS = pytest.StashKey[list]()
SESSION = pytest.StashKey[pytest.Session]()
# options that name an output file or folder: the second process would overwrite the first one's
NOT_FORWARDED = ("--junitxml", "--junit-xml", "--basetemp", "--resultlog", "--report-log")


def _in_dl_tests(path: pathlib.Path) -> bool:
    path = path.resolve()
    return path == DL_TESTS or DL_TESTS in path.parents


def _user_ignores_dl_tests(config) -> bool:
    root = config.invocation_params.dir
    ignored = [(root / p).resolve() for p in config.getoption("ignore") or []]
    globs = config.getoption("ignore_glob") or []
    return any(DL_TESTS == p or p in DL_TESTS.parents for p in ignored) or any(
        fnmatch.fnmatch(str(DL_TESTS), str(root / g)) for g in globs)


def _forwarded(args: list[str], given: set[str]) -> list[str]:
    """The command-line options for the second process: no test paths and no output files."""
    out, skip_next = [], False
    for arg in args:
        if skip_next:
            skip_next = False
        elif arg.split("=")[0] in NOT_FORWARDED:
            skip_next = "=" not in arg
        elif arg not in given:
            out.append(arg)
    return out


def pytest_configure(config):
    if sys.platform != "darwin" or os.environ.get(CHILD):
        return
    root = config.invocation_params.dir
    dl = [arg for arg in config.args if _in_dl_tests(root / arg.split("::")[0])]
    if len(dl) == len(config.args):  # only PyTorch tests were asked for, so nothing to keep apart
        return
    # pytest never asks pytest_ignore_collect about paths given on the command line, so explicit
    # PyTorch paths are taken out of the arguments here
    given = set(config.args) if config.args_source == pytest.Config.ArgsSource.ARGS else set()
    config.args[:] = [arg for arg in config.args if arg not in dl]
    # pytest does not apply --ignore to testpaths, so an ignored PyTorch folder is dropped here too
    if _user_ignores_dl_tests(config):
        return
    config.stash[DEFERRED] = dl
    # PYTEST_ADDOPTS is passed on explicitly and removed from the child's environment, so a path in
    # it cannot pull the other tests into the PyTorch process
    addopts = shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))
    config.stash[OPTIONS] = _forwarded([*addopts, *config.invocation_params.args], given)


def pytest_ignore_collect(collection_path, config):
    # a folder above the PyTorch tests (for example "src") reaches them during collection
    deferred = config.stash.get(DEFERRED, None)
    if deferred is None or not _in_dl_tests(collection_path):
        return None
    if str(DL_TESTS) not in deferred:
        deferred.append(str(DL_TESTS))
    return True


def pytest_sessionstart(session):
    session.config.stash[SESSION] = session


def pytest_unconfigure(config):
    # runs after the main run's summary, so the last lines on screen are the PyTorch tests' result
    deferred, session = config.stash.get(DEFERRED, None), config.stash.get(SESSION, None)
    if not deferred or session is None:
        return
    if session.exitstatus == pytest.ExitCode.INTERRUPTED or session.shouldfail:  # Ctrl-C, or -x/--maxfail hit
        print("\n\nPyTorch tests not run: the main run stopped early (see conftest.py)", flush=True)
        return
    print("\n\nPyTorch tests, in their own process (see conftest.py):", flush=True)
    argv = [sys.executable, "-m", "pytest", *config.stash[OPTIONS], *deferred]
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"} | {CHILD: "1"}
    child = subprocess.run(argv, cwd=config.invocation_params.dir, env=env).returncode
    main = int(session.exitstatus)
    # "no tests ran" (5) on one side is no result; any failure on either side fails the run
    if main == pytest.ExitCode.NO_TESTS_COLLECTED or (main == 0 and child != pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = child
    status = int(session.exitstatus)
    verdict = {0: "all passed", 5: "no tests ran"}.get(status, f"FAILED (exit code {status})")
    print(f"\nWhole run (main tests and PyTorch tests): {verdict}", flush=True)
