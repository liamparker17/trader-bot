"""
Task 2 — Blocker A: single-instance lock.

Covers:
- Acquiring the lock creates a PID file.
- A second acquire attempt fails while the first instance is alive.
- A stale lock (owning PID no longer running) is reclaimed.
- Releasing the lock allows a subsequent acquire to succeed.
"""

import os

from src.utils.instance_lock import InstanceLock, _pid_running


def _find_dead_pid():
    """Find a PID number that is (almost certainly) not currently running."""
    candidate = 999_999
    while _pid_running(candidate) and candidate > 2:
        candidate -= 1
    return candidate


def test_acquire_creates_lock_file_with_pid(tmp_path):
    lock_path = tmp_path / "traderbot.lock"
    lock = InstanceLock(lock_path)

    assert lock.acquire() is True
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    lock.release()


def test_second_acquire_fails_while_first_holds_lock(tmp_path):
    lock_path = tmp_path / "traderbot.lock"
    first = InstanceLock(lock_path)
    second = InstanceLock(lock_path)

    assert first.acquire() is True
    # This test process's own PID is alive, so the second lock must
    # treat the existing lock file as held by a live process.
    assert second.acquire() is False

    first.release()


def test_stale_lock_is_reclaimed(tmp_path):
    lock_path = tmp_path / "traderbot.lock"
    dead_pid = _find_dead_pid()
    lock_path.write_text(str(dead_pid), encoding="utf-8")

    lock = InstanceLock(lock_path)

    assert lock.acquire() is True
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    lock.release()


def test_release_allows_reacquire(tmp_path):
    lock_path = tmp_path / "traderbot.lock"
    lock = InstanceLock(lock_path)

    assert lock.acquire() is True
    lock.release()
    assert not lock_path.exists()

    lock2 = InstanceLock(lock_path)
    assert lock2.acquire() is True
    lock2.release()


def test_release_without_acquire_is_a_noop(tmp_path):
    lock_path = tmp_path / "traderbot.lock"
    lock = InstanceLock(lock_path)

    # Never acquired — release must not raise or touch the filesystem.
    lock.release()
    assert not lock_path.exists()


def test_pid_running_detects_current_process():
    assert _pid_running(os.getpid()) is True


def test_pid_running_false_for_dead_pid():
    assert _pid_running(_find_dead_pid()) is False
