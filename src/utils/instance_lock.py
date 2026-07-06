"""
Single-instance lock for TraderBot (pre-live hardening — Blocker A).

Prevents two bot processes from trading on the same account at the same
time. Uses an atomic file create (``os.O_CREAT | os.O_EXCL``) so it works
identically on Windows and POSIX without relying on ``fcntl``. The lock
file stores the owning PID; if the file already exists but its PID is no
longer running, the lock is considered stale and is safely reclaimed.

Usage:
    lock = InstanceLock()
    if not lock.acquire():
        logger.critical("Another TraderBot instance is already running.")
        sys.exit(1)
    ...
    lock.release()
"""

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("traderbot.instance_lock")

DEFAULT_LOCK_PATH = Path("data") / "traderbot.lock"

# Minimal access right needed to just check whether a process exists.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _pid_running(pid: int) -> bool:
    """Return True if a process with the given PID is currently running.

    Windows-safe: uses ctypes OpenProcess (no psutil/fcntl dependency).
    Falls back to os.kill(pid, 0) on POSIX platforms.
    """
    if pid is None or pid <= 0:
        return False

    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still running.
        return True
    except OSError:
        return False
    return True


class InstanceLock:
    """Exclusive single-instance lock backed by an atomic PID file."""

    def __init__(self, lock_path: Optional[Union[str, Path]] = None):
        self.lock_path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
        self._acquired = False

    def acquire(self) -> bool:
        """Attempt to acquire the lock.

        Returns True if acquired. Returns False if another live instance
        already holds the lock. A stale lock (owning PID not running) is
        automatically reclaimed and re-acquired.
        """
        if self._create_lock_file():
            self._acquired = True
            return True

        existing_pid = self._read_pid()
        if existing_pid is not None and _pid_running(existing_pid):
            logger.critical(
                f"Another TraderBot instance is already running "
                f"(PID {existing_pid}, lock file: {self.lock_path})"
            )
            return False

        logger.warning(
            f"Stale lock file found at {self.lock_path} "
            f"(PID {existing_pid} not running). Reclaiming."
        )
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.critical(f"Failed to remove stale lock file {self.lock_path}: {e}")
            return False

        if self._create_lock_file():
            self._acquired = True
            return True

        logger.critical(f"Failed to acquire instance lock at {self.lock_path}.")
        return False

    def release(self) -> None:
        """Release the lock if held by this instance. Safe to call if not held."""
        if not self._acquired:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Failed to remove lock file {self.lock_path} on release: {e}")
        finally:
            self._acquired = False

    def _create_lock_file(self) -> bool:
        """Atomically create the lock file with our PID. Returns False if it exists."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
        finally:
            os.close(fd)
        return True

    def _read_pid(self) -> Optional[int]:
        try:
            content = self.lock_path.read_text(encoding="utf-8").strip()
            return int(content)
        except (FileNotFoundError, ValueError):
            return None
