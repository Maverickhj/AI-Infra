"""Linux-only, per-invocation child subreaper for bounded local runtime processes.

Executed in its own Python process. It never changes the caller's reaper state
and never signals a process discovered outside its descendant tree. pidfds keep
signals tied to process identity even after PID reuse or setsid/double-fork.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import selectors
import signal
import subprocess
import sys
import time


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def children(pid):
    result = set()
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return result
    for task in tasks:
        try:
            result.update(map(int, (task/"children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


class OwnedTree:
    def __init__(self):
        self.owner = os.getpid()
        self.processes = {}
        self.signaled = set()

    @staticmethod
    def alive(fd):
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        return not poll.poll(0)

    def refresh(self):
        queue, seen = [self.owner], set()
        while queue:
            parent = queue.pop()
            if parent in seen:
                continue
            seen.add(parent)
            if parent != self.owner:
                identity = process_identity(parent)
                known = self.processes.get(parent)
                if identity is None or known is None or identity[1] != known[0]:
                    continue
            for pid in children(parent):
                identity = process_identity(pid)
                if identity is None or identity[0] != parent:
                    continue
                known = self.processes.get(pid)
                if known is None or known[0] != identity[1]:
                    try:
                        fd = os.pidfd_open(pid)
                    except ProcessLookupError:
                        continue
                    # Recheck after opening the identity-bound descriptor.
                    if process_identity(pid) != identity:
                        os.close(fd)
                        continue
                    if known is not None:
                        os.close(known[1])
                    self.processes[pid] = (identity[1], fd)
                queue.append(pid)

    def reap(self, proc):
        proc.poll()  # Preserve the leader's actual return code.
        for pid in children(self.owner):
            if pid == proc.pid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def active(self):
        return [(pid, birth, fd) for pid, (birth, fd) in self.processes.items()
                if self.alive(fd)]

    def cleanup(self, proc):
        start = time.monotonic()
        sent = set()
        while True:
            self.refresh()
            self.reap(proc)
            active = self.active()
            if not active and not children(self.owner):
                return dict(status="completed", descendants_seen=len(self.processes),
                            descendants_signaled=len(self.signaled))
            sig = signal.SIGTERM if time.monotonic()-start < 2 else signal.SIGKILL
            for pid, birth, fd in active:
                key = (pid, birth, sig)
                if key not in sent:
                    try:
                        signal.pidfd_send_signal(fd, sig)
                        self.signaled.add((pid, birth))
                    except ProcessLookupError:
                        pass
                    sent.add(key)
            if time.monotonic()-start >= 5:
                return dict(status="failed", descendants_seen=len(self.processes),
                            descendants_signaled=len(self.signaled),
                            remaining=len(active))
            time.sleep(.02)

    def close(self):
        for _, fd in self.processes.values():
            os.close(fd)


def supervise(argv, *, log_path, max_seconds, max_output_bytes, parent_pid):
    if (not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open")
            or not hasattr(signal, "pidfd_send_signal") or not Path("/proc/self/task").is_dir()):
        raise RuntimeError("runtime ownership requires Linux /proc and pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    for operation, argument in ((36, 1), (1, signal.SIGTERM)):
        # PR_SET_CHILD_SUBREAPER, PR_SET_PDEATHSIG from linux/prctl.h.
        if libc.prctl(operation, argument, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "cannot configure owned runtime reaper")
    if os.getppid() != parent_pid:
        raise RuntimeError("runtime supervisor parent exited before initialization")
    interrupted = []
    def stop(signum, _frame):
        interrupted.append(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start, total, reason = time.monotonic(), 0, None
    tree = OwnedTree()
    cleanup = None
    try:
        with Path(log_path).open("xb") as output:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            try:
                while selector.get_map() or proc.poll() is None:
                    tree.refresh()
                    if interrupted:
                        reason = "supervisor_interrupted"
                        break
                    if time.monotonic()-start >= max_seconds:
                        reason = "wall_time_limit"
                        break
                    ready = selector.select(timeout=min(.05, max_seconds))
                    for key, _ in ready:
                        raw = os.read(key.fileobj.fileno(), 65536)
                        if not raw:
                            selector.unregister(key.fileobj)
                            continue
                        available = max(0, max_output_bytes-total)
                        output.write(raw[:available])
                        total += len(raw)
                        if total > max_output_bytes:
                            reason = "output_limit"
                            break
                    if reason or proc.poll() is not None and not ready:
                        break
            finally:
                cleanup = tree.cleanup(proc)
                selector.close()
                proc.stdout.close()
            returncode = proc.poll()
            if cleanup["status"] != "completed":
                reason = "descendant_cleanup_failed"
            return dict(status="passed" if returncode == 0 and reason is None else "failed",
                        returncode=returncode, stop_reason=reason, output_bytes=total,
                        elapsed_wall_seconds=time.monotonic()-start,
                        descendant_cleanup=cleanup, ownership="dedicated_linux_subreaper_pidfd")
    finally:
        tree.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.seconds < float("inf") or args.bytes <= 0:
        raise ValueError("invalid bounded process arguments")
    result = supervise(command, log_path=args.log, max_seconds=args.seconds,
                       max_output_bytes=args.bytes, parent_pid=args.parent)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
