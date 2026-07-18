#!/usr/bin/env python3
"""Exec a child command after undoing the parent's temporary signal mask."""

import os
import signal
import sys


if len(sys.argv) < 2:
    raise SystemExit("usage: exec_unblocked.py COMMAND [ARGS ...]")

signal.pthread_sigmask(
    signal.SIG_UNBLOCK,
    {signal.SIGINT, signal.SIGTERM, signal.SIGHUP},
)
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
