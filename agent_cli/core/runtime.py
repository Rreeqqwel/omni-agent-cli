from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


class ShellRuntime:
    """Very small, safety-oriented shell runner.

    This is NOT a full sandbox. It is meant as a safe default for an MVP.
    """

    DANGEROUS = {
        "rm",
        "sudo",
        "dd",
        "mkfs",
        "shutdown",
        "reboot",
        "chmod",
        "chown",
    }

    def __init__(self, require_confirm: bool = True):
        self.require_confirm = require_confirm

    def is_dangerous(self, command: str) -> bool:
        try:
            first = shlex.split(command, posix=True)[0]
        except Exception:
            return True
        return first in self.DANGEROUS

    def run(self, command: str, confirm: bool = False, cwd: Optional[str] = None) -> CommandResult:
        if self.require_confirm and self.is_dangerous(command) and not confirm:
            raise RuntimeError(f"Refusing to run potentially dangerous command without confirmation: {command}")

        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        return CommandResult(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
