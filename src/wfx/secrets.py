"""Read secrets the same way on a laptop and on any cloud.

``secret("GOOGLE_API_KEY")`` looks for, in order:

1. ``GOOGLE_API_KEY_FILE``: a path to a file holding the value. Secret managers on every
   major platform can mount a secret as a file, which keeps it out of the process
   environment, crash dumps and ``/proc/<pid>/environ``.
2. ``GOOGLE_API_KEY``: the value itself (local development, via a gitignored ``.env``).

Values are never logged; errors name the variable, not the value.
"""

from __future__ import annotations

import os
from pathlib import Path


def secret(name: str, required: bool = True) -> str | None:
    path = os.environ.get(f"{name}_FILE")
    if path:
        file = Path(path)
        if not file.is_file():
            raise RuntimeError(f"{name}_FILE points to a file that doesn't exist")
        value = file.read_text().strip()
    else:
        value = os.environ.get(name, "").strip()
    if not value and required:
        raise RuntimeError(f"{name} is not set: provide {name}_FILE (mounted secret) or {name} (local .env), never in code")
    return value or None
