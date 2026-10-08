"""Controls as code: the security document must cite real tests, and the repository must hold no secrets."""

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECURITY_DOC = ROOT / "docs" / "security.md"
KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),  # Google API keys
    re.compile(r"sk-[A-Za-z0-9]{32,}"),  # common secret-key prefix
    re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][^'\"\s]{16,}['\"]"),
]


def test_every_cited_test_exists():
    doc = SECURITY_DOC.read_text()
    cited = set(re.findall(r"(tests/[\w/]+\.py)::(test_\w+)", doc))
    assert cited, "security.md cites no tests"
    missing = [f"{path}::{name}" for path, name in sorted(cited) if f"def {name}(" not in (ROOT / path).read_text()]
    assert not missing, f"security.md cites tests that don't exist: {missing}"


def test_every_cited_golden_exists():
    doc = SECURITY_DOC.read_text()
    cited = set(re.findall(r"goldens\.yaml#([\w-]+)", doc))
    assert cited, "security.md cites no golden questions"
    goldens = {g["id"] for g in yaml.safe_load((ROOT / "src/wfx/evals/goldens.yaml").read_text())}
    missing = sorted(c for c in cited if c not in goldens)
    assert not missing, f"security.md cites golden questions that don't exist: {missing}"


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line]


def test_no_secrets_in_tracked_files():
    findings = []
    for path in _tracked_files():
        if not path.is_file() or path.suffix in {".lock", ".parquet", ".duckdb", ".png"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in KEY_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)} matches {pattern.pattern[:20]}…")
    assert not findings, f"possible secrets in tracked files: {findings}"


def test_env_file_is_ignored():
    result = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert result.returncode == 0, ".env must be gitignored"
    assert ".env" not in {p.name for p in _tracked_files()}
