#!/usr/bin/env python3
"""Offline syntax and publication-boundary check. Does not import the app."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.venv', 'venv', 'node_modules',
        '.pytest_cache', '.mypy_cache'}
FORBIDDEN = {'.ppt', '.pptx', '.pdf', '.mp4', '.webm', '.mov', '.mkv',
             '.avi', '.m4v', '.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac',
             '.jpg', '.jpeg', '.png', '.key', '.pem', '.p12', '.pfx', '.crt',
             '.onnx', '.tflite', '.gguf', '.safetensors', '.pt', '.pth',
             '.bin', '.so', '.dylib', '.dll', '.sqlite', '.sqlite3', '.db',
             '.zip', '.zst', '.tar'}
SECRET_PATTERNS = [
    ('github-token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b')),
    ('provider-token', re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b')),
    ('aws-access-id', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('private-key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
]


def check(snapshot: bool = False) -> int:
    findings = []
    counts = {'files': 0, 'python': 0, 'json': 0, 'snapshot_hashes': 0}
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if any(part in SKIP for part in rel.parts):
            continue
        if p.is_symlink():
            findings.append({'path': str(rel), 'category': 'symlink'})
            continue
        if not p.is_file():
            continue
        counts['files'] += 1
        if (p.suffix.lower() in FORBIDDEN or p.name.startswith('.env') or
                any(part in {'data', 'state', 'recordings', 'captures'} for part in rel.parts)):
            findings.append({'path': str(rel), 'category': 'excluded-publication-material'})
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            findings.append({'path': str(rel), 'category': 'non-text-file'})
            continue
        for category, pattern in SECRET_PATTERNS:
            for hit in pattern.finditer(text):
                # Only filename, line and category are returned, never a value.
                findings.append({'path': str(rel), 'line': text.count('\n', 0, hit.start()) + 1,
                                 'category': category})
        try:
            if p.suffix == '.py':
                ast.parse(text, filename=str(rel))
                counts['python'] += 1
            elif p.suffix == '.json':
                json.loads(text)
                counts['json'] += 1
        except (SyntaxError, ValueError) as exc:
            findings.append({'path': str(rel), 'category': 'invalid-syntax',
                             'line': getattr(exc, 'lineno', None)})
    if snapshot:
        manifest = json.loads((ROOT / 'SOURCE_SNAPSHOT.json').read_text())
        for entry in manifest['files']:
            p = ROOT / entry['path']
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
                findings.append({'path': entry['path'], 'category': 'snapshot-mismatch'})
            else:
                counts['snapshot_hashes'] += 1
    print(json.dumps({'ok': not findings, 'counts': counts, 'findings': findings}, indent=2))
    return int(bool(findings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true',
                        help='Also verify the frozen on-device source manifest.')
    sys.exit(check(parser.parse_args().snapshot))

