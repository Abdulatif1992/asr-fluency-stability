#!/usr/bin/env python3
"""Verify packaged source integrity and that all notebooks have no outputs."""
from pathlib import Path
import csv,hashlib
import nbformat
root=Path(__file__).resolve().parents[1]
with (root/'MANIFEST_SHA256.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
for row in rows:
    p=root/row['path']
    assert p.is_file(),f'Missing {p}'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'], f'SHA mismatch {p}'
for p in (root/'notebooks').rglob('*.ipynb'):
    nb=nbformat.read(p,as_version=4)
    for c in nb.cells:
        if c.cell_type=='code':
            assert c.get('outputs',[])==[]
            assert c.get('execution_count') is None
print('PASS:',len(rows),'file SHA-256 hashes and notebook output counts verified.')
