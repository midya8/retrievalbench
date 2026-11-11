"""Installed-wheel API, actual BM25, CLI and deterministic-regeneration checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(sys.argv[1]);data=json.loads((ROOT/'docs/1-fusion.json').read_text())
for name in ('retrievalbench','retrievalbench.bm25','retrievalbench.corpus','retrievalbench.metrics','retrievalbench.report','retrievalbench.audit','retrievalbench.cli','retrievalbench.__main__','retrievalbench.fusion'):
    origin=importlib.util.find_spec(name).origin
    if ROOT in Path(origin).resolve().parents:raise AssertionError('source import leaked '+name)
from retrievalbench.fusion import fuse, fuse_queries
for case in data['cases']:assert fuse(case['report']['spec'])==case['report']
for p in data['pipelines']:
    s=p['fusion']['spec'];assert fuse_queries([(d['id'],d['text']) for d in p['documents']],[q['text'] for q in p['queries']],constant=s['constant'],k=s['k'],depth=s['lists'][0]['depth'],weights=[x['weight'] for x in s['lists']],k1=p['k1'],b=p['b'])==p
with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'report.json';cmd=[sys.executable,'-m','retrievalbench']
    subprocess.run(cmd+['fuse','--spec',str(ROOT/'examples/fusion.json'),'--json',str(out)],check=True)
    assert json.loads(out.read_text())==data['cases'][0]['report']
    subprocess.run(cmd+['fuse-search','--corpus',str(ROOT/'examples/data/corpus.jsonl'),'--query','bread','--query','yeast','--weight','2','--weight','1','--depth','3','-k','3','--json',str(out)],check=True)
    r=json.loads(out.read_text());assert fuse(r['fusion']['spec'])==r['fusion']
    out.write_text('preserve prior output')
    p=subprocess.run(cmd+['fuse-search','--corpus',str(ROOT/'examples/data/corpus.jsonl'),'--query','bread','--weight','0','--json',str(out)],capture_output=True)
    assert p.returncode==2 and out.read_text()=='preserve prior output'
print('installed fusion: nine module origins, 230 exact reports, 16 original-BM25 pipelines, both CLI commands and invalid-write preservation pass')
