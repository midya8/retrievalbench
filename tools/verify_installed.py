import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
fixture,root=map(Path,sys.argv[1:]);root=root.resolve()
modules=['retrievalbench','retrievalbench.__main__','retrievalbench.bm25','retrievalbench.cli','retrievalbench.corpus','retrievalbench.metrics','retrievalbench.report','retrievalbench.audit']
for name in modules:
 origin=Path(importlib.util.find_spec(name).origin).resolve();assert root not in origin.parents,(name,origin);print(name,origin)
from retrievalbench.audit import audit
for c in json.loads(fixture.read_text())['cases']:assert audit(c['report']['spec'])==c['report']
with tempfile.TemporaryDirectory() as d:
 out=Path(d)/'audit.json';subprocess.run([sys.executable,'-m','retrievalbench','audit','--spec',str(root/'examples/audit.json'),'--json',str(out)],check=True);assert json.loads(out.read_text())==audit(json.loads((root/'examples/audit.json').read_text()))
 out=Path(d)/'eval.json';subprocess.run([sys.executable,'-m','retrievalbench','evaluate','--corpus',str(root/'examples/data/corpus.jsonl'),'--queries',str(root/'examples/data/queries.jsonl'),'--qrels',str(root/'examples/data/qrels.jsonl'),'-k','3','--min-recall','1','--json',str(out)],check=True);assert json.loads(out.read_text())==json.loads((root/'docs/original-eval.json').read_text())
 out=Path(d)/'chunked.json';subprocess.run([sys.executable,'-m','retrievalbench','evaluate','--corpus',str(root/'examples/data/corpus.jsonl'),'--queries',str(root/'examples/data/queries.jsonl'),'--qrels',str(root/'examples/data/qrels.jsonl'),'-k','3','--chunk-tokens','30','--chunk-overlap','5','--json',str(out)],check=True);assert json.loads(out.read_text())==json.loads((root/'docs/original-chunked.json').read_text())
print('Installed wheel: eight module origins, 35 full audits, audit and original gated CLI pass')
