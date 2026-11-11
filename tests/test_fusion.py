import copy
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from retrievalbench.fusion import SCHEMA, fuse, fuse_queries, load_spec, validate
from retrievalbench.bm25 import BM25Index
from retrievalbench.corpus import load_documents, load_queries
ROOT=Path(__file__).resolve().parents[1]

def spec(lists=None,c=60,k=3):
    return dict(schema=SCHEMA,constant=c,k=k,lists=lists or [dict(id='one',weight=1,depth=3,ranking=['a','b','c'])])

class FusionTests(unittest.TestCase):
    def test_fraction_and_missing(self):
        s=spec([dict(id='one',weight=2,depth=1,ranking=['b','a']),dict(id='two',weight=1,depth=2,ranking=['a','c'])],c=0)
        r=fuse(s);self.assertEqual(r['selected'],['b','a','c']);self.assertEqual(r['rows'][0]['score'],dict(n='2',d='1'))
        self.assertEqual(r['rows'][1]['contributions'][0]['reason'],'beyond_depth')
        self.assertEqual(r['rows'][2]['contributions'][0]['reason'],'missing')
    def test_exact_tie(self):
        s=spec([dict(id='x',weight=1,depth=2,ranking=['b','a']),dict(id='y',weight=1,depth=2,ranking=['a','b'])],k=1)
        r=fuse(s);self.assertEqual(r['selected'],['a']);self.assertTrue(r['cutoff']['tie']);self.assertEqual(r['cutoff']['difference'],dict(n='0',d='1'))
    def test_zero_weight_depth_empty_and_k(self):
        r=fuse(spec([dict(id='x',weight=0,depth=2,ranking=['z','a']),dict(id='y',weight=1,depth=0,ranking=['q'])],k=1))
        self.assertEqual(r['selected'],['a']);self.assertEqual(r['excluded'],['q']);self.assertEqual(r['rows'][0]['score'],dict(n='0',d='1'))
        self.assertEqual(fuse(spec([dict(id='a',weight=1,depth=2,ranking=[])]))['rows'],[])
        self.assertEqual(fuse(spec(k=0))['selected'],[]);self.assertIsNone(fuse(spec(k=0))['cutoff']);self.assertEqual(len(fuse(spec(k=64))['selected']),3)
    def test_duplicate_ranking_is_explicit_vote(self):
        x=dict(id='x',weight=1,depth=2,ranking=['a','b']);y=dict(x,id='y');r=fuse(spec([x,y]));self.assertEqual(r['identical_rankings'],[['x','y']]);self.assertEqual(r['rows'][0]['score'],dict(n='2',d='61'))
    def test_input_copy_and_replay(self):
        s=spec();old=copy.deepcopy(s);r=fuse(s);self.assertEqual(s,old);s['lists'][0]['ranking'].reverse();self.assertEqual(fuse(r['spec']),r)
    def test_strict_invalid(self):
        bad=[]
        for key,values in [('constant',[-1,True,1.0,float('nan'),1000001]),('k',[-1,True,1.0,65]),('schema',['other',None])]:
            for value in values:s=spec();s[key]=value;bad.append(s)
        for key,values in [('weight',[-1,True,1.0,101]),('depth',[-1,True,1.0,65]),('ranking',[['a','a'],['é'],['\n'],[''],['x'*65],{},[1],list(map(str,range(65)))]),('id',['',1,'é'])]:
            for value in values:s=spec();s['lists'][0][key]=value;bad.append(s)
        bad += [None,[],{},dict(spec(),extra=1),spec([dict(id='x',weight=0,depth=1,ranking=[])]),spec([dict(id='x',weight=1,depth=1,ranking=[])]*2)]
        for s in bad:
            with self.subTest(s=s):
                with self.assertRaises(ValueError):fuse(s)
    def test_duplicate_json_and_integer_spellings(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.json'
            for s in ['{"schema":"x","schema":"y"}',json.dumps(spec()).replace('60','60.0'),json.dumps(spec()).replace('60','6e1'),json.dumps(spec()).replace('60','NaN'),' '*131073]:
                p.write_text(s)
                with self.assertRaises(ValueError):load_spec(p)
    def test_exhaustive_permutation_oracle(self):
        # Independent inverse-rank / common-denominator integer ordering.
        count=0
        for a,b in itertools.product(itertools.permutations('abc'),repeat=2):
            for depth,c in itertools.product(range(4),(0,1,60)):
                s=spec([dict(id='x',weight=2,depth=depth,ranking=list(a)),dict(id='y',weight=1,depth=3,ranking=list(b))],c=c)
                expected={d:sum((Fraction(src['weight'],c+src['ranking'].index(d)+1) for src in s['lists'] if d in src['ranking'][:src['depth']]),Fraction()) for d in 'abc'}
                r=fuse(s);self.assertEqual([v['doc'] for v in r['rows']],sorted(expected,key=lambda d:(-expected[d],d)))
                for row in r['rows']:self.assertEqual(Fraction(int(row['score']['n']),int(row['score']['d'])),expected[row['doc']])
                count+=1
        self.assertEqual(count,432)
    def test_random_conservation_reordering_scale(self):
        rng=random.Random(442)
        for _ in range(200):
            lists=[dict(id=f's{i}',weight=rng.randint(1,5),depth=rng.randint(0,6),ranking=rng.sample(list('abcdef'),6)) for i in range(3)];s=spec(lists,c=rng.choice([0,2,60,1000000]));r=fuse(s)
            def f(x):return Fraction(int(x['n']),int(x['d']))
            self.assertEqual(sum((f(x['score']) for x in r['rows']),Fraction()),sum((Fraction(x['weight'],s['constant']+i) for x in lists for i in range(1,min(x['depth'],len(x['ranking']))+1)),Fraction()))
            perm=copy.deepcopy(s);perm['lists'].reverse();self.assertEqual([(x['doc'],x['score']) for x in fuse(perm)['rows']],[(x['doc'],x['score']) for x in r['rows']])
            double=copy.deepcopy(s)
            for x in double['lists']:x['weight']*=2
            rd=fuse(double);self.assertEqual(rd['selected'],r['selected'])
            for a,b in zip(rd['rows'],r['rows']):self.assertEqual(f(a['score']),2*f(b['score']))
    def test_fixtures_and_original_pipeline(self):
        data=json.loads((ROOT/'docs/1-fusion.json').read_text())
        for case in data['cases']:self.assertEqual(fuse(case['report']['spec']),case['report'])
        for p in data['pipelines']:
            docs=[(d['id'],d['text']) for d in p['documents']];index=BM25Index(docs,k1=p['k1'],b=p['b'])
            for q in p['queries']:self.assertEqual(q['ranking'],[dict(doc=d,score=s) for d,s in index.rank(q['text'],top_k=index.size)])
            self.assertEqual(fuse(p['fusion']['spec']),p['fusion'])
    def test_pipeline_random_original_and_bounds(self):
        rng=random.Random(4531)
        for _ in range(100):
            docs=[(f'd{i}', ' '.join(rng.choices(['cat','dog','bird'],k=rng.randint(1,12)))) for i in range(rng.randint(1,12))];qs=[' '.join(rng.choices(['cat','dog','missing'],k=3)) for _ in range(3)];p=fuse_queries(docs,qs,k=3,depth=5);idx=BM25Index(docs)
            for q,r in zip(qs,p['queries']):self.assertEqual(r['ranking'],[dict(doc=d,score=s) for d,s in idx.rank(q,top_k=idx.size)])
        p=fuse_queries([('empty',''),('real','cat')],['unknown','cat']);self.assertEqual(p['omitted'],['empty']);self.assertEqual(p['queries'][0]['ranking'],[dict(doc='real',score=0)])
        for kwargs in [dict(queries=[]),dict(queries=['x']*9),dict(weights=[1,2]),dict(weights=[0]),dict(depth=65),dict(constant=True),dict(k=-1),dict(documents=[('bad\n','x')]),dict(documents=[('a','x'),('a','y')]),dict(documents=[('a','')])]:
            args=dict(documents=[('a','x')],queries=['x']);args.update(kwargs)
            with self.assertRaises(ValueError):fuse_queries(**args)
    def test_cli_both_and_no_output_on_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'out.json'
            cmd=[sys.executable,'-m','retrievalbench']
            self.assertEqual(subprocess.run(cmd+['fuse','--spec',str(ROOT/'examples/fusion.json'),'--json',str(p)],capture_output=True).returncode,0)
            self.assertEqual(json.loads(p.read_text()),fuse(load_spec(ROOT/'examples/fusion.json')))
            self.assertEqual(subprocess.run(cmd+['fuse-search','--corpus',str(ROOT/'examples/data/corpus.jsonl'),'--query','bread','--query','yeast','--weight','2','--weight','1','--json',str(p)],capture_output=True).returncode,0)
            old=p.read_bytes();self.assertEqual(subprocess.run(cmd+['fuse-search','--corpus',str(ROOT/'examples/data/corpus.jsonl'),'--query','bread','--weight','0','--json',str(p)],capture_output=True).returncode,2);self.assertEqual(p.read_bytes(),old)
