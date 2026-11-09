import copy,itertools,json,math,random,tempfile,unittest
from decimal import Decimal,localcontext
from pathlib import Path
from retrievalbench.audit import audit,load_spec
from retrievalbench.bm25 import BM25Index
from retrievalbench.corpus import chunk_records,load_qrels
from retrievalbench.metrics import precision_at_k,recall_at_k,reciprocal_rank,ndcg_at_k
from retrievalbench.cli import main
from tools.build_audits import build,specs
ROOT=Path(__file__).resolve().parents[1]
class AuditTests(unittest.TestCase):
 def base(self):return dict(documents=[dict(id='a',text='apple apple apple apple apple apple'),dict(id='b',text='apple pear')],query='apple',grades={'a':1,'b':3},unit='chunks',window=2,overlap=0,k=2)
 def test_all_retained_exact(self):self.assertEqual(build(),json.loads((ROOT/'docs/audit.json').read_text()))
 def test_chunk_crowding_policy(self):
  r=audit(self.base());self.assertEqual(r['policies']['chunks']['parents'],['a']);self.assertEqual(r['policies']['parents']['parents'],['a','b'])
 def test_complete_collapse_and_weights(self):
  r=audit(self.base());self.assertEqual(len(r['policies']['chunks']['collapse']),len(r['units']))
  for u in r['units']:self.assertAlmostEqual(u['score'],sum(w['weighted'] for w in u['weights']))
 def test_repeated_query_scales_score(self):
  a=audit(self.base());s=self.base();s['query']='apple apple';b=audit(s)
  for x,y in zip(a['units'],b['units']):self.assertAlmostEqual(y['score'],2*x['score'])
 def test_zero_query_stable_ties(self):
  s=self.base();s['query']='zebra';r=audit(s);self.assertEqual([u['id'] for u in r['units']],sorted(u['id'] for u in r['units']))
 def test_empty_units_omitted(self):
  s=self.base();s['documents'].append(dict(id='punct',text='!!!'));r=audit(s);self.assertEqual(len(r['omitted']),1);self.assertEqual(r['N'],4)
 def test_parent_delimiter_preserved(self):
  r=chunk_records([('a#c-old','apple pear')],max_tokens=3,overlap=0);self.assertEqual(r[0]['parent'],'a#c-old')
 def test_generated_id_collision_rejected(self):
  with self.assertRaises(ValueError):chunk_records([('a','one two three'),('a#c0','one')],max_tokens=2,overlap=0)
 def test_tail_windows_preserved(self):
  r=chunk_records([('a','a b c d e f g')],max_tokens=4,overlap=2);self.assertEqual([(x['start'],x['end']) for x in r],[(0,4),(2,6),(4,7),(6,7)])
 def test_token_spans_exact(self):
  for _,s in specs():
   for u in audit(s)['units']:
    for t in u['tokens']:self.assertEqual(u['text'][t['start']:t['end']].lower(),t['term'])
 def test_decimal_bm25_reference(self):
  rng=random.Random(17)
  with localcontext() as ctx:
   ctx.prec=70
   for _ in range(200):
    N=rng.randint(1,16);df=rng.randint(1,N);tf=rng.randint(1,20);length=rng.randint(tf,40);avg=rng.randint(1,40);k1=Decimal(str(rng.choice([.01,.7,1.5,12,100])));b=Decimal(str(rng.choice([0,.25,.75,1])))
    idf=(Decimal(1)+(Decimal(N-df)+Decimal('.5'))/(Decimal(df)+Decimal('.5'))).ln();norm=1-b+b*Decimal(length)/Decimal(avg);reference=idf*tf*(k1+1)/(tf+k1*norm)
    index=BM25Index([('x','apple')],k1=float(k1),b=float(b));index._idf={'apple':math.log((N-df+.5)/(df+.5)+1)};index._term_freqs=[{'apple':tf}];index._lengths=[length];index._avg_length=avg
    self.assertAlmostEqual(index._score(['apple'],0),float(reference),places=11)
 def test_exhaustive_metric_reference(self):
  docs=['a','b','c','d'];grades={'a':3,'b':1,'c':0};rel={'a','b'}
  for n in range(5):
   for ranked in itertools.permutations(docs,n):
    for k in range(1,6):
     ranked=list(ranked);hits=len(set(ranked[:k])&rel)
     self.assertEqual(precision_at_k(ranked,rel,k),hits/k);self.assertEqual(recall_at_k(ranked,rel,k),hits/2)
     ideal=7+1/math.log2(3) if k>1 else 7;dcg=sum((2**grades.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]));self.assertAlmostEqual(ndcg_at_k(ranked,grades,k),dcg/ideal)
 def test_duplicate_rankings_rejected(self):
  for func,args in [(precision_at_k,({'a'},2)),(recall_at_k,({'a'},2)),(reciprocal_rank,({'a'},)),(ndcg_at_k,({'a':3},2))]:
   with self.assertRaises(ValueError):func(['a','a'],*args)
 def test_large_grade_scaled_gain(self):
  self.assertEqual(ndcg_at_k(['a','b'],{'a':10000,'b':9999},2),1)
  self.assertLess(ndcg_at_k(['b','a'],{'a':10000,'b':9999},2),1)
  self.assertEqual(ndcg_at_k(['x'],{'a':10**1000},1),0)
 def test_nonfinite_bm25_rejected(self):
  for k1 in [math.nan,math.inf,-math.inf,True]:
   with self.assertRaises(ValueError):BM25Index([('a','one')],k1=k1)
 def test_huge_finite_k1_stays_finite(self):self.assertTrue(math.isfinite(BM25Index([('a','one one')],k1=1e308).rank('one')[0][1]))
 def test_duplicate_native_ids_rejected(self):
  with self.assertRaises(ValueError):BM25Index([('a','one'),('a','two')])
 def test_duplicate_qrel_pairs_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'q';p.write_text('{"query_id":"q","doc_id":"a"}\n'*2)
   with self.assertRaises(ValueError):load_qrels(p)
 def test_profile_rejects_bad_values(self):
  for patch in [{'k':None},{'k':True},{'k1':math.nan},{'grades':{'a':9}},{'documents':[{'id':'a','text':'é'}]},{'query':'word '*65},{'extra':1},{'window':1,'overlap':1}]:
   with self.subTest(patch=patch),self.assertRaises(ValueError):audit({**self.base(),**patch})
 def test_no_positive_query_explicit(self):
  s=self.base();s['grades']={'a':0};m=audit(s)['policies']['chunks']['metrics'];self.assertFalse(m['included_in_macro']);self.assertEqual(m['idcg'],0)
 def test_missing_judgment_document_in_ideal(self):
  s=self.base();s['grades']={'absent':3};m=audit(s)['policies']['chunks']['metrics'];self.assertEqual(m['values']['recall'],0);self.assertEqual(m['ideal'][0]['doc'],'absent')
 def test_strict_loader(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'s';p.write_text('{"query":"one","query":"two"}')
   with self.assertRaises(ValueError):load_spec(p)
 def test_audit_cli(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'s';o=Path(t)/'o';p.write_text(json.dumps(self.base()));self.assertEqual(main(['audit','--spec',str(p),'--json',str(o)]),0);self.assertEqual(json.loads(o.read_text()),audit(self.base()))
 def test_literal_parent_cli(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)/'d';q=Path(t)/'q';g=Path(t)/'g';o=Path(t)/'o';d.write_text('{"id":"a#c-old","text":"apple apple apple"}\n');q.write_text('{"id":"q","text":"apple"}\n');g.write_text('{"query_id":"q","doc_id":"a#c-old"}\n')
   args=['evaluate','--corpus',str(d),'--queries',str(q),'--qrels',str(g),'-k','2','--chunk-tokens','2','--chunk-overlap','0','--json',str(o)];self.assertEqual(main(args),0);self.assertEqual(json.loads(o.read_text())['averages']['recall_at_2'],1)
   self.assertEqual(main(args+['--min-recall','nan']),2)

 def test_integral_json_numbers(self):
  s=self.base();s['k']=2.0;s['grades']={'a':1.0};self.assertEqual(audit(s)['spec']['grades'],{'a':1})
 def test_original_source_hashes(self):
  import hashlib
  manifest=json.loads((ROOT/'docs/original-sha256.json').read_text())
  for name,digest in manifest.items():
   path=ROOT/('docs/original-readme.md' if name=='README.md' else name)
   self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name)
