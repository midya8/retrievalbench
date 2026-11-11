"""Complete deterministic RRF reports. No hand-entered expected scores."""
import json
import random
from pathlib import Path
from retrievalbench.fusion import SCHEMA, fuse, fuse_queries
from retrievalbench.corpus import load_documents, load_queries
ROOT = Path(__file__).resolve().parents[1]

def spec(lists, c=60, k=3):
    return dict(schema=SCHEMA, constant=c, k=k, lists=[dict(id=f"s{i}", weight=w, depth=d, ranking=r) for i, (w,d,r) in enumerate(lists, 1)])

def cases():
    fixed=[
        ('agreement',spec([(1,4,['a','b','c','d']),(1,4,['a','c','b','d'])])),
        ('exact tie',spec([(1,3,['b','a','c']),(1,3,['a','b','c'])],k=1)),
        ('zero weight still candidate',spec([(0,3,['0','z','b']),(1,1,['a','b'])],k=4)),
        ('zero depth',spec([(1,0,['a','b']),(1,0,['c'])])),
        ('empty rankings',spec([(1,5,[]),(0,3,[])])),
        ('missing versus below depth',spec([(1,1,['a','b','c']),(1,2,['b','x'])])),
        ('constant zero',spec([(2,4,['a','b','c']),(1,3,['c','b','a'])],c=0)),
        ('constant million',spec([(100,4,['a','b','c']),(99,3,['c','b','a'])],c=1000000)),
        ('duplicate ranking is extra vote',spec([(1,3,['a','b','c']),(1,3,['c','b','a']),(1,3,['c','b','a'])],k=1)),
        ('depth changes winner',spec([(1,1,['a','b','c']),(1,2,['b','c','a'])],k=1)),
        ('no selected rows',spec([(1,3,['a','b','c'])],k=0)),
        ('all selected rows',spec([(1,64,['a','b','c'])],k=64)),
        ('ASCII tie order',spec([(0,64,['a','A','<','&',' ','__proto__','constructor']),(1,0,[])],k=2)),
        ('maximum profile',spec([(100,64,[f'd{i:02}' for i in (range(64) if j%2 else reversed(range(64)))]) for j in range(8)],k=32)),
    ]
    rng=random.Random(270091)
    for n in range(200):
        universe=[f'r{i:02}' for i in range(rng.randint(1,24))];ls=[]
        for _ in range(rng.randint(1,8)):
            r=rng.sample(universe,rng.randint(0,len(universe)));ls.append((rng.randrange(5),rng.randrange(len(universe)+2),r))
        if not any(x[0] for x in ls):ls[0]=(1,ls[0][1],ls[0][2])
        fixed.append((f'seeded {n:03}',spec(ls,c=rng.choice([0,1,5,60,1000000]),k=rng.randrange(26))))
    return [dict(name=name,report=fuse(s)) for name,s in fixed]

def build():
    data=cases();docs=load_documents(ROOT/'examples/data/corpus.jsonl');queries=load_queries(ROOT/'examples/data/queries.jsonl');pipelines=[]
    for depth in (0,1,3,64):
        for constant in (0,60):
            for weights in ([1,1,1],[2,0,3]):
                texts=[q for _,q in queries[:3]]
                result=fuse_queries(docs,texts,depth=depth,constant=constant,k=3,weights=weights)
                pipelines.append(result);data.append(dict(name=f'actual BM25 depth{depth} c{constant} weights{weights}',report=result['fusion']))
    return dict(schema='retrievalbench-rrf-fixtures/v1',cases=data,pipelines=pipelines)
if __name__=='__main__':
    data=build();(ROOT/'docs/1-fusion.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (ROOT/'examples/fusion.json').write_text(json.dumps(data['cases'][0]['report']['spec'],indent=2)+'\n',encoding='utf-8')
    print(len(data['cases']),'complete reports,',len(data['pipelines']),'actual BM25 pipelines')
