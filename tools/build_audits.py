"""Retain controlled arithmetic/policy reports. Not a retrieval benchmark."""
import copy,json
from pathlib import Path
from retrievalbench.audit import audit
ROOT=Path(__file__).resolve().parents[1]
def specs():
    base=dict(documents=[dict(id='a',text='apple apple apple orchard apple apple fruit apple'),dict(id='b',text='apple pear'),dict(id='c',text='pear citrus lemon'),dict(id='empty',text='!!!')],query='apple',grades={'a':1,'b':3,'c':0},k=3,window=3,overlap=1)
    variants=[('Documents',{}),('Chunk budget',{'unit':'chunks'}),('Distinct-parent budget',{'unit':'chunks','budget':'parents'}),('Repeated query term',{'query':'apple apple pear'}),('Unknown term, zero-score ties',{'query':'zebra'}),('Empty query, zero-score ties',{'query':''}),('No length normalization',{'b':0}),('Full length normalization',{'b':1}),('Low saturation',{'k1':.01}),('High saturation',{'k1':100}),('Cutoff one',{'k':1}),('Cutoff beyond corpus',{'k':12}),('Relevant document absent',{'grades':{'missing':3,'a':1}}),('No positive judgments',{'grades':{'a':0}}),('Binary grades',{'grades':{'a':1,'b':1}}),('Exponential grade contrast',{'grades':{'a':1,'b':8}}),('Literal parent delimiter',{'documents':[{'id':'a#c-old','text':'apple pear'},{'id':'b','text':'pear'}],'grades':{'a#c-old':3},'unit':'chunks'}),('Overlapping tail windows',{'documents':[{'id':'a','text':'a b c d e f g'}],'window':4,'overlap':2,'unit':'chunks','query':'g','grades':{'a':1}}),('Zero overlap',{'unit':'chunks','overlap':0}),('Maximum overlap',{'unit':'chunks','overlap':2}),('Explicit unjudged',{'grades':{'a':0,'b':3}}),('Stable identifier ties',{'documents':[{'id':'z','text':'apple'},{'id':'a','text':'apple'}],'grades':{'z':1,'a':3}}),('HTML is text',{'documents':[{'id':'html','text':'<b>apple</b>'},{'id':'plain','text':'pear'}],'grades':{'html':1},'query':'apple'})]
    out=[]
    for label,changes in variants:
        s=copy.deepcopy(base);s.update(changes);out.append((label,s))
    docs=[{'id':d['id'],'text':d['text']} for d in map(json.loads,(ROOT/'examples/data/corpus.jsonl').read_text().splitlines())]
    qs=list(map(json.loads,(ROOT/'examples/data/queries.jsonl').read_text().splitlines()));qrels=list(map(json.loads,(ROOT/'examples/data/qrels.jsonl').read_text().splitlines()))
    for q in qs:
        grades={r['doc_id']:r.get('relevance',1) for r in qrels if r['query_id']==q['id']}
        for unit in ['documents','chunks']:
            out.append(('Original '+q['id']+' / '+unit,dict(documents=docs,query=q['text'],grades=grades,unit=unit,window=30,overlap=5,k=3)))
    return out

def build():return dict(schema_version=1,cases=[dict(label=l,report=audit(s)) for l,s in specs()])
if __name__=='__main__':
    text=json.dumps(build(),sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n'
    (ROOT/'docs/audit.json').write_text(text)
    print('retained',len(specs()),'reports',len(text),'bytes')
