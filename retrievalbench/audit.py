"""Bounded complete retrieval arithmetic witnesses, not a relevance oracle."""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from .bm25 import BM25Index, TOKEN, tokenize
from .corpus import chunk_records
from .metrics import evaluate_run


def load_spec(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw)>131072:
        raise ValueError("spec exceeds 128 KiB")
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError("duplicate JSON key")
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _integer(value,low,high,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not low<=value<=high or not math.isfinite(value) or int(value)!=value:
        raise ValueError(f"{label} must be an integer in [{low},{high}]")
    return int(value)


def _text(value,limit,label):
    if not isinstance(value,str) or len(value)>limit or any(ord(c)>127 or (ord(c)<32 and c not in "\n\r\t") for c in value):
        raise ValueError(f"{label} must be bounded ASCII text")
    return value


def validate(spec):
    if not isinstance(spec,dict) or set(spec)-{"documents","query","grades","k","k1","b","window","overlap","unit","budget"}:
        raise ValueError("unknown spec fields")
    docs=spec.get("documents")
    if not isinstance(docs,list) or not 1<=len(docs)<=16: raise ValueError("need 1..16 documents")
    clean=[];seen=set();total=0
    for d in docs:
        if not isinstance(d,dict) or set(d)!={"id","text"}: raise ValueError("document needs id/text")
        key=_text(d["id"],64,"id")
        if not key or key in seen: raise ValueError("ids must be nonempty and unique")
        seen.add(key);text=_text(d["text"],2048,"document");total+=len(tokenize(text));clean.append({"id":key,"text":text})
    if total>1024: raise ValueError("at most 1024 corpus tokens")
    query=_text(spec.get("query"),512,"query")
    if len(tokenize(query))>64: raise ValueError("at most 64 query tokens")
    grades=spec.get("grades",{})
    if not isinstance(grades,dict) or len(grades)>32: raise ValueError("at most 32 judgments")
    for key,g in grades.items():
        if not _text(key,64,"grade id"): raise ValueError("empty judgment id")
        _integer(g,0,8,"grade")
    grades={key:int(g) for key,g in grades.items()}
    k=_integer(spec.get("k",3),1,32,"k");window=_integer(spec.get("window",12),1,256,"window")
    overlap=_integer(spec.get("overlap",2),0,window-1,"overlap")
    k1=spec.get("k1",1.5);b=spec.get("b",.75)
    if isinstance(k1,bool) or not isinstance(k1,(int,float)) or not .01<=k1<=100 or not math.isfinite(k1): raise ValueError("k1 in [.01,100]")
    if isinstance(b,bool) or not isinstance(b,(int,float)) or not 0<=b<=1 or not math.isfinite(b): raise ValueError("b in [0,1]")
    unit=spec.get("unit","documents");budget=spec.get("budget","chunks")
    if unit not in ("documents","chunks") or budget not in ("chunks","parents"): raise ValueError("invalid unit/budget")
    return dict(documents=clean,query=query,grades=grades,k=k,window=window,overlap=overlap,k1=k1,b=b,unit=unit,budget=budget)


def _metrics(parents,grades,k):
    relevant=sorted(d for d,g in grades.items() if g>0)
    ideal=sorted(grades,key=lambda d:(-grades[d],d))[:k]
    ideal_rows=[dict(rank=i,doc=d,grade=grades[d],gain=2**grades[d]-1,discount=1/math.log2(i+1),contribution=(2**grades[d]-1)/math.log2(i+1)) for i,d in enumerate(ideal,1)]
    idcg=sum(r["contribution"] for r in ideal_rows)
    rows=[];dcg=0;hits=0;rr=0
    for i,d in enumerate(parents[:k],1):
        grade=grades.get(d,0);hit=grade>0;hits+=hit
        gain=2**grade-1;discount=1/math.log2(i+1);contribution=gain*discount;dcg+=contribution
        if hit and not rr: rr=1/i
        rows.append(dict(rank=i,doc=d,judged=d in grades,grade=grade,hit=hit,gain=gain,discount=discount,contribution=contribution,cumulative_dcg=dcg,cumulative_hits=hits))
    values=dict(precision=hits/k,recall=hits/len(relevant) if relevant else 0,rr=rr,ndcg=dcg/idcg if idcg else 0)
    if relevant:
        native=evaluate_run({"audit":parents[:k]},{"audit":grades},k=k)["queries"][0]
        for a,b in [("precision",f"precision_at_{k}"),("recall",f"recall_at_{k}"),("rr","reciprocal_rank"),("ndcg",f"ndcg_at_{k}")]:
            if abs(values[a]-native[b])>1e-12: raise AssertionError("metric mismatch")
    return dict(rows=rows,ideal=ideal_rows,relevant=relevant,dcg=dcg,idcg=idcg,values=values,included_in_macro=bool(relevant))


def audit(spec):
    spec=validate(spec);docs=[(d["id"],d["text"]) for d in spec["documents"]]
    records=chunk_records(docs,max_tokens=spec["window"],overlap=spec["overlap"]) if spec["unit"]=="chunks" else [dict(id=d,parent=d,start=0,end=len(tokenize(t)),text=t) for d,t in docs]
    if len(records)>128: raise ValueError("at most 128 retrieval units")
    omitted=[r for r in records if not tokenize(r["text"])];records=[r for r in records if tokenize(r["text"])]
    index=BM25Index([(r["id"],r["text"]) for r in records],k1=spec["k1"],b=spec["b"])
    query=tokenize(spec["query"]);qtf=Counter(query);units=[]
    terms=[dict(term=t,qtf=qtf[t],df=sum(t in tokenize(r["text"]) for r in records),idf=index._idf.get(t,0.0)) for t in sorted(qtf)]
    for i,r in enumerate(records):
        tokens=tokenize(r["text"]);counts=Counter(tokens);norm=1-spec["b"]+spec["b"]*len(tokens)/index._avg_length
        weights=[]
        for t in terms:
            tf=counts[t["term"]];denominator=tf+spec["k1"]*norm
            saturation=tf*(spec["k1"]+1)/denominator if tf else 0
            value=t["idf"]*saturation
            weights.append(dict(term=t["term"],tf=tf,df=t["df"],qtf=t["qtf"],idf=t["idf"],normalization=norm,denominator=denominator,saturation=saturation,per_occurrence=value,weighted=value*t["qtf"]))
        occurrences=[dict(term=m.group(),start=m.start(),end=m.end()) for m in TOKEN.finditer(r["text"].lower())]
        units.append(dict(**r,length=len(tokens),tokens=occurrences,weights=weights,score=index._score(query,i)))
    units.sort(key=lambda r:(-r["score"],r["id"]))
    for rank,r in enumerate(units,1):r["rank"]=rank
    if [(r["id"],r["score"]) for r in units]!=index.rank(spec["query"],top_k=len(units)):raise AssertionError("ranking mismatch")
    policies={}
    for policy in ("chunks","parents"):
        parents=[];witness=[]
        for r in units:
            eligible=r["rank"]<=spec["k"] if policy=="chunks" else len(parents)<spec["k"]
            duplicate=r["parent"] in parents;keep=eligible and not duplicate
            if keep:parents.append(r["parent"])
            witness.append(dict(rank=r["rank"],unit=r["id"],parent=r["parent"],within_budget=eligible,duplicate=duplicate,kept=keep,parent_rank=parents.index(r["parent"])+1 if r["parent"] in parents else None))
        policies[policy]=dict(parents=parents,collapse=witness,metrics=_metrics(parents,spec["grades"],spec["k"]))
    return dict(schema_version=1,spec=spec,N=index.size,avgdl=index._avg_length,query_tokens=query,terms=terms,units=units,omitted=omitted,policies=policies)
