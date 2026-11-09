// Independent scalar retrieval arithmetic. No Python execution or stored-score lookup.
const field=(s,k,d)=>Object.hasOwn(s,k)?s[k]:d;
const compare=(a,b)=>a<b?-1:a>b?1:0;
const tokens=s=>(s.toLowerCase().match(/[a-z0-9]+/g)||[]);
function integer(v,a,b,label){if(!Number.isInteger(v)||typeof v==='boolean'||v<a||v>b)throw Error(label+' outside bounds');return v;}
function text(v,n,label){if(typeof v!=='string'||v.length>n||/[^\x09\x0a\x0d\x20-\x7f]/.test(v))throw Error(label+' must be bounded ASCII');return v;}
export function validate(s){
 if(!s||typeof s!=='object'||Array.isArray(s)||Object.keys(s).some(k=>!['documents','query','grades','k','k1','b','window','overlap','unit','budget'].includes(k)))throw Error('unknown spec fields');
 if(!Array.isArray(s.documents)||s.documents.length<1||s.documents.length>16)throw Error('need 1..16 documents');
 const seen=new Set();let total=0;
 const documents=s.documents.map(d=>{if(!d||Object.keys(d).sort().join(',')!=='id,text')throw Error('document id/text required');const id=text(d.id,64,'id');if(!id||seen.has(id))throw Error('nonempty unique ids required');seen.add(id);const t=text(d.text,2048,'document');total+=tokens(t).length;return {id,text:t};});
 if(total>1024)throw Error('at most 1024 corpus tokens');
 const query=text(s.query,512,'query');if(tokens(query).length>64)throw Error('at most 64 query tokens');
 const grades=field(s,'grades',{});if(!grades||typeof grades!=='object'||Array.isArray(grades)||Object.keys(grades).length>32)throw Error('at most 32 judgments');for(const [id,g] of Object.entries(grades)){if(!text(id,64,'grade id'))throw Error('empty id');integer(g,0,8,'grade');}
 const k=integer(field(s,'k',3),1,32,'k'),window=integer(field(s,'window',12),1,256,'window'),overlap=integer(field(s,'overlap',2),0,window-1,'overlap'),k1=field(s,'k1',1.5),b=field(s,'b',.75);
 if(typeof k1!=='number'||!Number.isFinite(k1)||k1<.01||k1>100||typeof b!=='number'||!Number.isFinite(b)||b<0||b>1)throw Error('finite k1 [.01,100], b [0,1]');
 const unit=field(s,'unit','documents'),budget=field(s,'budget','chunks');if(!['documents','chunks'].includes(unit)||!['chunks','parents'].includes(budget))throw Error('invalid unit/budget');
 return {documents,query,grades,k,window,overlap,k1,b,unit,budget};
}
function metric(parents,grades,k){
 const relevant=Object.keys(grades).filter(d=>grades[d]>0).sort(compare),ideal=Object.keys(grades).sort((a,b)=>grades[b]-grades[a]||compare(a,b)).slice(0,k);
 const idealRows=ideal.map((d,i)=>({rank:i+1,doc:d,grade:grades[d],gain:2**grades[d]-1,discount:1/Math.log2(i+2),contribution:(2**grades[d]-1)/Math.log2(i+2)}));
 const idcg=idealRows.reduce((a,r)=>a+r.contribution,0);let dcg=0,hits=0,rr=0;
 const rows=parents.slice(0,k).map((d,i)=>{const grade=Object.hasOwn(grades,d)?grades[d]:0,hit=grade>0;hits+=hit?1:0;const gain=2**grade-1,discount=1/Math.log2(i+2),contribution=gain*discount;dcg+=contribution;if(hit&&!rr)rr=1/(i+1);return {rank:i+1,doc:d,judged:Object.hasOwn(grades,d),grade,hit,gain,discount,contribution,cumulative_dcg:dcg,cumulative_hits:hits};});
 return {rows,ideal:idealRows,relevant,dcg,idcg,values:{precision:hits/k,recall:relevant.length?hits/relevant.length:0,rr,ndcg:idcg?dcg/idcg:0},included_in_macro:!!relevant.length};
}
export function audit(input){
 const spec=validate(input);let records=[];const ids=new Set();
 for(const d of spec.documents){const ts=tokens(d.text),starts=[];if(spec.unit==='documents'||ts.length<=spec.window)starts.push(0);else for(let n=0;n<ts.length;n+=spec.window-spec.overlap)starts.push(n);
  starts.forEach((start,i)=>{const whole=spec.unit==='documents'||ts.length<=spec.window,id=whole?d.id:d.id+'#c'+i;if(ids.has(id))throw Error('chunk id collision');ids.add(id);const end=whole?ts.length:Math.min(ts.length,start+spec.window);records.push({id,parent:d.id,start,end,text:whole?d.text:ts.slice(start,end).join(' ')});});
 }
 if(records.length>128)throw Error('at most 128 retrieval units');const omitted=records.filter(r=>!tokens(r.text).length);records=records.filter(r=>tokens(r.text).length);if(!records.length)throw Error('corpus has no non-empty documents');
 const N=records.length,avgdl=records.reduce((a,r)=>a+tokens(r.text).length,0)/N,query=tokens(spec.query),qtf=new Map();for(const t of query)qtf.set(t,(qtf.get(t)||0)+1);
 const terms=[...qtf.keys()].sort(compare).map(term=>{const df=records.filter(r=>tokens(r.text).includes(term)).length;return {term,qtf:qtf.get(term),df,idf:df?Math.max(0,Math.log((N-df+.5)/(df+.5)+1)):0};});
 const units=records.map(r=>{const ts=tokens(r.text),counts=new Map();for(const t of ts)counts.set(t,(counts.get(t)||0)+1);const norm=1-spec.b+spec.b*ts.length/avgdl;
  const weights=terms.map(t=>{const tf=counts.get(t.term)||0,denominator=tf+spec.k1*norm,saturation=tf?tf*(spec.k1+1)/denominator:0,value=t.idf*saturation;return {term:t.term,tf,df:t.df,qtf:t.qtf,idf:t.idf,normalization:norm,denominator,saturation,per_occurrence:value,weighted:value*t.qtf};});
  let score=0;for(const t of query){const tf=counts.get(t)||0;if(tf){const idf=terms.find(x=>x.term===t).idf;score+=idf*(tf*(spec.k1+1))/(tf+spec.k1*norm);}}
  return {...r,length:ts.length,tokens:[...r.text.toLowerCase().matchAll(/[a-z0-9]+/g)].map(m=>({term:m[0],start:m.index,end:m.index+m[0].length})),weights,score};
 });units.sort((a,b)=>b.score-a.score||compare(a.id,b.id));units.forEach((r,i)=>r.rank=i+1);
 const policies={};for(const policy of ['chunks','parents']){const parents=[],collapse=[];for(const r of units){const within_budget=policy==='chunks'?r.rank<=spec.k:parents.length<spec.k,duplicate=parents.includes(r.parent),kept=within_budget&&!duplicate;if(kept)parents.push(r.parent);collapse.push({rank:r.rank,unit:r.id,parent:r.parent,within_budget,duplicate,kept,parent_rank:parents.includes(r.parent)?parents.indexOf(r.parent)+1:null});}policies[policy]={parents,collapse,metrics:metric(parents,spec.grades,spec.k)};}
 return {schema_version:1,spec,N,avgdl,query_tokens:query,terms,units,omitted,policies};
}
