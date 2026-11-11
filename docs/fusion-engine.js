/* Independent source-column accumulation using exact BigInt rational arithmetic.
 * Standard weighted RRF, Cormack/Clarke/Buettcher SIGIR 2009. See FUSION.md. */
export const SCHEMA='retrievalbench-rrf/v1';
const fail=m=>{throw Error(m)},gcd=(a,b)=>b?gcd(b,a%b):a;
function rat(n=0n,d=1n){const g=gcd(n<0n?-n:n,d);return [n/g,d/g]}
const plus=(a,b)=>rat(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
const packed=a=>({n:String(a[0]),d:String(a[1])});
const compare=(a,b)=>a[0]*b[1]<b[0]*a[1]?-1:a[0]*b[1]>b[0]*a[1]?1:0;
const integer=(v,l,h)=>typeof v==='number'&&Number.isInteger(v)&&v>=l&&v<=h?v:fail('bounded integer required');
const id=v=>typeof v==='string'&&v.length>=1&&v.length<=64&&/^[\x20-\x7e]+$/.test(v)?v:fail('printable ASCII id required');
const keys=(v,k)=>{if(!v||Array.isArray(v)||typeof v!=='object'||Object.keys(v).sort().join('\0')!==[...k].sort().join('\0'))fail('unexpected object fields')};
export function validate(input){
 keys(input,['schema','constant','k','lists']);if(input.schema!==SCHEMA)fail('unsupported schema');
 const constant=integer(input.constant,0,1000000),k=integer(input.k,0,64),universe=new Set(),names=new Set();
 if(!Array.isArray(input.lists)||input.lists.length<1||input.lists.length>8)fail('need 1..8 lists');
 const lists=input.lists.map(s=>{keys(s,['id','weight','depth','ranking']);const name=id(s.id);if(names.has(name))fail('duplicate list id');names.add(name);
 const weight=integer(s.weight,0,100),depth=integer(s.depth,0,64);
 if(!Array.isArray(s.ranking)||s.ranking.length>64)fail('need 0..64 ids');const ranking=s.ranking.map(id);if(new Set(ranking).size!==ranking.length)fail('duplicate document');for(const d of ranking)universe.add(d);return {id:name,weight,depth,ranking};});
 if(universe.size>64)fail('at most 64 unique documents');if(!lists.some(s=>s.weight>0))fail('need positive weight');return {schema:SCHEMA,constant,k,lists};
}
export function fuse(input){
 const spec=validate(input),scores=new Map(),columns=[];
 // Source-first walk, unlike the native row-first Fraction implementation.
 for(const s of spec.lists){const column=new Map();s.ranking.forEach((doc,i)=>{const rank=i+1,admitted=rank<=s.depth;if(admitted){if(!scores.has(doc))scores.set(doc,rat());scores.set(doc,plus(scores.get(doc),rat(BigInt(s.weight),BigInt(spec.constant+rank))))}column.set(doc,{rank,admitted});});columns.push(column);}
 const sorted=[...scores].sort(([a,x],[b,y])=>compare(y,x)||(a<b?-1:a>b?1:0));
 const rows=sorted.map(([doc,score],i)=>({doc,score:packed(score),contributions:spec.lists.map((s,n)=>{const entry=columns[n].get(doc),rank=entry?.rank??null,admitted=entry?.admitted??false;return {source:s.id,rank,admitted,reason:admitted?'included':rank===null?'missing':'beyond_depth',weight:s.weight,denominator:admitted?spec.constant+rank:null,contribution:packed(admitted?rat(BigInt(s.weight),BigInt(spec.constant+rank)):rat())}}),rank:i+1,selected:i<spec.k}));
 let cutoff=null;if(spec.k>0&&spec.k<rows.length){const [a,x]=sorted[spec.k-1],[b,y]=sorted[spec.k];cutoff={inside:a,outside:b,difference:packed(plus(x,[-y[0],y[1]])),tie:compare(x,y)===0};}
 const identical_rankings=[];spec.lists.forEach((s,i)=>spec.lists.slice(0,i).forEach(t=>{if(s.depth===t.depth&&JSON.stringify(s.ranking)===JSON.stringify(t.ranking))identical_rankings.push([t.id,s.id]);}));
 return {schema:'retrievalbench-rrf-report/v1',spec,sources:spec.lists.map(s=>({id:s.id,weight:s.weight,depth:s.depth,entries:s.ranking.map((doc,i)=>({doc,rank:i+1,admitted:i<s.depth,candidate:scores.has(doc)}))})),rows,selected:rows.filter(r=>r.selected).map(r=>r.doc),excluded:[...new Set(spec.lists.flatMap(s=>s.ranking))].filter(d=>!scores.has(d)).sort(),identical_rankings,cutoff,upper_bound:packed(spec.lists.reduce((a,s)=>plus(a,rat(BigInt(s.weight),BigInt(spec.constant+1))),rat()))};
}
export function parseSpec(text){
 if(typeof text!=='string'||new TextEncoder().encode(text).length>131072)fail('input exceeds 128 KiB');
 // Small recursive JSON reader: reject duplicate keys and every non-integer
 // numeric spelling before JSON.parse can discard that evidence.
 let p=0;const ws=()=>{while(/[\t\n\r ]/.test(text[p]||'!'))p++};
 const string=()=>{const start=p++;let escaped=false;for(;p<text.length;p++){const c=text[p];if(c==='"'&&!escaped){p++;return JSON.parse(text.slice(start,p))}if(c==='\\'&&!escaped)escaped=true;else escaped=false;}fail('unterminated string')};
 function value(depth=0){if(depth>16)fail('JSON nesting too deep');ws();const c=text[p];
 if(c==='"')return string();if(c==='{'){p++;ws();const result=Object.create(null),seen=new Set();if(text[p]==='}'){p++;return result}for(;;){ws();if(text[p]!=='"')fail('object key required');const key=string();if(seen.has(key))fail('duplicate JSON key');seen.add(key);ws();if(text[p++]!==':')fail('colon required');result[key]=value(depth+1);ws();const next=text[p++];if(next==='}')return result;if(next!==',')fail('comma required');}}
 if(c==='['){p++;ws();const result=[];if(text[p]===']'){p++;return result}for(;;){result.push(value(depth+1));ws();const next=text[p++];if(next===']')return result;if(next!==',')fail('comma required');}}
 for(const [word,v] of [['true',true],['false',false],['null',null]])if(text.slice(p,p+word.length)===word){p+=word.length;return v}
 const m=/^-?(?:0|[1-9][0-9]*)/.exec(text.slice(p));if(!m)fail('invalid JSON value');p+=m[0].length;const n=Number(m[0]);if(!Number.isSafeInteger(n))fail('integer outside safe range');return n;
 }const result=value();ws();if(p!==text.length)fail('invalid JSON suffix or non-integer number');return result;
}
