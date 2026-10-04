import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {spawnSync} from "node:child_process";
import {pipeline,partition,mergeAttention,cpCompute,spCompute} from "../web/sequence/compute.ts";
const read=p=>JSON.parse(readFileSync(new URL(p,import.meta.url),"utf8"));
const model=read("../content/fixtures/decoder-reference.json"),data=read("../content/fixtures/sft-data.json");
const requests=[];
for(const pp of [1,2])for(let microbatches=1;microbatches<=8;microbatches++)requests.push({kind:"pipeline",pp,microbatches});
for(const layout of ["ordinary","thd"])for(const padding of [1,8])for(const cp of [1,2])if(!(layout==="thd"&&padding===1&&cp===2))requests.push({kind:"cp",layout,padding,cp});
for(const layer of [0,1])requests.push({kind:"sp",layer});
const env={...process.env,CUDA_VISIBLE_DEVICES:""};delete env.PYTHONPATH;
const p=spawnSync("python",["-m","experiments.sequence_reference"],{input:JSON.stringify(requests),encoding:"utf8",env,maxBuffer:32*1024*1024});
assert.equal(p.status,0,p.stderr);
const expected=JSON.parse(p.stdout);
let count=0,maxError=0;
function close(a,b){
 if(Array.isArray(b)){assert.equal(a.length,b.length);b.forEach((v,i)=>close(a[i],v));return;}
 assert(Number.isFinite(a)&&Number.isFinite(b));const e=Math.abs(a-b);assert(e<=1e-10+1e-10*Math.abs(b));count++;maxError=Math.max(maxError,e);
}
requests.forEach((r,i)=>{
 const e=expected[i];
 if(r.kind==="pipeline"){assert.deepEqual(pipeline(r.pp,r.microbatches),e);return;}
 if(r.kind==="cp"){
  const got=cpCompute(model,data,r.layout,r.padding,r.cp);
  for(const k of ["shards","positions","valid","segments","cuSeqlens","cuSeqlensPadded"])assert.deepEqual(got[k],e[k]);
  close(got.outputs,e.outputs);
  if(r.cp===2){const wrong=cpCompute(model,data,r.layout,r.padding,r.cp,"local_kv");assert.throws(()=>close(wrong.outputs,e.outputs));}
  if(r.layout==="thd"){const wrong=cpCompute(model,data,r.layout,r.padding,r.cp,"leak");assert.throws(()=>close(wrong.outputs,e.outputs));}
 }else{
  for(const tp of [1,2]){
   const s=spCompute(model,e.input,r.layer,tp);close(s.gathered,e.norm);close(s.output,e.output);
   assert.equal(s.scattered.length,tp);assert.equal(s.scattered[0].length,20/tp);
   if(tp===2)assert.throws(()=>close(spCompute(model,e.input,r.layer,tp,true).output,e.output));
  }
 }
});
assert.deepEqual(partition([0,8],2),[[0,1,6,7],[2,3,4,5]]);
assert.deepEqual(partition([0,4,8],2),[[0,3,4,7],[1,2,5,6]]);
close(mergeAttention([[0],[Math.log(3)]],[[[2,2,2,2]],[[4,4,4,4]]]).output,[3.5,3.5,3.5,3.5]);
close(mergeAttention([[10000],[10000]],[[[2,2,2,2]],[[4,4,4,4]]]).output,[3,3,3,3]);
assert.throws(()=>mergeAttention([[],[]],[[],[]]));
for(const value of [0,3,1.5,NaN,Infinity]){assert.throws(()=>pipeline(value,4));assert.throws(()=>partition([0,8],value));}
assert.throws(()=>cpCompute(model,data,"thd",1,2));
assert.throws(()=>spCompute(model,[[1,2,3,4,5,6,7,8]],0,2));
console.log(JSON.stringify({status:"passed",scenarios:requests.length,values_compared:count,max_abs_error:maxError,atol:1e-10,rtol:1e-10,scope:"CPU authored reference; logical PP, per-document CP and SP; no distributed backend"},null,2));
