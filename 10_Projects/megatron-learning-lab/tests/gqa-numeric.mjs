import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {compute, softmax, validateSelection} from '../web/gqa/compute.ts';
const fixture=JSON.parse(readFileSync(new URL('../content/fixtures/gqa-reference.json',import.meta.url),'utf8'));
const py=spawnSync('python',['-S','experiments/gqa_reference.py'],{encoding:'utf8',env:{...process.env,PYTHONPATH:''}});
assert.equal(py.status,0,py.stderr);
const references=JSON.parse(py.stdout);
let count=0, masked=0, maxError=0;
function compare(actual,expected,path='') {
  if(Array.isArray(expected)) {
    assert.equal(actual.length,expected.length,path);
    expected.forEach((value,i)=>compare(actual[i],value,`${path}[${i}]`));
  } else if(expected && typeof expected==='object') {
    assert.deepEqual(Object.keys(actual).sort(),Object.keys(expected).sort(),path);
    for(const key of Object.keys(expected)) compare(actual[key],expected[key],`${path}.${key}`);
  } else if(expected==='-Infinity') {
    assert.equal(actual,-Infinity,path); masked++;
  } else {
    assert(Number.isFinite(actual) && Number.isFinite(expected),path);
    const error=Math.abs(actual-expected);
    assert(error<=1e-10+1e-10*Math.abs(expected),`${path}: ${actual} != ${expected}`);
    count++;maxError=Math.max(maxError,error);
  }
}
for(const variant of ['qwen3','qwen25']) {
  const correct=compute(fixture,variant);
  compare(correct,references[variant],variant);
  for(let boundary=0;boundary<3;boundary++) {
    const changed=structuredClone(fixture);
    for(let t=boundary+1;t<4;t++) changed.x[t]=changed.x[t].map(x=>x+7);
    const after=compute(changed,variant);
    assert.deepEqual(after.residual.slice(0,boundary+1),correct.residual.slice(0,boundary+1));
    assert.notDeepEqual(after.residual[3],correct.residual[3]);
  }
  for(let t=0;t<4;t++) for(let h=0;h<4;h++) {
    const p=correct.probabilities[t][h];
    assert(Math.abs(p.reduce((a,b)=>a+b)-1)<1e-14);
    assert.equal(validateSelection(h,t),[0,0,1,1][h]);
    for(let j=0;j<4;j++) {
      assert.equal(correct.masked[t][h][j]===-Infinity,j>t);
      if(j>t) assert.equal(p[j],0);
      const score=correct.qrope[t][h].reduce((sum,x,i)=>sum+x*correct.krope[j][Math.floor(h/2)][i],0);
      assert.equal(correct.scores[t][h][j],score);
    }
    for(let i=0;i<4;i++) assert.equal(correct.heads[t][h][i],p.reduce((sum,x,j)=>sum+x*correct.v[j][Math.floor(h/2)][i],0));
  }
  for(const fault of ['omit_scale','wrong_group']) {
    const wrong=compute(fixture,variant,fault);
    const delta=Math.max(...correct.residual.flatMap((row,t)=>row.map((x,i)=>Math.abs(x-wrong.residual[t][i]))));
    assert(delta>1e-4,`${variant}/${fault} did not change the result`);
    console.log(`${variant}/${fault}: max residual difference=${delta}`);
  }
  assert.equal(correct.merged[0].length,16);
  assert.equal(correct.projected[0].length,8);
}
for(const invalid of [-1,4,3.5,NaN,Infinity,'',null,undefined,true]) {
  assert.throws(()=>validateSelection(invalid,0));
  assert.throws(()=>validateSelection(0,invalid));
}
for(const invalid of [NaN,Infinity,'',null]) {
  const bad=structuredClone(fixture);bad.x[0][0]=invalid;assert.throws(()=>compute(bad));
}
const empty=structuredClone(fixture);empty.x=[];assert.throws(()=>compute(empty));
for(const row of [[],[-Infinity,-Infinity],[NaN,0],[Infinity,0]]) assert.throws(()=>softmax(row));
assert.deepEqual(softmax([10000,10000,-Infinity]),[.5,.5,0]);
console.log(JSON.stringify({status:'passed',finite_values_compared:count,masked_positions_compared:masked,max_abs_error:maxError,atol:1e-10,rtol:1e-10,scope:'authored_float64_fixture_only',training_execution:'not_run'},null,2));
