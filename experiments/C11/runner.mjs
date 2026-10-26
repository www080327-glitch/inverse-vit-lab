// C11: no packages, models, datasets, network calls, or accelerators.
// Node >=20. Full-parameter tiny linear-bottleneck mechanism experiment.
// All numeric claims emitted by this program are TOY evidence only.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {performance} from 'node:perf_hooks';
import assert from 'node:assert/strict';
const OUT=process.argv[2] || 'experiments/C11/results';
fs.mkdirSync(OUT,{recursive:true});
const START=performance.now(), BUDGET=1750, SEEDS=[11,29];
const CFG={seeds:SEEDS,train:2048,validation:1024,pretrain_steps:1200,
 new_steps:800,aux_steps:200,new_batch:32,old_batch:16,pretrain_batch:32,
 lr:0.01,temperature:2,ce_weight:1,kd_weight:1,diagnostic_every_new_steps:50,
 input:8,bottleneck:4,old_discrete_outputs:4,new_continuous_outputs:16,
 checkpoint_rule:'fixed final; no validation selection',training_processes:1};
function rng(seed){let a=seed>>>0;return ()=>{a+=0x6D2B79F5;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};}
function dataset(n,seed){const r=rng(seed);return Array.from({length:n},()=>Array.from({length:8},()=>{let z=r();return (z<.5?-1:1)*(.1+.9*r());}));}
function ids(n,k,r){return Array.from({length:k},()=>Math.floor(r()*n));}
function hash(x){return crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');}
const shapes={W:32,b:4,O:16,ob:4,V:64,vb:16};
function model(seed){const r=rng(seed);return Object.fromEntries(Object.entries(shapes).map(([k,n])=>[k,Array.from({length:n},()=>k.includes('b')?0:(r()-.5)*.4)]));}
function copy(m){return JSON.parse(JSON.stringify(m));}
function zeros(){return Object.fromEntries(Object.entries(shapes).map(([k,n])=>[k,Array(n).fill(0)]));}
function hidden(m,x){return m.b.map((b,i)=>b+x.reduce((a,v,j)=>a+m.W[i*8+j]*v,0));}
function head(w,b,h){return b.map((v,i)=>v+h.reduce((a,x,j)=>a+w[i*4+j]*x,0));}
function sigmoid(x){return x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));}
function softplus(x){return Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));}
function field(x,rel){const a=rel==='compatible'?0:4;return Array.from({length:16},(_,i)=>{let s=0;for(let j=0;j<4;j++)s+=(((i>>j)&1)?1:-1)*.5*x[a+j];return s;});}
// Bernoulli KL, positive in exact arithmetic. Distillation uses T^2 scaling.
function lossGrad(m,data,idx,kind,rel,teacher=null){
 const g=zeros();let loss=0;
 const old=kind!=='new', D=old?4:16, T=kind==='kd'?CFG.temperature:1;
 for(const id of idx){
  const x=data[id],h=hidden(m,x), w=old?m.O:m.V,b=old?m.ob:m.vb;
  const z=head(w,b,h), target=old?x.slice(0,4).map(v=>+(v>0)):field(x,rel);
  const tz=kind==='kd'?head(teacher.O,teacher.ob,hidden(teacher,x)):null;
  const dh=Array(4).fill(0);
  for(let i=0;i<D;i++){
   let dz;
   if(kind==='new'){let e=z[i]-target[i];loss+=e*e/(idx.length*D);dz=2*e/(idx.length*D);}
   else if(kind==='ce'){loss+=(softplus(z[i])-target[i]*z[i])/(idx.length*D);dz=(sigmoid(z[i])-target[i])/(idx.length*D);}
   else {let q=sigmoid(tz[i]/T),p=sigmoid(z[i]/T);loss+=T*T*(softplus(z[i]/T)-q*z[i]/T-softplus(tz[i]/T)+q*tz[i]/T)/(idx.length*D);dz=T*(p-q)/(idx.length*D);}
   const wk=old?'O':'V',bk=old?'ob':'vb';
   g[bk][i]+=dz;
   for(let j=0;j<4;j++){g[wk][i*4+j]+=dz*h[j];dh[j]+=dz*w[i*4+j];}
  }
  for(let j=0;j<4;j++){g.b[j]+=dh[j];for(let k=0;k<8;k++)g.W[j*8+k]+=dh[j]*x[k];}
 }
 return {loss,g,kind};
}
function add(a,b){let g=zeros();for(let k in g)g[k]=a.g[k].map((v,i)=>v+b.g[k][i]);return {loss:a.loss+b.loss,g,kind:'both'};}
function optimizer(){return {m:zeros(),v:zeros(),t:Object.fromEntries(Object.keys(shapes).map(k=>[k,0]))};}
function step(m,o,r){
 const keys=r.kind==='new'?['W','b','V','vb']:['W','b','O','ob'];
 for(let k of keys){let t=++o.t[k];for(let i=0;i<m[k].length;i++){
  let g=r.g[k][i];o.m[k][i]=.9*o.m[k][i]+.1*g;o.v[k][i]=.999*o.v[k][i]+.001*g*g;
  m[k][i]-=CFG.lr*(o.m[k][i]/(1-.9**t))/(Math.sqrt(o.v[k][i]/(1-.999**t))+1e-8);
 }}
}
function shared(r){return [...r.g.W,...r.g.b];}
function norm(a){return Math.sqrt(a.reduce((s,v)=>s+v*v,0));}
function cosine(a,b){let den=norm(a)*norm(b);return den<1e-14?null:a.reduce((s,v,i)=>s+v*b[i],0)/den;}
function diagnostic(m,t,train,probe,rel,seed,arm,n){
 let a=lossGrad(m,train,probe,'new',rel),b=lossGrad(m,train,probe,'ce',rel),c=lossGrad(m,train,probe,'kd',rel,t);
 let ag=shared(a),bg=shared(b),cg=shared(c);
 return {seed,relation:rel,arm,new_updates:n,new_mse:a.loss,old_ce:b.loss,kd_T2:c.loss,
  new_shared_norm:norm(ag),old_ce_shared_norm:norm(bg),kd_shared_norm:norm(cg),
  cosine_new_ce:cosine(ag,bg),cosine_new_kd:cosine(ag,cg),
  cosine_new_combined_preservation:cosine(ag,shared(add(b,c))),
  actual_preservation_norm:arm==='A'?0:arm==='B'?norm(bg):arm==='C'?norm(cg):norm(shared(add(b,c)))};
}
function evaluate(m,data,rel,t){
 let exact=0,tokens=0,ce=0,mse=0,base=0,kept=0,failures=[],newFailures=[];
 for(let id=0;id<data.length;id++){
  let x=data[id],h=hidden(m,x),z=head(m.O,m.ob,h),v=head(m.V,m.vb,h),target=x.slice(0,4).map(v=>+(v>0)),pred=z.map(v=>+(v>0));
  let ok=pred.every((v,i)=>v===target[i]);exact+=+ok;tokens+=pred.filter((v,i)=>v===target[i]).length;
  for(let i=0;i<4;i++)ce+=(softplus(z[i])-target[i]*z[i])/(data.length*4);
  let tz=head(t.O,t.ob,hidden(t,x)),tp=tz.map(v=>+(v>0)),was=tp.every((v,i)=>v===target[i]);
  base+=+was;kept+=+(was&&ok);
  let y=field(x,rel),e=v.reduce((s,v,i)=>s+(v-y[i])**2,0)/16;mse+=e/data.length;
  if(was&&!ok&&failures.length<8)failures.push({validation_index:id,input:x,target,teacher_prediction:tp,student_prediction:pred,student_logits:z});
  newFailures.push({validation_index:id,new_mse:e,input:x,target:y,prediction:v});
 }
 newFailures.sort((a,b)=>b.new_mse-a.new_mse);
 return {old_exact:exact/data.length,old_token_accuracy:tokens/(data.length*4),old_ce:ce,new_mse:mse,
  teacher_correct:base,retained_correct:kept,retention:base?kept/base:null,old_failures:failures,new_failures:newFailures.slice(0,3)};
}
function checkDeadline(){assert((performance.now()-START)/1000<BUDGET,'aggregate experiment exceeded time budget');}
function finiteDifference(){
 const m=model(313),d=dataset(3,99),t=model(98);let maxRel=0,maxAbs=0,checked=0;
 for(let kind of ['new','ce','kd']){
  const a=lossGrad(m,d,[0,1,2],kind,'conflict',t);
  for(let k of ['W','b',...(kind==='new'?['V','vb']:['O','ob'])])for(let i=0;i<m[k].length;i+=Math.max(1,Math.floor(m[k].length/5))){
   const v=m[k][i],eps=1e-5;m[k][i]=v+eps;let p=lossGrad(m,d,[0,1,2],kind,'conflict',t).loss;
   m[k][i]=v-eps;let q=lossGrad(m,d,[0,1,2],kind,'conflict',t).loss;m[k][i]=v;
   let num=(p-q)/(2*eps),an=a.g[k][i],err=Math.abs(num-an);maxAbs=Math.max(maxAbs,err);maxRel=Math.max(maxRel,err/Math.max(1e-6,Math.abs(num)+Math.abs(an)));checked++;
  }
 }
 assert(maxAbs<1e-7);assert(maxRel<1e-4);
 assert(Math.abs(lossGrad(t,d,[0,1,2],'kd','compatible',t).loss)<1e-12);
 // Diagnostics are read-only; no optimizer buffers or teacher/student arrays are changed.
 const before=hash(m),teacherBefore=hash(t);diagnostic(m,t,d,[0,1,2],'compatible',0,'D',0);
 assert.equal(hash(m),before);assert.equal(hash(t),teacherBefore);
 return {checked,max_absolute_error:maxAbs,max_relative_error:maxRel,self_kd_zero:true,diagnostic_non_mutation:true};
}
const tests=finiteDifference(),rows=[],traces=[],failures=[],histories=[],pretraining=[];
const affinity=process.env.C11_CPU_AFFINITY||'not reported';
for(const seed of SEEDS){
 checkDeadline();let train=dataset(CFG.train,seed+1000),val=dataset(CFG.validation,seed+2000),rr=rng(seed+3000);
 let m=model(seed),opt=optimizer(),preStart=performance.now();
 for(let n=0;n<CFG.pretrain_steps;n++){step(m,opt,lossGrad(m,train,ids(train.length,CFG.pretrain_batch,rr),'ce','compatible'));if(n%200===0)checkDeadline();}
 const preSeconds=(performance.now()-preStart)/1000,teacher=copy(m),teacherHash=hash(teacher),snapshotHash=hash(m);
 for(let k in teacher)Object.freeze(teacher[k]);Object.freeze(teacher);
 const base=evaluate(m,val,'compatible',teacher);
 assert(base.old_exact>.95,'pretraining failed fixed criterion; report failure rather than tuning validation');
 pretraining.push({seed,seconds:preSeconds,old_exact:base.old_exact,old_ce:base.old_ce,snapshot_hash:snapshotHash,train_hash:hash(train),validation_hash:hash(val),supervised_samples:CFG.pretrain_steps*CFG.pretrain_batch});
 const sr=rng(seed+4000),newSchedule=Array.from({length:800},()=>ids(train.length,32,sr)),auxSchedule=Array.from({length:200},()=>ids(train.length,16,sr)),probe=ids(train.length,64,rng(seed+5000));
 const scheduleHash=hash({newSchedule,auxSchedule,probe});
 for(const rel of ['compatible','conflict'])for(const arm of ['A','B','C','D']){
  checkDeadline();let student=copy(m),o=optimizer(),diagSeconds=0,teacherSeconds=0,newSeconds=0,auxSeconds=0,newLoss=0,ceLoss=0,kdLoss=0,gradNew=0,gradPres=0;
  const timer=performance.now();let initial=evaluate(student,val,rel,teacher);
  function record(n){let s=performance.now();traces.push(diagnostic(student,teacher,train,probe,rel,seed,arm,n));diagSeconds+=(performance.now()-s)/1000;}
  record(0);
  // Four new updates then one optional preservation update: 200 groups.
  for(let group=0;group<200;group++){
   for(let inner=0;inner<4;inner++){
    let n=group*4+inner,s=performance.now(),r=lossGrad(student,train,newSchedule[n],'new',rel);
    newLoss+=r.loss;gradNew+=norm(shared(r));step(student,o,r);newSeconds+=(performance.now()-s)/1000;
    if((n+1)%50===0 && n+1<800)record(n+1);
   }
   if(arm!=='A'){
    let idsOld=auxSchedule[group],s=performance.now(),r=null,ce=null,kd=null;
    if(arm==='B'||arm==='D'){ce=lossGrad(student,train,idsOld,'ce',rel);ceLoss+=ce.loss;}
    if(arm==='C'||arm==='D'){
     let ts=performance.now(); // Teacher only forward, no gradients; precompute per-batch logits.
     let cached={};
     for(let id of idsOld)cached[id]=head(teacher.O,teacher.ob,hidden(teacher,train[id]));
     teacherSeconds+=(performance.now()-ts)/1000;
     // Frozen copy with an explicit cache is passed through a dedicated KD function.
     kd=kdCached(student,train,idsOld,cached);kdLoss+=kd.loss;
    }
    r=ce&&kd?add(ce,kd):ce||kd;gradPres+=norm(shared(r));step(student,o,r);auxSeconds+=(performance.now()-s)/1000;
   }
   if((group+1)%50===0){checkDeadline();let ev=evaluate(student,val,rel,teacher);histories.push({seed,relation:rel,arm,new_updates:(group+1)*4,old_exact:ev.old_exact,retention:ev.retention,new_mse:ev.new_mse});}
  }
  record(800);
  assert.equal(hash(teacher),teacherHash,'teacher mutated');
  let final=evaluate(student,val,rel,teacher),total=(performance.now()-timer)/1000;
  let row={seed,relation:rel,arm,parameters:Object.values(shapes).reduce((a,b)=>a+b,0),
   baseline_old_exact:base.old_exact,initial_new_mse:initial.new_mse,old_exact:final.old_exact,old_token_accuracy:final.old_token_accuracy,retention:final.retention,teacher_correct:final.teacher_correct,retained_correct:final.retained_correct,
   old_ce:final.old_ce,new_mse:final.new_mse,new_mse_improvement:initial.new_mse-final.new_mse,
   mean_train_new_mse:newLoss/800,mean_train_old_ce:arm==='B'||arm==='D'?ceLoss/200:null,mean_train_kd_T2:arm==='C'||arm==='D'?kdLoss/200:null,
   new_updates:800,preservation_updates:arm==='A'?0:200,total_updates:arm==='A'?800:1000,
   new_supervised_samples:25600,old_labeled_samples:arm==='B'||arm==='D'?3200:0,old_distilled_samples:arm==='C'||arm==='D'?3200:0,old_input_presentations:arm==='A'?0:3200,
   new_supervised_scalar_positions:409600,old_supervised_discrete_tokens:arm==='B'||arm==='D'?12800:0,old_distilled_positions:arm==='C'||arm==='D'?12800:0,
   summed_new_shared_grad_norms:gradNew,summed_preservation_shared_grad_norms:gradPres,
   preservation_gradient_norm_share:gradPres/(gradPres+gradNew),
   seconds_total:total,seconds_new_updates:newSeconds,seconds_preservation_updates:auxSeconds,seconds_training_teacher_forward:teacherSeconds,seconds_diagnostics:diagSeconds,
   paired_snapshot_hash:snapshotHash,paired_schedule_hash:scheduleHash,teacher_frozen_verified:true};
  rows.push(row);failures.push({seed,relation:rel,arm,old_failure_count:final.teacher_correct-final.retained_correct,examples:firstExamples(final)});
  console.log(JSON.stringify({seed,relation:rel,arm,old_exact:row.old_exact,retention:row.retention,new_mse:row.new_mse,seconds:total}));
 }
}
// Same KD as lossGrad but teacher logits were already computed and timed separately.
function kdCached(m,data,idx,cache){
 const g=zeros(),T=CFG.temperature;let loss=0;
 for(let id of idx){let x=data[id],h=hidden(m,x),z=head(m.O,m.ob,h),tz=cache[id],dh=Array(4).fill(0);
  for(let i=0;i<4;i++){let q=sigmoid(tz[i]/T),p=sigmoid(z[i]/T),dz=T*(p-q)/(idx.length*4);
   loss+=T*T*(softplus(z[i]/T)-q*z[i]/T-softplus(tz[i]/T)+q*tz[i]/T)/(idx.length*4);
   g.ob[i]+=dz;for(let j=0;j<4;j++){g.O[i*4+j]+=dz*h[j];dh[j]+=dz*m.O[i*4+j];}}
  for(let j=0;j<4;j++){g.b[j]+=dh[j];for(let k=0;k<8;k++)g.W[j*8+k]+=dh[j]*x[k];}}
 return {loss,g,kind:'kd'};
}
function firstExamples(ev){return {previously_correct_now_wrong:ev.old_failures,worst_new_task:ev.new_failures};}
function write(name,s){fs.writeFileSync(path.join(OUT,name),s);}
function csv(arr){const keys=Object.keys(arr[0]);return keys.join(',')+'\n'+arr.map(r=>keys.map(k=>r[k]===null||r[k]===undefined?'':typeof r[k]==='string'?JSON.stringify(r[k]):String(r[k])).join(',')).join('\n')+'\n';}
const result={status:'completed',evidence_scope:'toy mechanism evidence only; no Qwen retention claim',config:CFG,tests,
 environment:{runtime:process.version,platform:os.platform(),arch:os.arch(),cpu_affinity:affinity,training_processes:1,source_sha256:crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex')},
 pretraining,rows,aggregate_wall_seconds:(performance.now()-START)/1000,
 limitations:['linear 4D bottleneck deliberately exposes capacity competition','two seeds, no significance claim','C and D have frozen teacher forward overhead; diagnostic overhead separately recorded','A holds new-data exposure fixed but has fewer total optimizer steps','D has stronger total preservation loss than B/C; not a matched-gradient comparison','gradient norm shares are not Adam update attribution','validation used only for fixed checks/reporting, no hyperparameter selection','vector positions are not LLM tokens or image quality','no full Qwen, no LoRA comparison, no formal replay-ratio recommendation']};
write('RESULT.json',JSON.stringify(result,null,2)+'\n');write('results.csv',csv(rows));write('gradient_diagnostics.csv',csv(traces));write('learning_curves.csv',csv(histories));write('failures.json',JSON.stringify(failures,null,2)+'\n');
const md=['# C11 结果表：仅 toy 机制证据','','旧任务为 4 位离散序列，新增任务为 16 维连续场。所有百分比均为验证集最终快照。','',
'|关系|seed|方案|旧 exact %|原答对保留 %|新 MSE|旧 CE|新更新秒|旧保持秒|教师前向秒|',
'|---|---:|---|---:|---:|---:|---:|---:|---:|---:|'];
for(let r of rows)md.push('|'+[r.relation,r.seed,r.arm,(100*r.old_exact).toFixed(2),(100*r.retention).toFixed(2),r.new_mse.toFixed(6),r.old_ce.toFixed(4),r.seconds_new_updates.toFixed(3),r.seconds_preservation_updates.toFixed(3),r.seconds_training_teacher_forward.toFixed(3)].join('|')+'|');
md.push('','A=new；B=new+真实旧重放；C=new+冻结旧教师 KL；D=new+重放+KL。','',
'## 20% 更新步骤不等于 20% 监督量或梯度贡献','',
'B/C/D 的旧保持步骤是 200/1000=20%；旧输入样本呈现 3200/(25600+3200)=11.11%。',
'B 的旧离散监督位置是 12800/(409600+12800)=3.03%；新增 16 维连续位置不能等同于自然语言 Token。',
'D 同一旧样本同时接受 CE 和 KL，不能当作两次独立样本。C 不使用旧标签。',
'若只比较两类输入长度相同的 8 个标量，旧输入位置占比为 11.11%，仍非 20%。',
'下面梯度比例是每步共享梯度范数之和占比，受方向、顺序和 Adam 状态影响，不能当作精确参数更新归因。','',
'|关系|seed|方案|保持步骤 %|保持梯度范数和占比 %|保持训练耗时占比 %|','|---|---:|---|---:|---:|---:|');
for(let r of rows.filter(x=>x.arm!=='A'))md.push('|'+[r.relation,r.seed,r.arm,'20.00',(100*r.preservation_gradient_norm_share).toFixed(2),(100*r.seconds_preservation_updates/(r.seconds_new_updates+r.seconds_preservation_updates)).toFixed(2)].join('|')+'|');
md.push('','教师仅前向时间已包含在旧保持训练时间内，不能再次相加；诊断时的额外教师开销包含于 diagnostics。完整字段见 CSV/JSON。','',
'## 结论边界','',
'这是刻意限制共享表示秩的机制实验，不是模型规模、真实指令分布或视觉生成质量实验。请同时读取旧保留率、新 MSE、轨迹和梯度，不能仅凭余弦符号决定是否训练。',
'没有检验 LoRA；没有据此选定正式重放比例；没有证明真实 Qwen 旧能力保持。');
write('RESULTS.md',md.join('\n')+'\n');
console.log('C11_RESULT '+JSON.stringify({status:result.status,seconds:result.aggregate_wall_seconds,tests,rows:rows.length}));
