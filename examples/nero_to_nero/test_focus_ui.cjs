// 对焦交互独立测试：真实控件逻辑，模拟 DOM 和设置请求。
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
class Element {
  constructor(){this.children=[];this.events={};this.disabled=false;this.value='';this.checked=false;}
  append(...nodes){this.children.push(...nodes);}
  setAttribute(name,value){this[name]=value;}
  addEventListener(name,fn){this.events[name]=fn;}
}
const context=vm.createContext({document:{createElement:()=>new Element()}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'nero_platform/static/camera_focus.js'),'utf8'),context);
const requests=[]; let finish,fail;
const controller=new context.NeroCameraFocus({
  post:(url,setting)=>{requests.push({url,setting});return new Promise((resolve,reject)=>{finish=()=>resolve({setting,focus:check(setting,100)});fail=()=>reject(new Error('USB 写入失败'));});},
  refresh:async()=>{},
});
function check(setting,max){return {setting,values:{focus_auto:setting.mode==='auto'?1:0,focus_absolute:setting.value},
  limits:{focus_auto:{min:0,max:1,step:1,default:1,flags:0},focus_absolute:{min:0,max,step:1,default:50,flags:0}}};}
const a={mode:'auto',value:50}, b={mode:'manual',value:128};
let status={connected:false,device_busy:[],camera_focus:{A:check(a,100),B:check(b,255),C:{values:{},limits:{}}},camera_focus_settings:{A:a,B:b}};
controller.load(status.camera_focus_settings);
for(const id of ['A','B','C'])controller.add(new Element(),id,{type:'rgb'});
controller.add(new Element(),'D',{type:'rgbd'});
controller.update(status,true);
const A=controller.rows.get('A'), B=controller.rows.get('B'), C=controller.rows.get('C'),D=controller.rows.get('D');
const event=(element,name)=>element.events[name]();
const settle=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  assert.equal(A.auto.checked,true);assert.equal(A.manual.disabled,true);assert.equal(A.slider.disabled,true);
  assert.equal(A.slider.max,100);assert.equal(B.slider.max,255);assert.equal(B.auto.disabled,true);
  assert.equal(C.manual.disabled,true);assert.equal(C.auto.disabled,true);assert.equal(D.manual.disabled,true);
  A.auto.checked=false;event(A.auto,'change');
  assert.equal(A.manual.disabled,false);assert.equal(A.auto.disabled,false);
  assert.throws(()=>controller.assertReady(),/完成相机对焦选择/);
  controller.update(status,true);assert.equal(A.auto.checked,false,'轮询保留未完成的选择');
  A.manual.checked=true;event(A.manual,'change');
  assert.equal(requests.length,1);assert.equal(requests[0].setting.mode,'manual');assert.equal(requests[0].setting.value,50);
  assert.equal(B.slider.disabled,true,'设置期间所有设备控件暂时锁定');
  finish();await settle();
  assert.equal(A.auto.disabled,true);assert.equal(A.slider.disabled,false);
  A.slider.value='72';event(A.slider,'input');
  assert.match(A.value.textContent,/72/);assert.equal(requests.length,1,'拖动期间不反复发送');
  controller.update(status,true);assert.equal(Number(A.slider.value),72,'轮询不能覆盖滑动值');
  event(A.slider,'change');assert.equal(requests.length,2);assert.equal(requests[1].setting.value,72);
  fail();await settle();assert.match(A.note.textContent,/USB 写入失败/);assert.equal(A.retry.hidden,false);
  event(A.retry,'click');finish();await settle();assert.equal(A.retry.hidden,true);
  controller.assertReady();
  controller.update({...status,connected:true},true);
  assert.equal(A.manual.disabled,true);assert.equal(A.auto.disabled,true);assert.equal(B.slider.disabled,true);
  controller.update(status,true);controller.lock();assert.equal(B.slider.disabled,true,'网络中断也锁定');
  console.log('对焦 UI：互斥勾选、实际范围、滑块释放写入、轮询保留草稿、失败重试、会话和断线锁定通过。');
})().catch(error=>{console.error(error);process.exitCode=1;});
