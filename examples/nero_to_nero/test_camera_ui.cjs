// 执行实际页面的设备列表与检测按钮逻辑；不访问网络、设备。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname,'nero_platform/index.html'),'utf8');
const source = html.match(/<script>([\s\S]*)<\/script>/)[1].replace(/\ninit\(\);\s*$/,'');
class Element {
  constructor(){this.children=[];this.dataset={};this.style={};this.value='';this.disabled=false;this.events={};this.textContent='';}
  set innerHTML(value){this.markup=value;this.children=[];}
  get innerHTML(){return this.markup || '';}
  append(...children){this.children.push(...children);}
  appendChild(child){this.children.push(child);}
  add(option){this.children.push(option);}
  addEventListener(name,handler){this.events[name]=handler;}
  setAttribute(name,value){this[name]=value;}
  removeAttribute(name){delete this[name];}
  querySelector(selector){return this.children.find(child=>selector.startsWith('.') ? child.className?.split(' ').includes(selector.slice(1)) : child.tagName===selector);}
}
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
for(const [id,value] of Object.entries({record_camera:'all',fps:30,episode_sec:10}))element(id).value=value;
const requests=[];
let release,reply;
const context=vm.createContext({
  AbortController,setTimeout,clearTimeout,addEventListener(){},
  NeroVoice:class{say(){return Promise.resolve();}},
  Option:class{constructor(text,value){this.text=text;this.value=value;}},
  document:{getElementById:element,createElement:tag=>Object.assign(new Element(),{tagName:tag}),querySelectorAll:selector=>selector==='.cam[data-camera-id]' ? element('cams').children : [],addEventListener(){}},
  fetch:async(url,options)=>{requests.push([url,options]);await new Promise(resolve=>{release=resolve;});return {ok:true,json:async()=>reply};},
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'nero_platform/static/camera_focus.js'),'utf8'),context);
vm.runInContext(source,context);
vm.runInContext("configLoaded=camerasLoaded=statusAvailable=true; MAP={'usb:A':'left','usb:B':'right','old-assigned':'head','old-unused':'unused'}",context);
context.refresh=async()=>{};
const inventory={
  'usb:A':{type:'rgb',backend:'opencv',path:'/fake/A'},
  'usb:B':{type:'rgb',backend:'opencv',path:'/fake/B'},
  'realsense:R':{type:'rgbd',backend:'realsense',name:'RealSense D455',serial:'R',recording_supported:true,preview_supported:true},
};
const titles=()=>element('cams').children.map(card=>card.children[0].textContent);
(async()=>{
  context.renderCameras({cameras:inventory,detection:{realsense:{label:'RealSense',message:'检测到 1 台'}}});
  assert.deepEqual(titles(),['相机 1','相机 2','相机 3']);
  assert.equal(element('cams').children.length,3,'未检测到的历史配置不占编号');
  assert.equal(element('missing-cameras').children.length,1);
  assert.match(element('missing-cameras').children[0].children[0].textContent,/old-assigned/);
  assert.match(element('camera-detection').textContent,/3 台.*RealSense.*1 台/);
  const card=element('cams').children.find(card=>card.dataset.cameraId==='usb:A');
  const select=card.children.at(-1);
  assert.deepEqual(select.children.map(option=>option.value),['unused','left','right','head']);
  select.value='head';select.events.change();
  assert.deepEqual(titles(),['相机 1','相机 2','相机 3'],'修改用途不改标题或重排');
  assert.equal(vm.runInContext('MAP["usb:A"]',context),'head');
  reply={ok:true,cameras:{...inventory,'future:U':{type:'rgbd',backend:'future',serial:'R',recording_supported:false}},
    camera_map:{'usb:A':'left'},detection:{future:{label:'Future',message:'检测到 1 台'}}};
  const pending=context.detectCameras('depth');
  assert.equal(element('btn-detect-depth').disabled,true);
  assert.equal(element('btn-detect-rgb').disabled,true);
  await context.detectCameras('rgb');
  assert.equal(requests.length,1,'检测中不能重复发送');
  release();await pending;
  assert.equal(requests[0][0],'/api/cameras/detect/depth');
  assert.equal(vm.runInContext('MAP["usb:A"]',context),'head','检测不能覆盖未保存用途');
  assert.deepEqual(titles(),['相机 1','相机 2','相机 3','相机 4']);
  const unsupported=element('cams').children.find(card=>card.dataset.cameraId==='future:U');
  assert.match(unsupported.children.find(child=>child.className==='warn').textContent,/尚未接通 RGBD 录制/);
  assert.doesNotMatch(html,/id="btn-rescan-cams"|async function rescanCameras/);
  assert.match(html,/id="btn-detect-depth"/);assert.match(html,/id="btn-detect-rgb"/);
  const rgbd=element('cams').children.find(card=>card.dataset.cameraId==='realsense:R');
  assert.deepEqual(rgbd.querySelector('.rgbd-labels').children.map(child=>child.textContent),['RGB','深度']);
  assert.equal(rgbd.querySelector('img').className,'rgbd-pair');
  element('camera-settings').open=true;
  const status={connected:false,device_busy:[],record:{phase:'idle'},camera_previews:{}};
  requests.length=0;reply={ok:true,phase:'ready'};
  context.updatePreviews(status);context.updatePreviews(status);
  assert.equal(requests.length,1,'轮询和多次渲染不应重复打开');
  assert.equal(requests[0][0],'/api/camera/realsense%3AR/preview/start');
  release();await new Promise(resolve=>setImmediate(resolve));
  status.camera_previews={'realsense:R':{phase:'ready'}};
  context.updatePreviews(status);
  assert.match(rgbd.querySelector('img').src,/realsense%3AR\/rgbd\/frame/,'并排图走一次 RGBD 快照接口');
  assert.match(unsupported.querySelector('.mapping-empty').textContent,/仅支持检测/);
  element('camera-settings').open=false;context.updatePreviews(status);
  assert.equal(rgbd.querySelector('img').hidden,true,'折叠移除旧图，不发图像请求');
  element('camera-settings').open=true;status.camera_previews={'realsense:R':{phase:'error',message:'USB 占用'}};
  context.updatePreviews(status);
  assert.match(rgbd.querySelector('.mapping-empty').textContent,/USB 占用/);
  assert.equal(rgbd.querySelector('.preview-retry').hidden,false);
  assert.equal(requests.length,1,'打开失败等待手动重试，不循环抢占设备');
  status.camera_focus={'usb:A':{state:'verified',stage:'episode_start',message:'读回确认：自动对焦已关闭，手动对焦 128',values:{zoom_absolute:100,exposure_auto:1,exposure_absolute:2500}}};
  context.updatePreviews(status);
  const rgb=element('cams').children.find(card=>card.dataset.cameraId==='usb:A');
  assert.match(rgb.querySelector('.camera-focus').textContent,/本集开始前.*手动对焦 128.*变焦读数 100.*曝光读数 2500/);
  assert.equal(element('camera-focus-warning').hidden,true);
  status.camera_focus['usb:A']={state:'warning',stage:'episode_end',message:'对焦未确认：focus_auto 期望 0，读回 1',values:{focus_auto:1}};
  context.updatePreviews(status);
  assert.match(rgb.querySelector('.camera-focus').className,/warn/);
  assert.equal(element('camera-focus-warning').hidden,false);
  assert.match(element('camera-focus-warning').textContent,/focus_auto.*读回 1/);
  console.log('相机 UI：统一编号、分类检测、配置保留、RGBD 并排预览、打开互斥与失败重试通过。');
})().catch(error=>{console.error(error);process.exitCode=1;});
