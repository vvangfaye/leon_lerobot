// 控制实际页面脚本与播放器；模拟浏览器音频和 HTTP，绝不访问设备。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, 'nero_platform');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'static/voice/prompts.json')));
const voiceSource = fs.readFileSync(path.join(root, 'static/voice.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const source = html
  .match(/<script>([\s\S]*)<\/script>/)[1]
  .replace('/*VOICE_MANIFEST*/{}', JSON.stringify(manifest)).replace(/\ninit\(\);\s*$/, '');
const flush = async()=>{ for(let i=0;i<8;i++) await Promise.resolve(); };

function setup(sharedStorage = null, initialized = true) {
  const elements = new Map(), audio = [], requests = [], timers = new Map(), events = {};
  let rejected = false, timerId = 0;
  const element = id=>{
    if(!elements.has(id)) elements.set(id, {
      value:'', checked:true, disabled:false, hidden:false, textContent:'', open:false,
      focus(){}, showModal(){this.open=true;}, close(){this.open=false;}, removeAttribute(name){delete this[name];},
      appendChild(){}, append(){}, dataset:{}, style:{}, addEventListener(){}, add(){},
    });
    return elements.get(id);
  };
  for(const [id,value] of Object.entries({leader_label:'主左',follower_label:'从左',record_camera:'right',fps:30,episode_sec:10,repo_id:'nero/test',task:'test'})) element(id).value=value;
  const config = {leader_label:'主左',follower_label:'从左',record_camera:'right',fps:30,episode_sec:10,repo_id:'nero/restored',task:'已保存的任务',arms:{can0:'主左',can1:'从左',can2:'未分配',can3:'未分配'}};
  const status = {
    arms:Object.fromEntries(['can0','can1','can2','can3'].map((ch,i)=>[ch,{label:['主左','从左','未分配','未分配'][i],status:'已连接',detected:i<2}])),
    arm_count:2,connected:true,recording:false,episode_count:0,leader_label:'主左',follower_label:'从左',
    homing:false,home_status:'',home_error:null,disabling:false,device_busy:[],active_cameras:{right:'/fake'},
    record:{phase:'idle',message:'',frames:0,elapsed_seconds:0,limit_seconds:10,frame_errors:0,last_episode:null,writer_error:null},
  };
  class FakeAudio {
    constructor(url){this.url=url;this.paused=false;audio.push(this);}
    play(){return rejected ? Promise.reject(new Error('NotAllowedError')) : Promise.resolve();}
    pause(){this.paused=true;}
    end(){this.onended?.();}
  }
  const context = vm.createContext({
    AbortController,
    Audio:FakeAudio, setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id),
    localStorage:sharedStorage || {getItem:()=>null,setItem(){}},
    addEventListener:(event,fn)=>{events[event]=fn;},
    document:{getElementById:element,createElement:()=>element(Symbol()),querySelectorAll:()=>[],addEventListener:(event,fn)=>{events[event]=fn;}},
    fetch:async(url,options={})=>{
      if(options.method==='POST') requests.push({url,body:options.body ? JSON.parse(options.body) : null});
      let result={ok:true};
      if(url==='/api/status') result=status;
      else if(url==='/api/config') result=config;
      else if(url==='/api/cameras') result={cameras:{},camera_map:{}};
      else if(url==='/api/pair/home') status.homing=true;
      else if(url==='/api/record/start') {status.recording=true;status.record.phase='preparing';}
      else if(url==='/api/record/stop') {status.record.phase='stopping';status.record.stop_requested=true;}
      else if(url==='/api/record/cancel') {status.record.phase='cancelling';status.record.cancel_requested=true;status.record.stop_requested=true;}
      else if(url==='/api/record/delete-last/preview') result={ok:true,target:{repo_id:'nero/test',episode_index:2,frames:300,revision:'confirmed-version'}};
      else if(url==='/api/record/delete-last') {status.disk_episode_count=2;status.episode_count=2;status.record.last_episode=null;}
      else if(url==='/api/collection/end') {status.connected=false;status.recording=false;status.end_status='采集已结束，未执行失能';}
      else if(url==='/api/connect') status.connected=true;
      return {ok:true,json:async()=>result};
    },
  });
  vm.runInContext(voiceSource, context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'nero_platform/static/camera_focus.js'),'utf8'), context);
  vm.runInContext(source, context);
  if(initialized) vm.runInContext('configLoaded = camerasLoaded = true', context);
  return {context,status,config,element,audio,requests,timers,events,reject:()=>{rejected=true;},voice:vm.runInContext('voice',context)};
}

(async()=>{
  // 遥操作模式与频率可保存；旧配置沿用 JS/100 Hz，退休的平滑入口不再提交。
  {
    assert.doesNotMatch(html,/试验 [12]|id="follow_smoothing"/);
    const select = html.match(/<select id="control_frequency"[^>]*>([\s\S]*?)<\/select>/)[1];
    assert.deepEqual([...select.matchAll(/value="([^"]+)"/g)].map(m=>m[1]),['capture','50','100','150','200']);
    const h=setup();
    await h.context.loadConfig();
    assert.equal(h.element('follow_motion_mode').value,'js');
    assert.equal(h.element('control_frequency').value,'100');
    assert.equal('follow_smoothing' in h.context.getConfig(),false);
    for(const mode of ['js','j']) {
      h.element('follow_motion_mode').value=mode;
      for(const frequency of ['capture','50','100','150','200']) {
        h.element('control_frequency').value=frequency;
        assert.equal(h.context.getConfig().follow_motion_mode,mode);
        assert.equal(h.context.getConfig().control_frequency,frequency==='capture' ? frequency : Number(frequency));
      }
    }
    h.config.control_frequency='capture';
    h.config.fps=15;
    await h.context.loadConfig();
    assert.equal(h.element('control_frequency').value,'capture');
    assert.match(h.element('teleop-summary').textContent,/跟随采集（当前 15 Hz）/);
    h.element('fps').value=30;
    h.context.updateTeleopSummary();
    assert.match(h.element('teleop-summary').textContent,/跟随采集（当前 30 Hz）/);
    h.status.record.follow_motion_mode='j';
    h.status.record.control_frequency=150;
    h.status.record.control_timing={target_hz:150,effective_hz:149.9,max_interval_ms:7,skipped_slots:0,warnings:[]};
    await h.context.refresh();
    assert.match(h.element('rec-motion-mode').textContent,/底层平滑（高延迟 move_j）/);
    assert.match(h.element('rec-control').textContent,/目标 150 Hz · 实测 149.9 Hz/);
    h.status.record.follow_motion_mode='js';
    h.status.record.control_frequency='capture';
    h.status.record.control_timing={target_hz:30,effective_hz:30,max_interval_ms:33.3,skipped_slots:0,warnings:[]};
    await h.context.refresh();
    assert.match(h.element('rec-motion-mode').textContent,/分离控制（低延迟 move_js）/);
    assert.match(h.element('rec-control').textContent,/跟随采集.*目标 30 Hz/);
  }
  // 双对回位弹窗与请求必须包含四条臂，确认前不下发，目标按角色而非 CAN 顺序选择。
  {
    const h=setup();h.status.pair_mode='dual';h.status.arm_count=4;
    const labels={can0:'从右',can1:'主左',can2:'从左',can3:'主右'};
    for(const [ch,label] of Object.entries(labels)) Object.assign(h.status.arms[ch],{label,detected:true});
    await h.context.refresh();
    assert.equal(h.element('pair_mode').value,'dual');
    const pending=h.context.returnHome();
    assert.match(h.element('safety-text').textContent,/共 4 条臂/);
    for(const [ch,label] of Object.entries(labels)) assert.ok(h.element('safety-text').textContent.includes(`${label} (${ch})`));
    assert.equal(h.requests.length,0);
    h.audio.at(-1).end();await flush();h.context.finishSafety(true);await pending;
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save','/api/pair/home']);
    assert.equal(h.requests[0].body.pair_mode,'dual');
    assert.deepEqual(h.requests[1].body.targets,labels);
  }
  // 结束采集后仍按四条臂分别确认失能；两步独立，不会只处理旧单对选择。
  {
    const h=setup();h.status.pair_mode='dual';h.status.arm_count=4;
    h.status.arms.can2={label:'主右',detected:true,status:'已连接'};
    h.status.arms.can3={label:'从右',detected:true,status:'已连接'};
    await h.context.refresh();await h.context.endSession();
    const targets={can0:'主左',can1:'从左',can2:'主右',can3:'从右'};
    assert.deepEqual(h.requests[0].body.targets,targets);
    assert.equal(h.requests.some(r=>r.url==='/api/pair/disable'),false);
    const pending=h.context.disconnect();await flush();
    assert.match(h.element('safety-text').textContent,/主右 \(can2\).*从右 \(can3\)/);
    h.audio.at(-1).end();await flush();h.context.finishSafety(true);await pending;
    assert.deepEqual(h.requests.at(-1).body.targets,targets);
    assert.equal(h.requests.at(-1).url,'/api/pair/disable');
  }
  // 初始化分步重试：配置成功后不重新覆盖输入，失败期间不允许用默认值下发操作。
  {
    const h=setup(null,false); h.voice.enabled=false;
    const fetch=h.context.fetch;
    let configReads=0,cameraReads=0,statusReads=0;
    h.context.fetch=async(url,options)=>{
      if(url==='/api/config' && ++configReads===1) return {ok:false,status:503};
      if(url==='/api/cameras' && ++cameraReads===1) throw new Error('相机列表网络故障');
      if(url==='/api/status') statusReads++;
      return fetch(url,options);
    };
    const retry=async()=>{
      const [id,run]=[...h.timers].find(([,fn])=>fn===h.context.poll);
      h.timers.delete(id); await run();
    };
    await h.context.init();
    assert.equal(statusReads,1,'初始化查询失败后仍读取后台状态');
    assert.match(h.element('connection-status').textContent,/初始化未完成.*503/);
    assert.equal(h.element('btn-start').disabled,true);
    await h.context.save();
    assert.equal(h.requests.length,0,'配置未读到时不保存页面默认值');
    await retry();
    assert.equal(configReads,2);
    assert.match(h.element('connection-status').textContent,/相机列表网络故障/);
    assert.equal(h.element('repo_id').value,'nero/restored');
    h.element('task').value='用户正在编辑';h.context.markConfigDirty();
    await retry();
    assert.equal(configReads,2,'恢复相机列表不重新载入已读到的配置');
    assert.equal(h.element('task').value,'用户正在编辑');
    assert.equal(h.element('connection-status').hidden,true);
    assert.equal(h.element('btn-start').disabled,false);
    assert.equal(h.requests.length,0,'自动恢复始终只有查询');
  }
  // 挂起的查询超时退出；周期查询完成前不安排下一轮，也不覆盖数据写盘错误。
  {
    const h=setup();h.voice.enabled=false;await h.context.refresh();
    h.status.record.phase='error';h.status.record.writer_error='磁盘写入失败';
    await h.context.refresh();
    const fetch=h.context.fetch;
    let reads=0;
    h.context.fetch=(url,{signal})=>{
      assert.equal(url,'/api/status');reads++;
      return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new Error('aborted'))));
    };
    const polling=h.context.poll();
    assert.equal(reads,1);
    assert.equal([...h.timers.values()].includes(h.context.poll),false,'慢查询期间没有周期请求队列');
    for(const run of [...h.timers.values()]) run();
    await polling;
    assert.match(h.element('connection-status').textContent,/后台运行状态暂未知.*超时/);
    assert.match(h.element('rec-error').textContent,/磁盘写入失败/);
    assert.equal(h.element('btn-start').disabled,true);
    const [id,retry]=[...h.timers].find(([,fn])=>fn===h.context.poll);
    h.timers.delete(id);h.context.fetch=fetch;await retry();
    assert.equal(h.element('connection-status').hidden,true);
    assert.equal(h.element('btn-start').disabled,true,'网络恢复仍受真实写入错误门禁限制');
    assert.equal(h.requests.length,0);
  }
  // 会话结束后的相机刷新失败只锁定页面准备，不伪造设备连接中断。
  {
    const h=setup();await h.context.refresh();
    h.status.connected=false;
    const fetch=h.context.fetch;
    h.context.fetch=(url,options)=>url==='/api/cameras' ? Promise.reject(new Error('相机列表失败')) : fetch(url,options);
    await h.context.refresh();
    assert.match(h.element('connection-status').textContent,/相机分配刷新失败/);
    assert.doesNotMatch(h.element('connection-status').textContent,/后台运行状态暂未知/);
    assert.equal(h.element('btn-session-start').disabled,true);
    h.context.fetch=fetch;await h.context.poll();
    assert.equal(h.element('btn-session-start').disabled,false);
  }
  // 关闭/刷新页面只提示：会话、使能、未知状态应提醒，全部失能后不打扰。
  {
    const h=setup();await h.context.refresh();
    const warns=()=>{
      let prevented=false;const event={preventDefault(){prevented=true;}};
      h.events.beforeunload(event);
      if(prevented) assert.equal(event.returnValue,'');
      return prevented;
    };
    assert.equal(warns(),true,'采集会话仍开启');
    h.status.connected=false;await h.context.refresh();
    assert.equal(warns(),true,'连接过的设备尚无完整使能反馈');
    for(const arm of Object.values(h.status.arms)) arm.hardware={communication:{state:arm.detected?'live':'disconnected'},enabled:{state:'disabled'},gripper:{present:false},health:{state:'normal'},mode:{}};
    await h.context.refresh();
    assert.equal(warns(),false,'设备均确认失能且没有会话');
    h.status.arms.can0.hardware.gripper={present:true,enabled:true};await h.context.refresh();
    assert.equal(warns(),true,'单独夹爪使能也提醒');
    h.status.arms.can0.hardware.gripper.enabled=false;await h.context.refresh();
    assert.equal(warns(),false);
    h.context.fetch=async()=>{throw new Error('连接中断');};await h.context.refresh();
    assert.equal(warns(),true,'网络中断后不能把上一次失能反馈视为当前确认');
    assert.equal(h.requests.length,0,'离开事件绝不发送设备指令');
  }
  // 停止逻辑读取业务状态，即使语音模块停用、未观察状态也可取消对齐。
  {
    const h=setup();h.status.recording=true;h.status.record.phase='aligning';
    await h.context.refresh();h.voice.previous=null;
    await h.context.stop();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/record/stop']);
    assert.ok(h.audio.some(a=>a.url.includes('/cancel_pending.mp3')));
  }
  // 已保存需复核仍播报保存完成，不误报成操作失败，也不重复播报。
  {
    const h=setup();await h.context.refresh();
    h.status.record.phase='saved_review';
    h.status.record.last_episode={repo_id:'nero/test',number:1,frames:1,elapsed_seconds:.1,needs_review:true,termination_error:'反馈中断'};
    await h.context.refresh();await h.context.refresh();
    assert.equal(h.audio.filter(a=>a.url.includes('/saved.mp3')).length,1);
    assert.equal(h.audio.filter(a=>a.url.includes('/failed.mp3')).length,0);
  }
  // 准备提醒无需听完：确认立刻停音，在保存配置请求等待中也不会继续播报。
  {
    const h=setup();await h.context.refresh();await h.context.start();
    const reminder=h.audio.at(-1);
    assert.match(reminder.url,/prepare/);
    assert.equal(reminder.paused,false);
    assert.equal(h.requests.length,0);
    let releaseSave;
    const savePending=new Promise(resolve=>{releaseSave=resolve;});
    const fetch=h.context.fetch;
    h.context.fetch=async(url,options)=>{
      const response=await fetch(url,options);
      if(url==='/api/save') await savePending;
      return response;
    };
    const starting=h.context.confirmStart();
    assert.equal(reminder.paused,true,'确认时立即停止准备语音，不等待接口返回');
    assert.equal(reminder.onended,null,'不需要等待播报结束事件');
    assert.equal(h.element('start-reminder').hidden,true);
    await h.context.confirmStart();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save']);
    releaseSave();await starting;
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save','/api/record/start']);
    assert.match(h.audio.at(-1).url,/preparing/,'后续阶段仍正常播报');
  }
  // 等待提示期间、取消后、非确认快捷键均不发送回位命令（q 是明确确认）。
  {
    const h=setup(); await h.context.refresh();
    const pending=h.context.returnHome();
    assert.equal(h.element('safety-dialog').open,true);
    assert.match(h.audio.at(-1).url,/home_warning/);
    assert.equal(h.element('safety-confirm').disabled,false,'现有行为允许跳过语音，取消路径不点击确认');
    h.events.keydown({key:'s',target:{closest:()=>false},preventDefault(){}});
    await h.context.returnHome();
    assert.equal(h.requests.length,0);
    h.context.finishSafety(false); await pending;
    assert.equal(h.requests.length,0);
    assert.equal(h.element('safety-dialog').open,false);
  }
  // 播完提醒仍不会自动执行；手动确认后只执行一次。
  {
    const h=setup(); await h.context.refresh();
    const pending=h.context.returnHome();
    h.audio.at(-1).end(); await flush();
    assert.equal(h.element('safety-confirm').disabled,false);
    assert.equal(h.requests.length,0);
    h.events.keydown({key:'q',target:{closest:()=>false},preventDefault(){}});
    h.context.finishSafety(true); await pending;
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save','/api/pair/home']);
    assert.match(h.audio.at(-1).url,/home_started/);
    h.status.homing=false; h.status.home_status='均已回默认位置'; await h.context.refresh();
    assert.match(h.audio.at(-1).url,/home_done/);
  }
  // 结束采集只调用收尾接口；独立失能必须先完整提醒、手动确认。
  {
    const h=setup(); await h.context.refresh();
    assert.equal(h.element('btn-disconnect').disabled,true,'会话开启时不能直接失能');
    await h.context.disconnect();assert.equal(h.requests.length,0);
    await h.context.endSession();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/collection/end']);
    assert.match(h.audio.at(-1).url,/session_ended/);
    h.requests.length=0;
    const pending=h.context.disconnect(); await flush();
    assert.match(h.element('safety-text').textContent,/托住主从机械臂.*缓冲垫子/);
    assert.equal(h.requests.length,0);
    h.audio.at(-1).end(); await flush();
    h.context.finishSafety(true); await pending;
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save','/api/pair/disable']);
    assert.match(h.audio.at(-1).url,/pair_disable_done/);
  }
  // 开始采集只开启会话，不偷偷录一集；结束侧回位也需同一确认。
  {
    const h=setup(); h.status.connected=false; await h.context.refresh();
    assert.equal(h.element('btn-start').disabled,true);
    await h.context.openSession();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/save','/api/connect']);
    assert.equal(h.element('btn-start').disabled,false);
    assert.match(h.audio.at(-1).url,/session_ready/);
    h.requests.length=0;
    const home=h.context.returnHome('btn-home-end');await flush();
    assert.equal(h.element('safety-dialog').open,true);
    h.context.finishSafety(false);await home;
    assert.equal(h.requests.length,0);
  }
  // 播放被拦截、资源丢失或静音都必须显示文字并等待明确点击，不能自动确认。
  for(const mode of ['blocked','missing','muted','timeout']) {
    const h=setup(); await h.context.refresh();
    if(mode==='blocked') h.reject();
    if(mode==='muted') h.voice.configure({enabled:false});
    const pending=h.context.returnHome();
    if(mode==='missing') h.audio.at(-1).onerror();
    if(mode==='timeout') for(const callback of [...h.timers.values()]) callback();
    await flush();
    assert.equal(h.requests.length,0,mode);
    assert.equal(h.element('safety-confirm').disabled,false,mode);
    assert.match(h.element('safety-audio').textContent,/可立即确认/,mode);
    h.context.finishSafety(false); await pending;
    assert.equal(h.requests.length,0,mode);
  }
  // 初次加载不播历史结果；轮询不重复播报；自动结束也播保存结果。
  {
    const h=setup(); h.status.record.phase='saved';
    h.status.record.last_episode={repo_id:'nero/test',number:1,frames:300,elapsed_seconds:10};
    await h.context.refresh(); assert.equal(h.audio.length,0);
    for(const phase of ['preparing','aligning','recording','saving']) {
      h.status.record.phase=phase; await h.context.refresh();
      assert.match(h.audio.at(-1).url,new RegExp(phase));
      const count=h.audio.length; await h.context.refresh(); assert.equal(h.audio.length,count);
    }
    h.status.record.phase='saved';h.status.record.last_episode.number=2;
    await h.context.refresh();assert.match(h.audio.at(-1).url,/saved/);
    h.status.record.phase='error';h.status.last_error='写盘失败';
    await h.context.refresh();assert.match(h.audio.at(-1).url,/failed/);
    assert(h.audio.slice(0,-1).every(a=>a.paused),'新状态会取消过时播报');
  }
  // 停止采集不等待语音结束，也不为播报增加设备操作。
  {
    const h=setup();h.status.recording=true;h.status.record.phase='recording';await h.context.refresh();
    await h.context.stop();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/record/stop']);
    assert.match(h.audio.at(-1).url,/saving/);
    assert.equal(h.audio.filter(a=>a.url.includes('/saving.mp3')).length,1,'按钮与状态变化只能触发一次保存播报');
    h.audio.at(-1).end();
    h.status.record.phase='saving';await h.context.refresh();
    assert.equal(h.audio.filter(a=>a.url.includes('/saving.mp3')).length,1,'停止中进入保存中不重新播报');
    h.status.record.phase='recording';await h.context.refresh();
    h.status.record.phase='saving';await h.context.refresh();
    assert.equal(h.audio.filter(a=>a.url.includes('/saving.mp3')).length,2,'下一集的保存仍正常播报');
  }
  // 操作后刷新与定时刷新交错：旧状态晚到不得重复播报回位或误报完成。
  {
    const h=setup();await h.context.refresh();
    const pending=[];
    h.context.fetch=async url=>{
      assert.equal(url,'/api/status');
      const snapshot=JSON.parse(JSON.stringify(h.status));
      return new Promise((resolve,reject)=>pending.push({resolve:()=>resolve({ok:true,json:async()=>snapshot}),reject}));
    };
    const old=h.context.refresh();
    h.status.homing=true;
    const recent=h.context.refresh();
    pending[1].resolve();await recent;
    assert.equal(h.audio.filter(a=>a.url.includes('/home_started.mp3')).length,1);
    pending[0].resolve();await old;
    assert.equal(h.audio.filter(a=>a.url.includes('/home_done.mp3')).length,0,'旧的未回位状态不能被当作已完成');
    const next=h.context.refresh();pending[2].resolve();await next;
    assert.equal(h.audio.filter(a=>a.url.includes('/home_started.mp3')).length,1,'回位期间不会重新播放开头');
    const staleError=h.context.refresh();
    h.status.homing=false;h.status.home_status='主臂和从臂均已回默认位置，保持使能';
    const done=h.context.refresh();pending[4].resolve();await done;
    pending[3].reject(new Error('过期请求超时'));await staleError;
    assert.equal(h.audio.filter(a=>a.url.includes('/home_done.mp3')).length,1);
    assert.doesNotMatch(h.element('rec-error').textContent,/过期请求/);
  }
  // 正在播放的同一普通提示不从头重播；下次操作仍能播放。
  {
    const h=setup();
    const first=h.voice.say('home_started');
    const duplicate=h.voice.say('home_started');
    assert.equal(h.audio.length,1);
    assert.equal(h.audio[0].paused,false);
    h.audio[0].end();await Promise.all([first,duplicate]);
    void h.voice.say('home_started');assert.equal(h.audio.length,2);
  }
  // 隐藏的旧标签页不跟随播报；切回页面不补播过时状态。
  {
    const h=setup();await h.context.refresh();
    const pending=h.context.returnHome();
    assert.equal(h.audio.length,1);
    h.context.document.hidden=true;h.events.visibilitychange();await flush();
    assert.equal(h.audio[0].paused,true,'切到后台立即停音');
    assert.equal(h.requests.length,0,'暂停安全语音不会执行回位');
    h.context.finishSafety(false);await pending;
    h.status.homing=true;await h.context.refresh();
    assert.equal(h.audio.length,1,'后台轮询只更新状态，不播放');
    h.context.document.hidden=false;h.events.visibilitychange();await h.context.refresh();
    assert.equal(h.audio.length,1,'切回来不补播旧状态');
    h.status.homing=false;await h.context.refresh();
    assert.match(h.audio.at(-1).url,/home_done/,'后续真实状态仍正常播报');
  }
  // 安全提醒不被后台播报打断；重新播放也不取消用户可立即确认的能力。
  {
    const h=setup();await h.context.refresh();const pending=h.context.returnHome();
    const warning=h.audio.at(-1);void h.voice.say('saved');assert.equal(h.audio.at(-1),warning);
    void h.context.replaySafety();await flush();assert.equal(h.element('safety-confirm').disabled,false);
    h.audio.at(-1).end();await flush();assert.equal(h.element('safety-confirm').disabled,false);
    h.context.finishSafety(false);await pending;
    assert.equal(h.requests.length,0);
  }
  // 单臂回位后失能与重试失能同样受弹窗门禁；只读复核不弹危险操作确认。
  for(const action of ['single','retry','recheck']) {
    const h=setup(); h.status.connected=false;
    h.status.disable_error= action==='single' ? null : '夹爪未确认';
    h.status.close_result={targets:{can0:'主左',can1:'从左'},data:{state:'done',message:''},arms:{can1:{label:'从左',state:'error',message:'夹爪未确认'}},preview:{},warnings:[]};
    await h.context.refresh();
    const pending=action==='single' ? h.context.armDisable('can0') : h.context.retryClose(action==='retry'?'arm':'status','can1');
    await flush();
    if(action==='recheck') {await pending;assert.equal(h.element('safety-dialog').open,false);}
    else {assert.equal(h.element('safety-dialog').open,true);h.context.finishSafety(false);await pending;assert.equal(h.requests.length,0);}
  }
  // 两个窗口都可见时，仅最近操作的页面播报；切换窗口立刻停止原页面声音。
  {
    const values=new Map();
    const storage={getItem:key=>values.get(key) ?? null,setItem:(key,value)=>values.set(key,value)};
    const first=setup(storage);await first.context.refresh();
    first.status.homing=true;await first.context.refresh();
    const oldAudio=first.audio.at(-1);
    assert.match(oldAudio.url,/home_started/);
    const second=setup(storage);second.status.homing=true;await second.context.refresh();
    const ownerChanged={key:'nero.voice.playbackOwner'};
    first.events.storage(ownerChanged);
    assert.equal(oldAudio.paused,true,'第二个可见窗口接管时停止第一个窗口声音');
    for(const h of [first,second]) {
      h.status.homing=false;h.status.home_status='主臂和从臂均已回默认位置，保持使能';
      await h.context.refresh();
    }
    const doneCount=()=>[...first.audio,...second.audio].filter(a=>a.url.includes('/home_done.mp3')).length;
    assert.equal(doneCount(),1,'两个页面收到同一回位完成状态，只播报一次');
    const completed=second.audio.at(-1);
    first.events.focus();second.events.storage(ownerChanged);
    assert.equal(completed.paused,true,'切回第一个窗口时取消另一个窗口尚未结束的播报');
    await first.context.refresh();await second.context.refresh();
    assert.equal(doneCount(),1,'切换窗口不补播已经观察到的历史完成结果');
    for(const h of [first,second]) {h.status.homing=true;await h.context.refresh();}
    for(const h of [first,second]) {h.status.homing=false;await h.context.refresh();}
    assert.equal(doneCount(),2,'下一次真实回位仍只播报一次');
    second.events.pointerdown();first.events.storage(ownerChanged);
    assert.equal(first.audio.at(-1).paused,true,'点击已有页面也能接管语音');
    void second.voice.say('home_warning',{critical:true});
    assert.match(second.audio.at(-1).url,/home_warning/,'当前操作页面的安全警告正常播放');
    second.events.pagehide();assert.equal(second.audio.at(-1).paused,true);
  }
  // 禁用存储时退回焦点判断，其他可见窗口仍不能同时发声。
  {
    const h=setup({getItem(){throw new Error('storage blocked');},setItem(){throw new Error('storage blocked');}});
    h.context.document.hasFocus=()=>false;
    assert.equal(await h.voice.say('home_done'),'suppressed');assert.equal(h.audio.length,0);
    h.context.document.hasFocus=()=>true;void h.voice.say('home_done');assert.equal(h.audio.length,1);
  }
  // 取消有独立状态：不播保存或无有效帧，清理完成后按钮恢复文案。
  {
    const h=setup();h.status.recording=true;h.status.record.phase='recording';
    await h.context.refresh();
    assert.equal(h.element('btn-cancel-record').disabled,false);
    await h.context.cancelRecord();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/record/cancel']);
    assert.match(h.audio.at(-1).url,/cancel_recording/);
    assert.equal(h.element('btn-stop').disabled,true);
    h.status.recording=false;h.status.record.phase='cancelled';h.status.record.cancel_requested=false;
    await h.context.refresh();
    assert.equal(h.element('btn-cancel-record').textContent,'中途取消本集');
    assert.equal(h.element('phase-badge').textContent,'已取消 · 未保存');
    assert.equal(h.audio.some(a=>/\/(saving|saved|empty_episode)\.mp3/.test(a.url)),false);
  }
  // 预检在确认之前；提交固定目标，后续状态刷新不能悄悄改变删除集号。
  {
    const h=setup();h.status.connected=false;h.status.disk_episode_count=3;
    await h.context.refresh();
    assert.equal(h.element('btn-delete-last').disabled,false);
    const pending=h.context.deleteLastEpisode();await flush();
    await h.context.deleteLastEpisode();
    assert.deepEqual(h.requests.map(r=>r.url),['/api/record/delete-last/preview']);
    assert.match(h.element('safety-text').textContent,/nero\/test.*\n.*第 3 集（300 帧）/);
    h.status.disk_episode_count=2;await h.context.refresh();
    assert.equal(h.element('btn-session-start').disabled,true);
    h.context.finishSafety(true);await pending;
    assert.deepEqual(h.requests.map(r=>r.url),['/api/record/delete-last/preview','/api/record/delete-last']);
    assert.deepEqual(h.requests[1].body,{repo_id:'nero/test',episode_index:2,frames:300,revision:'confirmed-version'});
    assert.match(h.audio.at(-1).url,/deleted/);
  }
  console.log('语音播放、多窗口独占播报、状态播报、危险操作确认、取消/快捷键/重复点击/播放失败检查通过（模拟浏览器，无设备命令）。');
})().catch(error=>{console.error(error);process.exitCode=1;});
