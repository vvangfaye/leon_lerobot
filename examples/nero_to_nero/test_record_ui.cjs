// 无浏览器、无网络：执行页面自身的状态更新和点击逻辑，检查按钮互斥。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, 'nero_platform/index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*)<\/script>/)[1].replace(/\ninit\(\);\s*$/, '');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, { value: '', disabled: false, textContent: '', removeAttribute(name){delete this[name];} });
  return elements.get(id);
}
for (const [id, value] of Object.entries({leader_label:'主左', follower_label:'从左', record_camera:'right', fps:30, episode_sec:10, repo_id:'nero/test', task:'测试'})) {
  element(id).value = value;
}
const calls = [];
const retryBodies = [];
let pendingSave;
let status;
function setPhase(phase) {
  const busy = ['preparing', 'aligning', 'recording', 'stopping', 'saving'].includes(phase);
  status = {
    arms: Object.fromEntries(['can0','can1','can2','can3'].map((ch,i)=>[ch, {label:['主左','从左','未分配','未分配'][i], status:'已连接', detected:i<2, data:null}])),
    arm_count:2, connected:true, recording:busy, episode_count:0, leader_label:'主左', follower_label:'从左',
    homing:false, disabling:false, device_busy:[], active_cameras:{right:'/fake/right'},
    record:{phase, message:phase, frames:5, elapsed_seconds:0.5, limit_seconds:10, frame_errors:0, last_episode:null, writer_error:null, stop_requested:phase==='stopping'},
  };
}
const context = vm.createContext({
  AbortController, setTimeout, clearTimeout, addEventListener:()=>{},
  NeroVoice: class {
    constructor() { this.enabled = true; this.voice = 'xiaoxiao'; this.volume = .85; }
    say() { return Promise.resolve('muted'); }
    stop() {}
    observe() {}
    trackArm() {}
  },
  document: {getElementById:element, querySelectorAll:()=>[], addEventListener:()=>{}},
  alert: message=>{ throw new Error(message); },
  confirm: ()=>true,
  fetch: async (url, options) => {
    calls.push(url);
    if (url === '/api/status') return {ok:true,json:async()=>status};
    if (url === '/api/cameras') return {ok:true,json:async()=>({cameras:{}, camera_map:{}})};
    if (url === '/api/arm/status/can0') return {ok:true, json:async()=>({ok:true})};
    if (url === '/api/close/retry') {
      retryBodies.push(JSON.parse(options.body));
      return {ok:false, json:async()=>({ok:false, msg:'其他步骤尚未完成'})};
    }
    if (url === '/api/save') return pendingSave;
    if (url === '/api/record/start') {
      setPhase('preparing');
      return {ok:true, json:async()=>({ok:true})};
    }
    throw new Error(`Unexpected request: ${url} ${options?.method}`);
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'nero_platform/static/camera_focus.js'),'utf8'), context);
vm.runInContext(source, context);
vm.runInContext('configLoaded = camerasLoaded = true', context);
// 弹窗与真实语音控制器由 test_voice_ui.cjs 单独覆盖；本文件保持原状态门禁回归。
context.safetyConfirm = async()=>true;

(async () => {
  for (const phase of ['preparing','aligning','recording','stopping','saving']) {
    setPhase(phase);
    await context.refresh();
    assert.equal(element('btn-start').disabled, true, phase);
    assert.equal(element('btn-home').disabled, true, phase);
    assert.equal(element('btn-stop').disabled, ['stopping','saving'].includes(phase), phase);
    assert.equal(element('btn-cancel-prepare').hidden, !['preparing','aligning'].includes(phase), phase);
    if (['preparing','aligning'].includes(phase)) assert.equal(element('btn-disconnect').disabled, true);
  }
  // 配置选择、采集中来源、画面过期和集间取景相互独立。
  setPhase('idle'); status.connected=false;
  await context.refresh();
  vm.runInContext('CAMERAS={"fake-left":"/fake/left","fake-right":"/fake/right"}; MAP={"fake-left":"left","fake-right":"right"}', context);
  await context.refresh();
  assert.equal(element('live-left').hidden,true);
  assert.equal(element('live-right').hidden,false);
  assert.match(element('live-image-right').src,/\/api\/camera\/fake-right\/frame/);
  const oldLoaded=element('live-image-right').onload;
  element('record_camera').value='left';
  await context.refresh();
  oldLoaded();
  assert.equal(element('live-image-right').hidden,true,'换来源后旧请求不能恢复画面');
  assert.match(element('live-image-left').src,/\/api\/camera\/fake-left\/frame/);
  element('live-image-left').onerror();
  assert.equal(element('live-image-left').hidden,true);
  assert.match(element('live-empty-left').textContent,/暂无新画面/);
  setPhase('recording');
  await context.refresh();
  assert.equal(element('live-left').hidden,true,'会话实际 right 来源优先于页面暂存选择');
  assert.match(element('live-image-right').src,/\/api\/record\/camera\/right\/frame/);
  assert.match(element('preview-mode').textContent,/录制采样画面/);
  element('live-image-right').onload();
  assert.equal(element('live-image-right').hidden,false);
  setPhase('saved');
  await context.refresh();
  assert.equal(element('live-image-right').hidden,true,'集间切换为实时取景前先清除录制旧帧');
  assert.match(element('preview-mode').textContent,/实时取景/);
  element('live-image-right').onload();
  const fetch = context.fetch;
  context.fetch = async()=>{throw new Error('网络中断');};
  await context.refresh();
  assert.equal(element('live-image-right').hidden,true,'状态连接失败时清除旧画面');
  assert.match(element('preview-mode').textContent,/连接中断/);
  context.fetch = fetch;
  element('record_camera').value='right';
  setPhase('saving');
  await context.refresh();
  await context.stop();
  assert.equal(calls.includes('/api/record/stop'), false, '保存阶段不会再次发出停止指令');
  setPhase('saved');
  status.episode_count = 1;
  status.record.last_episode = {repo_id:'nero/test', number:1, frames:5, elapsed_seconds:0.5};
  await context.refresh();
  assert.equal(element('btn-start').disabled, false);
  assert.equal(element('phase-badge').textContent, '已保存');
  assert.match(element('rec-last').textContent, /第 1 集.*5 帧/);
  status.record.phase = 'saved_review';
  status.record.last_episode.needs_review = true;
  status.record.last_episode.termination_error = '从臂反馈停止更新';
  await context.refresh();
  assert.equal(element('phase-badge').textContent, '已保存 · 需复核');
  assert.equal(element('btn-start').disabled, false, '有效帧已保存，下一集重新检查设备反馈');
  assert.match(element('rec-last').textContent, /已?保存.*从臂反馈停止更新/);
  assert.equal(element('rec-error').textContent, '', '数据质量提示不冒充保存失败');
  status.record.timing = {target_fps:30, effective_fps:27, max_interval_ms:80, mean_processing_ms:12, max_processing_ms:70, skipped_slots:2, nominal_playback_seconds:9, warnings:['实测采样 FPS 低于目标的 95%']};
  status.record.last_episode.timing = status.record.timing;
  status.record.last_episode.timing_report = '/temporary/meta/nero_timing/episode_000000.json';
  await context.refresh();
  assert.match(element('rec-timing').textContent, /目标 30 FPS.*实测采样 27.0 FPS/);
  assert.match(element('timing-warning').textContent, /95%/);
  assert.match(element('rec-last').textContent, /播放 9.0 秒.*节拍报告/);
  status.record.timing_report_error = '本集数据已保存，但节拍报告保存失败';
  await context.refresh();
  assert.equal(element('btn-start').disabled, false, '诊断报告失败不把已保存的数据当作写入失败');
  assert.match(element('timing-warning').textContent, /报告保存失败/);
  setPhase('preparing');
  await context.refresh();
  assert.match(element('rec-timing').textContent, /至少两帧/);
  assert.equal(element('timing-warning').textContent, '', '新一集不会沿用旧节拍警告');
  setPhase('error');
  status.record.writer_error = '磁盘错误';
  await context.refresh();
  assert.equal(element('btn-start').disabled, true);
  assert.match(element('rec-error').textContent, /暂停续录/);
  assert.equal(element('phase-badge').textContent, '保存异常 · 暂停续录');

  // 保存配置的网络请求尚未返回时，轮询旧状态也不能重新启用“开始”。
  setPhase('idle');
  await context.refresh();
  await context.start();
  assert.equal(element('start-reminder').hidden, false);
  assert.equal(calls.includes('/api/save'), false, '提醒阶段不开始设备操作');
  await context.start();
  assert.equal(calls.includes('/api/record/start'), false, '重复按 q 不跳过提醒');
  await context.stop();
  assert.equal(element('start-reminder').hidden, true);
  assert.equal(calls.includes('/api/record/stop'), false, '取消提醒不发送硬件请求');
  await context.start();
  let resolveSave;
  pendingSave = new Promise(resolve=>{ resolveSave=resolve; });
  const starting = context.confirmStart();
  await context.refresh();
  assert.equal(element('btn-start').disabled, true);
  await context.confirmStart();
  assert.equal(calls.filter(url=>url==='/api/save').length, 1);
  resolveSave({ok:true, json:async()=>({ok:true})});
  await starting;
  assert.equal(calls.filter(url=>url==='/api/record/start').length, 1);
  assert.equal(element('btn-start').disabled, true);
  setPhase('idle');
  status.arms.can0.hardware = {
    communication:{state:'live', label:'反馈正在更新'}, mode:{code:1, label:'CAN 受控'},
    enabled:{state:'unknown', label:'使能状态未完整确认', joints:[true,false,null,true,true,true,true]},
    health:{state:'unknown', label:'未知（控制状态无完整新反馈）'},
    gripper:{present:true, enabled:null}, angles:null, guidance:'请检查状态',
  };
  await context.refresh();
  assert.match(element('joints-can0').textContent, /J1 使能.*J2 失能.*J3 未知/);
  assert.match(element('fault-can0').textContent, /未知/);
  assert.match(element('power-can0').textContent, /夹爪：未知/);
  assert.match(element('st-can0').textContent, /^最近操作：/);
  status.arms.can0.hardware.health = {state:'not_ready', label:'关节未使能／控制器未就绪（状态码 0x06）'};
  await context.refresh();
  assert.match(element('fault-can0').textContent, /未使能.*0x06/);
  assert.equal(element('btn-start').disabled, false, '已确认失能且状态 0x06 时允许用户发起准备使能');
  assert.equal(element('btn-home').disabled, false);
  status.arms.can0.hardware.health = {state:'fault', label:'急停'};
  await context.refresh();
  assert.equal(element('btn-start').disabled, true);
  assert.equal(element('btn-home').disabled, true);
  await context.checkArm('can0');
  assert.equal(calls.includes('/api/arm/status/can0'), true);

  status.connected = false;
  status.close_error = '磁盘错误';
  status.close_result = {
    targets:{can0:'主左',can1:'从左'},
    data:{state:'error',message:'数据收尾失败：磁盘错误'},
    arms:{can0:{label:'主左',state:'done',message:'已失能'}, can1:{label:'从左',state:'done',message:'已失能'}},
    preview:{state:'done',message:'相机预览已恢复'},warnings:[],
  };
  await context.refresh();
  assert.equal(element('retry-data').hidden, false);
  assert.equal(element('retry-data').disabled, false);
  assert.equal(element('retry-can0').hidden, true);
  assert.equal(element('retry-can1').hidden, true);
  assert.equal(element('recheck-can1').hidden, true);
  assert.match(element('close-data').textContent, /磁盘错误/);
  assert.equal(element('btn-disconnect').disabled, false, '数据收尾失败不阻止用户单独确认失能');
  assert.equal(element('btn-disconnect').textContent.includes('重试结束采集'), false);
  await context.retryClose('data');
  assert.equal(retryBodies[0].stage, 'data');
  assert.equal(retryBodies[0].channel, undefined);

  status.close_error = null;
  status.disable_error = 'can1 失能未确认';
  status.close_result.data = {state:'done',message:'数据已收尾'};
  status.close_result.arms.can1 = {label:'从左',state:'error',message:'夹爪未确认',details:{joints:'disabled',gripper:'unknown'}};
  await context.refresh();
  assert.equal(element('retry-data').hidden, true);
  assert.equal(element('retry-can0').hidden, true);
  assert.equal(element('retry-can1').hidden, false);
  assert.equal(element('recheck-can1').hidden, false);
  assert.equal(element('recheck-can0').hidden, true);
  assert.match(element('close-arm-can1').textContent, /关节已失能；夹爪未确认/);
  await context.retryClose('arm','can0');
  assert.equal(retryBodies.length, 1, '不能重试已完成的机械臂');
  await context.retryClose('arm','can1');
  assert.deepEqual(retryBodies[1], {stage:'arm',channel:'can1',targets:{can0:'主左',can1:'从左'}});
  context.confirm = ()=>{ throw new Error('只读复核不应要求失能授权'); };
  await context.retryClose('status','can1');
  assert.deepEqual(retryBodies[2], {stage:'status',channel:'can1',targets:{can0:'主左',can1:'从左'}});
  status.disabling = true;
  await context.refresh();
  assert.equal(element('retry-can1').disabled, true);
  assert.equal(element('recheck-can1').disabled, true);
  await context.retryClose('status','can1');
  assert.equal(retryBodies.length, 3, '设备忙时不能并发复核');
  setPhase('idle');status.recovery_required='急停恢复：请先结束采集，检查设备状态';
  await context.refresh();
  assert.equal(element('system-warning').hidden,false);
  assert.match(element('system-warning').textContent,/急停恢复/);
  assert.equal(element('btn-start').disabled,true);
  assert.equal(element('btn-home').disabled,true);
  assert.equal(element('btn-end-session').disabled,false,'运动锁定不阻止数据收尾');
  status.connected=false;await context.refresh();
  assert.equal(element('btn-disconnect').disabled,false,'运动锁定后仍可单独确认失能');
  status.recovery_required=null;status.config_error='配置读取失败：请修复配置后重启';
  await context.refresh();
  assert.equal(element('btn-save').disabled,true);
  assert.equal(element('btn-session-start').disabled,true);
  assert.match(element('system-warning').textContent,/配置读取失败/);
  setPhase('saved');
  status.record.timing={target_fps:30,frames:300,skipped_slots:0,cameras:{right:{frames:300,repeated_frames:20,max_age_ms:80}}};
  status.record.last_episode={repo_id:'nero/test',number:1,frames:300,elapsed_seconds:10,needs_review:true,termination_error:'反馈未更新'};
  await context.refresh();
  assert.match(element('rec-timing').textContent,/重复取用缓存 20 帧.*80.0 ms/);
  assert.match(element('rec-last').textContent,/本集需复核.*反馈未更新/);
  // 双对需要四个独立角色；右侧故障也会阻止开始，单对仍只检查选中的配对。
  setPhase('idle');status.connected=false;
  element('pair_mode').value='dual';
  await context.refresh();
  assert.equal(element('btn-session-start').disabled,true);
  assert.match(element('gate').textContent,/双对需要 4 条臂/);
  status.arm_count=4;
  status.arms.can2.label='主右';status.arms.can2.detected=true;
  status.arms.can3.label='从右';status.arms.can3.detected=true;
  await context.refresh();
  assert.equal(element('btn-session-start').disabled,false);
  assert.equal(element('single-leader').hidden,true);
  assert.equal(context.getConfig().pair_mode,'dual');
  assert.match(element('ready-pair').textContent,/主左.*从左.*主右.*从右/);
  status.arms.can3.hardware={health:{state:'fault',label:'急停'},communication:{},mode:{},enabled:{},gripper:{}};
  await context.refresh();
  assert.equal(element('btn-session-start').disabled,true);
  element('pair_mode').value='single';await context.refresh();
  assert.equal(element('btn-session-start').disabled,false,'单对不控制未选择的右侧');
  vm.runInContext(`CAMERAS={'HEAD-123':{type:'rgbd'},'WRIST':{type:'rgbd'},'fake-right':{type:'rgb'}}; MAP={'HEAD-123':'head','WRIST':'left','fake-right':'right'}`,context);
  element('record_camera').value='all';
  context.updateCameraSummary();
  assert.equal(context.getConfig().camera_map['HEAD-123'],'head');
  assert.match(element('record-camera-summary').textContent,/左深度.*头部深度/);
  setPhase('recording');
  status.active_cameras={left:'WRIST',left_depth:'WRIST',right:'/fake/right',head:'HEAD-123',head_depth:'HEAD-123'};
  await context.refresh();
  for(const name of ['left','left_depth','right','head','head_depth']){
    assert.equal(element('live-'+name).hidden,false);
    assert.match(element('live-image-'+name).src,new RegExp('/api/record/camera/'+name+'/frame'));
  }
  assert.equal(element('live-right_depth').hidden,true);
  assert.match(context.rgbdTimingSummary({frames:300,max_rgb_depth_skew_ms:0}),/时间同步未验证/);
  assert.doesNotMatch(context.rgbdTimingSummary({frames:300,max_rgb_depth_skew_ms:0}),/差.*0.*ms/);
  assert.match(context.rgbdTimingSummary({frames:300,timestamp_checked_frames:299,timestamp_unverified_frames:1,timestamp_global_frames:299,max_rgb_depth_skew_ms:0}),/最大差 0.00 ms.*1 帧未验证/);
  // 分辨率列表保留设备全部档位；按 FPS 标记不可用项，缩放尺寸写进配置。
  setPhase('idle'); status.connected=false;
  await context.refresh();
  element('fps').value=30;
  vm.runInContext(`MAP = {'fake-left':'left'}; resolutionProfiles = {left:{source:'fake-left',profiles:[
    {width:1920,height:1080,fps:[30]}, {width:1280,height:720,fps:[30]}, {width:640,height:480,fps:[15]}
  ]}}`, context);
  context.renderResolutionControls();
  assert.match(element('resolution-left').innerHTML,/1920x1080/);
  assert.match(element('resolution-left').innerHTML,/1280x720/);
  assert.match(element('resolution-left').innerHTML,/value="640x480" disabled/);
  element('resolution-left').value='1920x1080';context.changeResolution('left');
  assert.match(element('scale-left').innerHTML,/value="4"/);
  assert.doesNotMatch(element('scale-left').innerHTML,/value="8"/);
  element('scale-left').value='4';context.changeScale('left');
  assert.deepEqual(JSON.parse(JSON.stringify(context.getConfig().camera_resolution.left)),{width:1920,height:1080,scale:4});
  assert.match(element('resolution-output-left').textContent,/1920×1080 → 保存 480×270/);
  element('resolution-head').value='1280x720';context.changeResolution('head');
  element('scale-head').value='2';context.changeScale('head');
  vm.runInContext(`CAMERAS['HEAD-123']={type:'rgbd'}; MAP['HEAD-123']='head'`,context);
  element('record_camera').value='head';context.updateCameraSummary();
  assert.match(element('record-camera-summary').textContent,/头部 RGB 640×360、头部深度 640×360/);
  assert.match(element('depth-storage').textContent,/27\.6 MB\/s/);
  assert.deepEqual([...context.scaleOptions(800,1200)],[1,2]);
  const readsBefore = calls.filter(url=>url.includes('/resolutions/')).length;
  await context.refresh();
  assert.equal(calls.filter(url=>url.includes('/resolutions/')).length,readsBefore,'状态轮询不能查询硬件分辨率');
  console.log('页面状态、双对门禁、采样质量提示、恢复锁定和重复点击检查通过（未打开浏览器或连接设备）。');
})().catch(error=>{ console.error(error); process.exitCode=1; });
