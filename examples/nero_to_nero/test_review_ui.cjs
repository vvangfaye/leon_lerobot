/* 离线控制器验证：不启动浏览器、不连接 CAN 或相机。 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {Replay, ReviewPage, displayValue, frameAt} = require('./nero_platform/static/review.js');
let now = 0, nextRaf = 0;
const animations = new Map();
global.performance = {now: () => now};
global.requestAnimationFrame = callback => { animations.set(++nextRaf, callback); return nextRaf; };
global.cancelAnimationFrame = id => animations.delete(id);

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.listeners = {}; this.dataset = {}; this.attributes = {};
    this.textContent = ''; this.value = ''; this.hidden = false; this.disabled = false;
    this.currentTime = 0; this.paused = true; this.ended = false; this.clientWidth = 400;
    this.style = {};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  emit(name, event) { for (const callback of this.listeners[name] || []) callback(event); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; if (key === 'src') this.src = ''; }
  getContext() { return null; }
  getBoundingClientRect() { return {left: 0, width: 400}; }
  focus() { this.focused = true; }
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
  load() { this.loadCalls = (this.loadCalls || 0) + 1; }
}

function setup() {
  // 缺少实际 HTML 元素时测试直接失败，不自动生成不存在的 ID。
  const html = fs.readFileSync(__dirname + '/nero_platform/review.html', 'utf8');
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  global.document = {
    getElementById: id => { assert(elements.has(id), `HTML 缺少 #${id}`); return elements.get(id); },
    createElement: tag => new Element(tag), addEventListener() {}
  };
  global.window = {devicePixelRatio: 1, addEventListener() {}};
  elements.get('joint').value = '0';
  return elements;
}
function episode(index = 0) {
  return {
    repo_id: 'nero/review', index, frames: 4, fps: 10, duration: 0.4, pair_mode: 'dual',
    tasks: ['<img src=x onerror=alert(1)>'],
    state: Array.from({length: 4}, (_, i) => [...Array(7).fill(Math.PI / 2 + i), 0.012, ...Array(7).fill(0.3), 0.05]),
    action: Array.from({length: 4}, () => [...Array(7).fill(Math.PI), 0.02, ...Array(7).fill(0.4), 0.06]),
    videos: [{camera: 'left', start: 0.6, end: 1, codec: 'av1'}, {camera: 'right', start: 1.2, end: 1.6, codec: 'av1'}],
    quality: {state: 'unknown', warnings: ['未找到报告'], report_path: 'meta/report.json', summary: null},
    events: [{frame: 2, time: 0.2, host_time: 0.3, message: '重复帧'}]
  };
}
const flush = () => new Promise(resolve => setImmediate(resolve));

async function main() {
  assert.equal(displayValue(Math.PI, false), 180);
  assert.equal(displayValue(0.012, true), 12);
  assert.equal(frameAt(0.4, 10, 4), 3);
  assert.equal(frameAt(-1, 10, 4), 0);

  // 旧合并视频的起始偏移、两路同步、集末钳制和暂停。
  const frames = [], statuses = [];
  const replay = new Replay(frame => frames.push(frame), (playing, message) => statuses.push([playing, message]));
  const left = new Element('video'), right = new Element('video');
  replay.load(episode(), [{video: left, start: 0.6, ready: true}, {video: right, start: 1.2, ready: true}]);
  replay.seek(2);
  assert(Math.abs(left.currentTime - 0.8) < 1e-8);
  assert(Math.abs(right.currentTime - 1.4) < 1e-8);
  assert.equal(frames.at(-1), 2);
  await replay.play();
  left.currentTime = 0.85; right.currentTime = 3;
  replay.tick();
  assert(Math.abs(right.currentTime - 1.45) < 1e-8);
  left.currentTime = 1;
  replay.tick();
  assert.equal(replay.playing, false);
  assert.equal(replay.frame, 3);
  assert(left.paused && right.paused);
  assert(left.currentTime < 1, '集末应停在最后一个有效帧，不能落入下一集');
  replay.clear();
  assert.equal(left.src, '');
  assert.equal(replay.media.length, 0);

  // 旧播放请求迟到时，不得重新启动已被取消的动画。
  let resolvePlay;
  left.play = () => new Promise(resolve => { resolvePlay = resolve; });
  replay.load(episode(), [{video: left, start: 0.6, ready: true}]);
  const pending = replay.play();
  replay.pause();
  resolvePlay();
  await pending;
  assert.equal(replay.playing, false);
  assert.equal(replay.raf, null);
  replay.clear();

  // 没有视频时，曲线也可按名义时间播放到集末。
  now = 0;
  replay.load(episode(), []);
  await replay.play();
  now = 250; replay.tick();
  assert.equal(replay.frame, 2);
  now = 450; replay.tick();
  assert.equal(replay.playing, false);
  assert.equal(replay.frame, 3);

  // 页面实际 DOM ID、左右维度、单位、质量信息和失败的视频。
  const elements = setup(), page = new ReviewPage(), data = episode();
  page.showEpisode(data);
  assert.equal(page.plots.length, 4);
  assert.equal(page.plots[0].state[0], 90);
  assert(Math.abs(page.plots[1].state[0] - 0.3 * 180 / Math.PI) < 1e-8);
  assert.equal(page.plots[2].state[0], 12);
  assert.equal(page.plots[3].state[0], 50);
  assert.equal(elements.get('tasks').textContent, '任务：<img src=x onerror=alert(1)>');
  assert.equal(elements.get('tasks').children.length, 0, '任务文本不可作为 HTML 执行');
  elements.get('events').children[0].emit('click');
  assert.equal(page.replay.frame, 2);
  const first = page.replay.media[0];
  first.video.emit('loadeddata');
  assert(Math.abs(first.video.currentTime - 0.8) < 1e-8, '视频迟到时应追上已选帧');
  first.video.emit('error');
  assert.equal(first.bad, true);
  assert.equal(page.replay.playing, false);
  const oldVideo = page.replay.media[1].video;
  page.clearEpisode('读取失败');
  oldVideo.emit('loadeddata');
  assert.equal(page.episode, null);
  assert.equal(elements.get('episode-content').hidden, true);
  assert.equal(elements.get('empty').textContent, '读取失败');
  assert.equal(oldVideo.src, '');

  // 切换集时忽略旧响应，即使底层 fetch 未及时执行 abort。
  const requests = [];
  global.fetch = (url, options) => new Promise(resolve => requests.push({url, options, resolve}));
  page.repo = 'nero/review';
  const firstRequest = page.loadEpisode(0), secondRequest = page.loadEpisode(1);
  assert(requests[0].options.signal.aborted);
  requests[1].resolve({ok: true, json: async () => episode(1)});
  await secondRequest;
  requests[0].resolve({ok: true, json: async () => episode(0)});
  await firstRequest;
  assert.equal(page.episode.index, 1);
  assert(requests.every(request => request.options.method === undefined), '回放仅执行 GET');
  const failed = page.loadEpisode(2);
  requests[2].resolve({ok: false, status: 409, json: async () => ({msg: '请先结束采集，无需失能'})});
  await failed;
  assert.equal(page.episode, null);
  assert.match(elements.get('empty').textContent, /无需失能/);

  // 活动会话不发起重型数据读取；选择另一集或刷新可恢复。
  page.datasets = [{repo_id: 'nero/review', blocked: '会话尚未收尾', pair_mode: 'dual'}];
  const count = requests.length;
  page.loadList(0);
  assert.equal(requests.length, count);
  assert.equal(elements.get('list-status').textContent, '会话尚未收尾');
  page.replay.clear();
  // 12-bit 深度使用显式解码的伪彩色预览，不交给浏览器当普通 RGB 播放。
  const depthEpisode = episode(3);
  depthEpisode.videos = [{camera:'depth',start:0,end:.4,is_depth:true,codec:'hevc'}];
  depthEpisode.quality.summary = {cameras:{depth:{frames:4,repeated_frames:0,max_age_ms:1,max_rgb_depth_skew_ms:0}}};
  page.showEpisode(depthEpisode);
  let cameraQuality = elements.get('camera-quality').children.map(child=>child.textContent).join(' ');
  assert.match(cameraQuality,/时间同步未验证/);
  assert.doesNotMatch(cameraQuality,/最大差 0/);
  depthEpisode.quality.summary.cameras.depth={...depthEpisode.quality.summary.cameras.depth,timestamp_checked_frames:3,timestamp_unverified_frames:1,timestamp_global_frames:3};
  page.clearEpisode();
  page.showEpisode(depthEpisode);
  cameraQuality = elements.get('camera-quality').children.map(child=>child.textContent).join(' ');
  assert.match(cameraQuality,/最大差 0.00 ms.*1 帧未验证/);
  assert.equal(page.replay.media.length,0);
  assert.equal(page.depthViews.length,1);
  const depthView = page.depthViews[0];
  assert.match(depthView.image.src,/frame_index=0/);
  page.replay.seek(2);
  depthView.image.onload();
  assert.match(depthView.image.src,/frame_index=2/,'暂停时迟到的深度请求完成后应追上最后选择的帧');
  page.clearEpisode();
  depthView.image.onload();
  assert.equal(page.depthViews.length,0);
  await flush();
  console.log('review UI: 时间轴、两路视频、左右曲线、异常定位、异步取消与只读请求通过');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
