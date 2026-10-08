'use strict';

// 单一时间轴。原始数组保留 rad / m，单位转换只发生在显示时。
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const displayValue = (value, gripper) => value * (gripper ? 1000 : 180 / Math.PI);
const frameAt = (seconds, fps, frames) => clamp(Math.floor(seconds * fps + 1e-6), 0, frames - 1);
const cameraLabel = name => ({left: '左相机', right: '右相机', head: '头部 RGB', depth: '深度（旧字段）', left_depth:'左深度', right_depth:'右深度', head_depth:'头部深度'}[name] || name);
const qualityLabel = state => ({ok: '节拍未见异常', review: '需复核', unknown: '缺少报告'}[state] || '未知');

class Replay {
  constructor(onFrame, onStatus) {
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.episode = null;
    this.media = [];
    this.frame = 0;
    this.rate = 1;
    this.playing = false;
    this.generation = 0;
    this.raf = null;
  }
  clear() {
    this.pause();
    this.generation++;
    for (const item of this.media) {
      item.video.removeAttribute('src');
      item.video.load();
    }
    this.media = [];
    this.episode = null;
    this.frame = 0;
  }
  load(episode, media) {
    this.clear();
    this.episode = episode;
    this.media = media;
    this.seek(0);
  }
  active() { return this.media.filter(item => !item.bad && item.ready); }
  seek(frame) {
    if (!this.episode) return;
    this.pause();
    this.frame = clamp(Math.round(frame), 0, this.episode.frames - 1);
    const seconds = this.frame / this.episode.fps;
    for (const item of this.active()) item.video.currentTime = item.start + seconds;
    this.onFrame(this.frame);
  }
  pause() {
    this.playing = false;
    this.generation++;
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
    for (const item of this.media) item.video.pause();
    this.onStatus(false);
  }
  async play() {
    if (!this.episode || this.playing) return;
    if (this.media.some(item => !item.bad && !item.ready)) {
      this.onStatus(false, '视频正在加载；可先拖动时间轴查看曲线。');
      return;
    }
    if (this.frame >= this.episode.frames - 1) this.seek(0);
    const generation = ++this.generation;
    this.playing = true;
    const active = this.active();
    try {
      await Promise.all(active.map(item => {
        item.video.currentTime = item.start + this.frame / this.episode.fps;
        item.video.playbackRate = this.rate;
        return item.video.play();
      }));
      if (generation !== this.generation) return;
      this.anchor = performance.now();
      this.anchorSeconds = this.frame / this.episode.fps;
      this.onStatus(true, active.length ? '视频与曲线按保存时间轴播放。' : '没有可播放的视频，仅播放数据曲线。');
      this.raf = requestAnimationFrame(() => this.tick());
    } catch (error) {
      if (generation !== this.generation) return;
      this.pause();
      this.onStatus(false, `视频未能播放：${error.message}。可拖动时间轴查看数据。`);
    }
  }
  tick() {
    if (!this.playing || !this.episode) return;
    const active = this.active(), master = active[0];
    const seconds = master ? Math.max(0, master.video.currentTime - master.start) :
      this.anchorSeconds + (performance.now() - this.anchor) / 1000 * this.rate;
    if (seconds >= this.episode.duration - 0.0001 || (master && master.video.ended)) {
      this.seek(this.episode.frames - 1);
      this.onStatus(false, '本集播放完毕。');
      return;
    }
    for (const item of active.slice(1)) {
      if (Math.abs(item.video.currentTime - item.start - seconds) > Math.max(0.08, 2 / this.episode.fps)) {
        item.video.currentTime = item.start + seconds;
      }
    }
    const frame = frameAt(seconds, this.episode.fps, this.episode.frames);
    if (frame !== this.frame) { this.frame = frame; this.onFrame(frame); }
    this.raf = requestAnimationFrame(() => this.tick());
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

class ReviewPage {
  constructor() {
    this.get = id => document.getElementById(id);
    this.datasets = [];
    this.repo = '';
    this.offset = 0;
    this.total = 0;
    this.episode = null;
    this.plots = [];
    this.depthViews = [];
    this.eventOffset = 0;
    this.requests = new Map();
    this.replay = new Replay(frame => this.renderFrame(frame), (playing, message) => {
      this.get('play').textContent = playing ? '暂停' : '播放';
      if (message !== undefined) this.get('play-status').textContent = message;
    });
  }
  cancel(key) {
    this.requests.get(key)?.abort();
    this.requests.delete(key);
  }
  async request(key, path, params, accept, reject) {
    this.cancel(key);
    const controller = new AbortController();
    this.requests.set(key, controller);
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(path + (params ? '?' + new URLSearchParams(params) : ''), {
        signal: controller.signal, cache: 'no-store'
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.msg || `读取失败（${response.status}）`);
      if (this.requests.get(key) === controller) accept(data);
    } catch (error) {
      if (this.requests.get(key) === controller) {
        reject(error.name === 'AbortError' ? '读取超时，请稍后刷新重试。' : error.message);
      }
    } finally {
      clearTimeout(timer);
      if (this.requests.get(key) === controller) this.requests.delete(key);
    }
  }
  clearEpisode(message = '选择一集开始查看。') {
    this.cancel('episode');
    this.replay.clear();
    for(const view of this.depthViews) view.image.removeAttribute('src');
    this.depthViews = [];
    this.episode = null;
    this.plots = [];
    this.get('videos').replaceChildren();
    this.get('plots').replaceChildren();
    this.get('episode-content').hidden = true;
    this.get('empty').hidden = false;
    this.get('empty').textContent = message;
    for (const id of ['play', 'timeline', 'frame-prev', 'frame-next']) this.get(id).disabled = true;
  }
  refresh() {
    this.cancel('list');
    this.clearEpisode('正在读取本地数据集…');
    this.get('refresh').disabled = true;
    this.get('dataset').disabled = true;
    this.get('episodes').replaceChildren();
    this.get('list-status').textContent = '';
    this.get('page-message').textContent = '';
    this.get('list-prev').disabled = this.get('list-next').disabled = true;
    return this.request('datasets', '/api/review/datasets', null, data => {
      this.datasets = data.datasets;
      const select = this.get('dataset');
      select.replaceChildren();
      for (const entry of this.datasets) {
        const option = element('option', entry.repo_id + (entry.blocked ? ' · 会话未收尾' : entry.error ? ' · 不可读取' : ''));
        option.value = entry.repo_id;
        select.append(option);
      }
      this.get('refresh').disabled = false;
      select.disabled = !this.datasets.length;
      if (!this.datasets.length) {
        select.append(element('option', '暂无本地数据集'));
        this.get('dataset-info').textContent = '';
        this.clearEpisode('还没有可查看的本地数据集。先在工作台完成采集与收尾，再刷新列表。');
        return;
      }
      this.repo = [this.repo, data.selected_repo].find(repo => this.datasets.some(d => d.repo_id === repo)) || this.datasets[0].repo_id;
      select.value = this.repo;
      this.loadList(0);
    }, message => {
      this.get('refresh').disabled = false;
      this.get('page-message').textContent = message;
      this.clearEpisode('列表加载失败，可点击“刷新列表”重试。');
    });
  }
  loadList(offset) {
    this.cancel('list');
    this.clearEpisode();
    this.offset = offset;
    this.get('episodes').replaceChildren();
    this.get('list-prev').disabled = this.get('list-next').disabled = true;
    this.get('list-page').textContent = '';
    const entry = this.datasets.find(d => d.repo_id === this.repo);
    this.get('dataset-info').textContent = entry?.error ? '' :
      `${entry?.pair_mode === 'dual' ? '左右双对 · 16 维' : '单对 · 8 维'} · ${entry?.episodes ?? 0} 集 · ${entry?.fps ?? '—'} FPS`;
    if (!entry || entry.error || entry.blocked) {
      this.get('list-status').textContent = entry?.blocked || entry?.error || '数据集不存在，请刷新。';
      return;
    }
    this.get('list-status').textContent = '正在读取集索引…';
    return this.request('list', '/api/review/episodes', {repo_id: this.repo, offset, limit: 20}, data => {
      this.total = data.total;
      this.get('list-status').textContent = data.total ? '' : '此数据集尚无已保存的集。';
      this.get('list-page').textContent = data.total ? `${Math.floor(offset / 20) + 1} / ${Math.ceil(data.total / 20)}` : '';
      this.get('list-prev').disabled = offset === 0;
      this.get('list-next').disabled = offset + 20 >= data.total;
      for (const episode of data.episodes) {
        const button = element('button', `第 ${episode.index + 1} 集 · ${episode.duration.toFixed(1)} 秒\n${episode.frames} 帧 · ${qualityLabel(episode.quality.state)}`, 'episode');
        button.dataset.index = String(episode.index);
        button.setAttribute('aria-current', 'false');
        button.addEventListener('click', () => this.loadEpisode(episode.index));
        this.get('episodes').append(button);
      }
    }, message => { this.get('list-status').textContent = message; });
  }
  loadEpisode(index) {
    this.clearEpisode(`正在读取第 ${index + 1} 集…`);
    for (const button of this.get('episodes').children) button.setAttribute('aria-current', String(Number(button.dataset.index) === index));
    return this.request('episode', '/api/review/episode', {repo_id: this.repo, episode_index: index},
      episode => this.showEpisode(episode), message => this.clearEpisode(message));
  }
  showEpisode(episode) {
    this.episode = episode;
    this.eventOffset = 0;
    this.get('empty').hidden = true;
    this.get('episode-content').hidden = false;
    this.get('episode-title').textContent = `${episode.repo_id} · 第 ${episode.index + 1} 集`;
    this.get('episode-info').textContent = `episode_index ${episode.index} · ${episode.frames} 帧 · ${episode.fps} FPS · 按数据集 FPS 播放 ${episode.duration.toFixed(3)} 秒 · ${episode.pair_mode === 'dual' ? '左右双对，左 8 维 + 右 8 维' : '单对，8 维'}`;
    this.get('tasks').textContent = `任务：${episode.tasks.join('；')}`;
    this.get('quality-badge').textContent = qualityLabel(episode.quality.state);
    this.get('timeline').max = String(episode.frames - 1);
    for (const id of ['play', 'timeline', 'frame-prev', 'frame-next']) this.get(id).disabled = false;
    this.get('play-status').textContent = '拖动时间轴或点击播放。';
    const media = [];
    for (const source of episode.videos) {
      const card = element('article', undefined, 'video-card');
      card.append(element('h3', cameraLabel(source.camera)));
      const status = element('div', source.error || `正在加载 ${source.codec || ''} 视频…`, 'video-status');
      if(!source.error && source.is_depth){
        const image = element('img');
        image.alt = '已保存深度的伪彩色预览';
        image.style.width = '100%';
        const view = {image, source, status, episode, pending:false, requested:null, desired:0, updated:0};
        const completed = error => {
          if(this.episode !== episode) return;
          view.pending = false;
          status.textContent = error ? '深度解码失败，可重新选集重试。' : `深度帧索引 ${view.requested} · 0～10 m 固定伪彩色，零值黑色 · 播放时最多 5 次/秒，暂停后按所选帧查看`;
          if(!this.replay.playing && view.desired !== view.requested) this.renderDepthViews(view.desired);
        };
        image.onload = ()=>completed(false);
        image.onerror = ()=>completed(true);
        this.depthViews.push(view);
        card.append(image);
      }else if (!source.error) {
        const video = element('video');
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.setAttribute('aria-label', `${cameraLabel(source.camera)}已保存视频`);
        const item = {video, start: source.start, end: source.end, ready: false, bad: false};
        media.push(item);
        video.addEventListener('loadeddata', () => {
          if (this.episode !== episode || item.ready || item.bad) return;
          item.ready = true;
          video.currentTime = source.start + this.replay.frame / episode.fps;
          status.textContent = `${source.codec || '视频'} · 文件内 ${source.start.toFixed(3)}～${source.end.toFixed(3)} 秒`;
        });
        video.addEventListener('error', () => {
          if (this.episode !== episode) return;
          item.bad = true;
          this.replay.pause();
          status.textContent = `视频读取或解码失败（${source.codec || '未知编码'}）。可尝试支持该编码的新版浏览器；仍可拖动时间轴查看曲线。`;
          this.get('play-status').textContent = '部分视频不可用；再次播放会使用其余视频或仅播放曲线。';
        });
        video.addEventListener('timeupdate', () => {
          if(this.episode !== episode) return;
          const quality = video.getVideoPlaybackQuality?.();
          if(quality && !item.bad) status.textContent = `${source.codec || '视频'} · 文件内 ${source.start.toFixed(3)}～${source.end.toFixed(3)} 秒 · 浏览器解码丢帧 ${quality.droppedVideoFrames}/${quality.totalVideoFrames}（仅播放器统计）`;
          if (this.episode === episode && this.replay.playing && video.currentTime >= source.end) {
            this.replay.seek(episode.frames - 1);
          }
        });
        video.addEventListener('waiting', ()=>{ if(this.episode === episode && !item.bad) status.textContent = '浏览器正在缓冲 / 等待解码；这不是采集报告的丢帧统计。'; });
        video.src = '/api/review/video?' + new URLSearchParams({repo_id: episode.repo_id, episode_index: episode.index, camera: source.camera}) + `#t=${source.start},${source.end}`;
        card.append(video);
      }
      card.append(status);
      this.get('videos').append(card);
    }
    if (!episode.videos.length) this.get('videos').append(element('p', '本集没有视频，仅显示数据曲线。', 'muted'));
    this.replay.load(episode, media);
    this.buildPlots();
    this.renderQuality();
    this.renderEvents();
  }
  buildPlots() {
    this.plots = [];
    this.get('plots').replaceChildren();
    if (!this.episode) return;
    const sides = this.episode.pair_mode === 'dual' ? [['左侧', 0], ['右侧', 8]] : [['单对', 0]];
    for (const gripper of [false, true]) for (const [label, offset] of sides) {
      const joint = Number(this.get('joint').value), dimension = offset + (gripper ? 7 : joint);
      const section = element('div', undefined, 'plot');
      section.append(element('h3', `${label} · ${gripper ? '夹爪（mm）' : `J${joint + 1}（°）`}`));
      const canvas = element('canvas');
      canvas.setAttribute('aria-label', `${label}${gripper ? '夹爪' : '关节'}状态和动作曲线；使用上方时间轴查看数值`);
      const values = element('div', '', 'values');
      const state = this.episode.state.map(row => displayValue(row[dimension], gripper));
      const action = this.episode.action.map(row => displayValue(row[dimension], gripper));
      let low = Infinity, high = -Infinity;
      for (let i = 0; i < state.length; i++) { low = Math.min(low, state[i], action[i]); high = Math.max(high, state[i], action[i]); }
      const pad = Math.max((high - low) * 0.12, gripper ? 0.5 : 1);
      const plot = {canvas, values, state, action, low: low - pad, high: high + pad, gripper};
      canvas.addEventListener('click', event => {
        const rect = canvas.getBoundingClientRect();
        this.replay.seek(clamp((event.clientX - rect.left - 53) / Math.max(1, rect.width - 65), 0, 1) * (this.episode.frames - 1));
      });
      section.append(canvas, values);
      this.get('plots').append(section);
      this.plots.push(plot);
    }
    this.renderFrame(this.replay.frame);
  }
  renderFrame(frame) {
    if (!this.episode) return;
    this.renderDepthViews(frame);
    this.get('timeline').value = String(frame);
    this.get('position').textContent = `${(frame / this.episode.fps).toFixed(3)} / ${this.episode.duration.toFixed(3)} 秒 · 第 ${frame + 1} / ${this.episode.frames} 帧（索引 ${frame}）`;
    this.get('frame-prev').disabled = frame === 0;
    this.get('frame-next').disabled = frame === this.episode.frames - 1;
    for (const plot of this.plots) {
      const a = plot.state[frame], b = plot.action[frame], unit = plot.gripper ? 'mm' : '°';
      plot.values.textContent = `反馈 ${a.toFixed(2)} ${unit} · 目标 ${b.toFixed(2)} ${unit} · 目标−反馈 ${(b - a).toFixed(2)} ${unit}`;
      this.drawPlot(plot, frame);
    }
  }
  renderDepthViews(frame){
    for(const view of this.depthViews){
      view.desired = frame;
      if(view.pending || view.requested === frame || (this.replay.playing && Date.now()-view.updated < 200)) continue;
      view.pending = true; view.requested = frame; view.updated = Date.now();
      view.image.src = '/api/review/depth?' + new URLSearchParams({repo_id:this.episode.repo_id, episode_index:this.episode.index, camera:view.source.camera, frame_index:frame});
    }
  }
  drawPlot(plot, frame) {
    const {canvas, low, high} = plot, width = Math.max(canvas.clientWidth, 150), height = 190;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      plot.background = null;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const x = i => 53 + i / Math.max(1, this.episode.frames - 1) * (width - 65);
    const y = v => 12 + (high - v) / (high - low) * (height - 40);
    // 背景曲线只在选关节或尺寸变化时绘制；逐帧仅恢复背景并移动游标。
    if (!plot.background) {
      ctx.clearRect(0, 0, width, height); ctx.font = '11px system-ui'; ctx.setLineDash([]);
      for (let i = 0; i <= 4; i++) {
        const value = low + (high - low) * i / 4;
        ctx.strokeStyle = '#2c394b'; ctx.beginPath(); ctx.moveTo(53, y(value)); ctx.lineTo(width - 12, y(value)); ctx.stroke();
        ctx.fillStyle = '#adbbce'; ctx.fillText(value.toFixed(1), 4, y(value) + 4);
      }
      ctx.fillText('0 s', 53, height - 5);
      ctx.fillText(`${((this.episode.frames - 1) / this.episode.fps).toFixed(2)} s`, Math.max(53, width - 75), height - 5);
      for (const [values, color, dash] of [[plot.state, '#83bcfa', []], [plot.action, '#ffbe79', [5, 4]]]) {
        ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dash); ctx.beginPath();
        values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
        ctx.stroke();
        if (values.length === 1) { ctx.fillStyle = color; ctx.fillRect(x(0) - 2, y(values[0]) - 2, 4, 4); }
      }
      plot.background = document.createElement('canvas');
      plot.background.width = canvas.width; plot.background.height = canvas.height;
      plot.background.getContext('2d').drawImage(canvas, 0, 0);
    } else {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(plot.background, 0, 0, width, height);
    }
    ctx.strokeStyle = '#f3f6fa'; ctx.lineWidth = 1; ctx.setLineDash([]); ctx.beginPath();
    ctx.moveTo(x(frame), 7); ctx.lineTo(x(frame), height - 25); ctx.stroke();
  }
  renderQuality() {
    const quality = this.episode.quality, summary = quality.summary;
    this.get('quality-warnings').textContent = quality.warnings.join('\n') || '节拍报告未记录异常；仍需结合视频确认动作内容。';
    this.get('report-path').textContent = `报告位置：${this.episode.repo_id}/${quality.report_path}`;
    this.get('metrics').replaceChildren();
    this.get('camera-quality').replaceChildren();
    if (!summary) return;
    const metrics = [
      ['目标 FPS', summary.target_fps], ['实测采样 FPS', summary.effective_fps],
      ['已写入帧 / 采样尝试', `${summary.frames} / ${summary.attempts}`],
      ['实际录制时长（秒）', summary.elapsed_seconds], ['按数据集 FPS 播放（秒）', summary.nominal_playback_seconds],
      ['最大采样间隔（ms）', summary.max_interval_ms], ['平均处理（ms）', summary.mean_processing_ms],
      ['最长处理（ms）', summary.max_processing_ms], ['处理超时次数', summary.over_budget_samples],
      ['错过采样时刻', summary.skipped_slots], ['采样失败次数', summary.failed_samples]
    ];
    for (const [label, value] of metrics) {
      const metric = element('div', undefined, 'metric');
      metric.append(element('span', label), element('strong', value ?? '—'));
      this.get('metrics').append(metric);
    }
    for (const [name, stats] of Object.entries(summary.cameras || {})) {
      this.get('camera-quality').append(element('p', `${cameraLabel(name)}：写入 ${stats.frames} 帧，重复取用缓存 ${stats.repeated_frames} 帧，最大帧龄 ${Number(stats.max_age_ms).toFixed(2)} ms。`));
      if(name === 'depth' || name.endsWith('_depth')) {
        const checked = stats.timestamp_checked_frames || 0;
        const unverified = stats.timestamp_unverified_frames ?? stats.frames ?? 0;
        const timing = checked && Number.isFinite(stats.max_rgb_depth_skew_ms)
          ? `有效时间戳已检查 ${checked} 帧（设备 ${stats.timestamp_device_frames || 0} 帧、全局 ${stats.timestamp_global_frames || 0} 帧），最大差 ${stats.max_rgb_depth_skew_ms.toFixed(2)} ms；另有 ${unverified} 帧未验证。时差检查不代表同步曝光。`
          : `${unverified} 帧时间戳不可用或未通过有效性检查，时间同步未验证。`;
        this.get('camera-quality').append(element('p', `深度全无效 ${stats.all_invalid_frames || 0} 帧，含超 10 米数值 ${stats.above_range_frames || 0} 帧。RGB/深度：${timing}`));
      }
    }
    for(const [role, device] of Object.entries(quality.camera_devices || {})){
      if(device.type !== "rgbd" && !device.alignment) continue;
      this.get('camera-quality').append(element('p', `${cameraLabel(role)} RGBD 空间对齐：${device.alignment === 'hardware_D2C' ? '硬件 D2C' : '未记录'}；SDK 帧同步：${device.frame_sync_requested ? '已请求' : (device.frame_sync_error ? '请求失败：'+device.frame_sync_error : '未记录')}；全局时间戳启用请求：${device.global_timestamp_enabled ? '成功，实际可用性见上方检查' : (device.global_timestamp_supported === false ? '设备不支持' : '未成功或未记录')}。这些状态不代表与其他相机、机械臂同步曝光。`));
    }
    for (const pair of quality.pairs || []) {
      const side = ({left: '左侧', right: '右侧', single: '单对'}[pair.name] || pair.name || '单对');
      this.get('camera-quality').append(element('p', `${side}：${pair.leader_label || ''} ${pair.leader || ''} → ${pair.follower_label || ''} ${pair.follower || ''}`));
    }
  }
  renderEvents() {
    const events = this.episode.events;
    this.get('events-title').textContent = `异常时刻 · ${events.length} 项`;
    this.get('events').replaceChildren();
    if (!events.length) this.get('events').append(element('p', this.episode.quality.summary ? '逐帧报告未定位到异常。' : '没有可用的逐帧报告。', 'muted'));
    for (const event of events.slice(this.eventOffset, this.eventOffset + 12)) {
      const button = element('button', `${event.time.toFixed(3)} 秒 · 帧索引 ${event.frame} · ${event.message}（主机采样 ${event.host_time.toFixed(3)} 秒）`);
      button.addEventListener('click', () => { this.replay.seek(event.frame); this.get('timeline').focus(); });
      this.get('events').append(button);
    }
    this.get('event-prev').disabled = this.eventOffset === 0;
    this.get('event-next').disabled = this.eventOffset + 12 >= events.length;
    this.get('event-page').textContent = events.length ? `${Math.floor(this.eventOffset / 12) + 1} / ${Math.ceil(events.length / 12)}` : '';
  }
  init() {
    this.get('refresh').addEventListener('click', () => this.refresh());
    this.get('dataset').addEventListener('change', () => { this.repo = this.get('dataset').value; this.loadList(0); });
    this.get('list-prev').addEventListener('click', () => this.loadList(Math.max(0, this.offset - 20)));
    this.get('list-next').addEventListener('click', () => this.loadList(this.offset + 20));
    this.get('play').addEventListener('click', () => this.replay.playing ? this.replay.pause() : this.replay.play());
    this.get('timeline').addEventListener('input', () => this.replay.seek(Number(this.get('timeline').value)));
    this.get('frame-prev').addEventListener('click', () => this.replay.seek(this.replay.frame - 1));
    this.get('frame-next').addEventListener('click', () => this.replay.seek(this.replay.frame + 1));
    this.get('rate').addEventListener('change', () => { this.replay.pause(); this.replay.rate = Number(this.get('rate').value); });
    this.get('joint').addEventListener('change', () => this.buildPlots());
    this.get('event-prev').addEventListener('click', () => { this.eventOffset -= 12; this.renderEvents(); });
    this.get('event-next').addEventListener('click', () => { this.eventOffset += 12; this.renderEvents(); });
    window.addEventListener('resize', () => { if (this.episode) this.renderFrame(this.replay.frame); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.replay.pause(); });
    // 保留返回缓存中的视频和已选帧，返回本页后仍可继续播放。
    window.addEventListener('pagehide', () => this.replay.pause());
    this.refresh();
  }
}

// Node 使用同一份控制器做离线时序验证；浏览器不加载测试依赖。
if (typeof module !== 'undefined' && module.exports) module.exports = {Replay, ReviewPage, displayValue, frameAt};
else new ReviewPage().init();
