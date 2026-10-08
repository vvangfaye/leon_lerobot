/* 固定提示音在本地播放；播报不进入采样线程，也不向网络发送采集数据。 */
class NeroVoice {
  constructor(manifest, update = () => {}) {
    this.manifest = manifest;
    this.update = update;
    this.enabled = true;
    this.voice = manifest.default_voice;
    this.volume = 0.85;
    this.current = null;
    this.locked = false;
    this.previous = null;
    this.jobs = new Map();
    this.playbackId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    this.playbackKey = 'nero.voice.playbackOwner';
    globalThis.document?.addEventListener('visibilitychange', () => {
      if (globalThis.document.hidden) this.stop();
      else this.claimPlayback();
    });
    // 多个可见窗口也会同时轮询；只让最近操作的页面发声。
    globalThis.addEventListener?.('focus', () => this.claimPlayback());
    globalThis.document?.addEventListener('pointerdown', () => this.claimPlayback());
    globalThis.addEventListener?.('storage', event => {
      if (event.key === this.playbackKey && !this.ownsPlayback()) this.stop();
    });
    globalThis.addEventListener?.('pagehide', () => this.stop());
    try {
      const saved = JSON.parse(localStorage.getItem('nero.voice') || '{}');
      this.enabled = saved.enabled !== false;
      if (manifest.voices[saved.voice]) this.voice = saved.voice;
      if (Number.isFinite(saved.volume)) this.volume = Math.max(0, Math.min(1, saved.volume));
    } catch (_) { /* 禁用浏览器存储时仍可使用默认设置。 */ }
    this.claimPlayback();
  }

  claimPlayback() {
    if (globalThis.document?.hidden) return;
    try { localStorage.setItem(this.playbackKey, this.playbackId); } catch (_) {}
  }

  ownsPlayback() {
    try {
      const owner = localStorage.getItem(this.playbackKey);
      return !owner || owner === this.playbackId;
    } catch (_) {
      // 存储被浏览器禁用时，退回只允许当前有焦点的窗口播报。
      return globalThis.document?.hasFocus?.() ?? true;
    }
  }

  configure({enabled = this.enabled, voice = this.voice, volume = this.volume} = {}) {
    this.stop();
    this.enabled = enabled;
    if (this.manifest.voices[voice]) this.voice = voice;
    this.volume = Math.max(0, Math.min(1, Number(volume)));
    try { localStorage.setItem('nero.voice', JSON.stringify({enabled:this.enabled, voice:this.voice, volume:this.volume})); } catch (_) {}
  }

  stop() { if (this.current) this.current.finish('cancelled'); }

  say(cue, {critical = false} = {}) {
    if (this.locked && !critical) return Promise.resolve('suppressed');
    const text = this.manifest.prompts[cue];
    if (!text) return Promise.resolve('missing');
    if (globalThis.document?.hidden) {
      this.stop();
      this.update('页面在后台，语音已暂停 · ' + text);
      return Promise.resolve('hidden');
    }
    if (!this.ownsPlayback()) {
      this.stop();
      this.update('语音由最近操作的工作台页面播放 · ' + text);
      return Promise.resolve('suppressed');
    }
    // 普通状态重复到达时续播原句；弹窗“重新播放”必须显式从头开始。
    if (!critical && this.current?.cue === cue) return this.current.promise;
    this.stop(); // 新状态替换旧提示，避免排队播报已过时的“正在录制”。
    if (!this.enabled || this.volume === 0) {
      this.update('语音已关闭 · ' + text);
      return Promise.resolve('muted');
    }
    const promise = new Promise(resolve => {
      let audio, timer;
      const entry = {cue, finish: result => {
        if (this.current !== entry) return;
        clearTimeout(timer);
        if (audio) { audio.onended = audio.onerror = null; audio.pause(); }
        this.current = null;
        if (result === 'failed') this.update('语音未播放，请检查音量或点击“试听”。文字提醒仍有效。');
        resolve(result);
      }};
      this.current = entry;
      try {
        audio = new Audio(`/static/voice/${this.voice}/${cue}.mp3?v=${this.manifest.version}`);
        audio.volume = this.volume;
        audio.onended = () => entry.finish('ended');
        audio.onerror = () => entry.finish('failed');
        timer = setTimeout(() => entry.finish('failed'), 30000);
        this.update('播报：' + text);
        Promise.resolve(audio.play()).catch(() => entry.finish('failed'));
      } catch (_) { entry.finish('failed'); }
    });
    if (this.current) this.current.promise = promise;
    return promise;
  }

  trackArm(ch, action) { this.jobs.set(ch, action); }

  observe(status) {
    const now = {
      phase: status.record.phase,
      episode: status.record.last_episode ? `${status.record.last_episode.repo_id}/${status.record.last_episode.number}` : null,
      home: status.homing,
      homeStatus: status.home_status,
      error: status.last_error,
    };
    const old = this.previous;
    this.previous = now;
    if (!old) return; // 打开/刷新页面不播报历史保存或历史失能结果。
    let cue = null;
    for (const [ch, action] of this.jobs) {
      if (status.device_busy.includes(ch)) continue;
      this.jobs.delete(ch);
      const message = status.arms[ch].status;
      const completed = {connect: /^已连接/, micro: /^微动完成/, single_home: /^关节.*已失能/};
      cue = completed[action]?.test(message) ? {connect:'connected', micro:'micro_done', single_home:'single_home_done'}[action] : 'failed';
    }
    if (now.phase !== old.phase) {
      const phases = {preparing:'preparing', aligning:'aligning', recording:'recording', stopping:'saving', saving:'saving', error:'failed'};
      cue = phases[now.phase] || cue;
      // 停止和写盘属于同一次保存，即使原句已播完也不再重复。
      if (old.phase === 'stopping' && now.phase === 'saving') cue = null;
      if (now.phase === 'idle' && ['preparing','aligning'].includes(old.phase)) cue = 'cancelled';
      if (now.phase === 'idle' && ['recording','stopping'].includes(old.phase)) cue = 'empty_episode';
    }
    if (now.episode && now.episode !== old.episode && ['saved','saved_review'].includes(now.phase)) cue = 'saved';
    if (!old.home && now.home) cue = 'home_started';
    if (old.home && !now.home) cue = status.home_error ? 'failed' : 'home_done';
    if (now.homeStatus !== old.homeStatus && !now.home) {
      if (status.home_error) cue = 'failed';
      else if (now.homeStatus?.startsWith('主臂和从臂均已回默认位置')) cue = 'home_done';
    }
    if (now.error && now.error !== old.error) cue = 'failed';
    if (cue) void this.say(cue);
  }
}
globalThis.NeroVoice = NeroVoice;
