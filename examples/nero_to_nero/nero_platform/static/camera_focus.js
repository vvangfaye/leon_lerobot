// 每台物理相机独立的对焦控件；轮询只更新缓存，写设备由用户操作触发。
class NeroCameraFocus {
  constructor({post, refresh}) {
    this.post = post; this.refresh = refresh;
    this.rows = new Map(); this.drafts = new Map(); this.messages = new Map();
    this.settings = {}; this.checks = {}; this.pending = null; this.locked = true;
  }
  load(settings) { this.settings = JSON.parse(JSON.stringify(settings || {})); }
  clearRows() { this.rows.clear(); }
  lock() { this.locked = true; for(const id of this.rows.keys()) this.render(id); }
  add(card, id, info) {
    const box = document.createElement('div'); box.className = 'focus-controls';
    const makeChoice = (mode, text) => {
      const label = document.createElement('label'); label.className = 'focus-choice';
      const input = document.createElement('input'); input.type = 'checkbox'; input.disabled = true;
      input.setAttribute('aria-label', `${id} ${text}`);
      const caption = document.createElement('span'); caption.textContent = text;
      label.append(input, caption); box.append(label);
      input.addEventListener('change', () => this.choose(id, mode, input.checked));
      return input;
    };
    const manual = makeChoice('manual', '定焦'), auto = makeChoice('auto', '自动对焦');
    const rangeLabel = document.createElement('label'); rangeLabel.textContent = '焦距（设备对焦值）';
    const slider = document.createElement('input'); slider.type = 'range'; slider.disabled = true;
    slider.setAttribute('aria-label', `${id} 定焦值`);
    const value = document.createElement('span'); value.className = 'focus-value';
    rangeLabel.append(value, slider);
    const note = document.createElement('div'); note.className = 'muted';
    const retry = document.createElement('button'); retry.textContent = '应用并保存'; retry.hidden = true;
    retry.addEventListener('click', () => { if(!retry.disabled) void this.apply(id); });
    box.append(rangeLabel, note, retry); card.append(box);
    this.rows.set(id, {box, manual, auto, slider, value, note, retry, info});
    slider.addEventListener('input', () => {
      if(slider.disabled) return;
      this.drafts.set(id, {mode:'manual', value:Number(slider.value)});
      this.messages.set(id, '松开滑块后应用并保存'); this.render(id);
    });
    slider.addEventListener('change', () => { if(!slider.disabled) void this.apply(id); });
    this.render(id);
  }
  update(status, ready) {
    this.locked = !ready || !!status.config_error || status.connected || status.recording || status.homing
      || status.disabling || status.ending || !!status.device_busy?.length;
    this.checks = status.camera_focus || {};
    if(status.camera_focus_settings && !this.pending) this.settings = status.camera_focus_settings;
    for(const id of this.rows.keys()) this.render(id);
  }
  selection(id) {
    return this.drafts.get(id) || this.settings[id] || this.checks[id]?.setting || {mode:null, value:null};
  }
  render(id) {
    const row = this.rows.get(id); if(!row) return;
    const check = this.checks[id], limits = check?.limits || {}, selected = this.selection(id);
    const canManual = !!limits.focus_absolute && !(limits.focus_absolute.flags & 4);
    const canAuto = !!limits.focus_auto && !(limits.focus_auto.flags & 4);
    const locked = this.locked || !!this.pending || row.info.type !== 'rgb';
    row.manual.checked = selected.mode === 'manual'; row.auto.checked = selected.mode === 'auto';
    row.manual.disabled = locked || !canManual || row.auto.checked;
    row.auto.disabled = locked || !canAuto || row.manual.checked;
    row.manual.parentElement?.classList.toggle('disabled', row.manual.disabled);
    row.auto.parentElement?.classList.toggle('disabled', row.auto.disabled);
    const range = limits.focus_absolute;
    if(range) {
      row.slider.min = range.min; row.slider.max = range.max; row.slider.step = Math.max(1, range.step);
      const number = selected.value ?? check?.values?.focus_absolute ?? range.default;
      row.slider.value = number;
      row.value.textContent = ` ${number}（${range.min}～${range.max}）`;
    } else row.value.textContent = ' 不可调';
    row.slider.disabled = locked || !canManual || !row.manual.checked;
    const waitingChoice = this.drafts.has(id) && !selected.mode;
    row.note.textContent = row.info.type !== 'rgb' ? '此 RGBD 设备尚未接通对焦调节'
      : !check ? '等待相机打开并读取对焦能力'
      : !canManual && !canAuto ? '设备不提供可调对焦控制'
      : this.pending === id ? '正在应用并保存…'
      : this.messages.get(id) || (waitingChoice ? '请选择定焦或自动对焦'
        : '取消当前勾选后可切换模式；松开滑块即应用并保存。采集期间锁定。');
    row.retry.hidden = !this.drafts.has(id) || !selected.mode;
    row.retry.disabled = locked;
  }
  choose(id, mode, checked) {
    const row = this.rows.get(id), choice = mode === 'manual' ? row.manual : row.auto;
    if(choice.disabled) return;
    const previous = this.selection(id), range = this.checks[id]?.limits?.focus_absolute;
    let value = previous.value ?? this.checks[id]?.values?.focus_absolute ?? range?.default ?? null;
    // 自动模式不使用保存的手动值；第一次选定焦才选一个实际支持的起始位置。
    if(range && (value < range.min || value > range.max || (value-range.min) % Math.max(1,range.step))) value = range.default;
    this.drafts.set(id, {mode:checked ? mode : null, value}); this.messages.delete(id);
    this.render(id);
    if(checked) void this.apply(id);
  }
  async apply(id) {
    if(this.locked || this.pending) return;
    const setting = this.drafts.get(id); if(!setting?.mode) return;
    this.pending = id;
    for(const key of this.rows.keys()) this.render(key);
    try {
      const result = await this.post(`/api/camera/${encodeURIComponent(id)}/focus`, {...setting});
      this.settings = {...this.settings, [id]:result.setting};
      this.checks = {...this.checks, [id]:result.focus};
      this.drafts.delete(id); this.messages.set(id, '已应用并保存，预览与录制共用此设置');
    } catch(error) { this.messages.set(id, error.message); }
    finally {
      this.pending = null;
      await this.refresh();
      for(const key of this.rows.keys()) this.render(key);
    }
  }
  assertReady() {
    if(this.pending || this.drafts.size) throw new Error('请先完成相机对焦选择并应用保存，再开始采集或保存其他配置。');
  }
}
globalThis.NeroCameraFocus = NeroCameraFocus;
