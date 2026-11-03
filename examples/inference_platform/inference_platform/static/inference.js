"use strict";
(() => {
  const $ = id => document.getElementById(id);
  let state = {model_state: "empty", revision: "", job: null};
  let devices = {confirmed: false, online: false};
  let online = false, pending = false, localJob = null, serial = 0, notice = "", lastBusy = null;
  const settings = () => window.platformSettings();
  const labels = {empty: "未加载", loading: "加载中", ready: "已加载", error: "加载异常"};
  function render() {
    const running = Boolean(state.job?.active);
    const busy = pending || running || state.model_state === "loading";
    const config = settings();
    const match = state.checkpoint === config.path && ["robot", "arms", "space"].every(key => state.config?.[key] === config[key]);
    $("model-tag").textContent = labels[state.model_state] || "状态未知";
    $("runtime-model").textContent = online ? `${labels[state.model_state]}${state.device ? ` · ${state.device.toUpperCase()}` : ""}${state.rtc?.enabled ? " · RTC" : ""}` : "服务离线";
    $("runtime-latency").textContent = state.inference_ms == null ? "— ms" : `${state.inference_ms} ms`;
    $("runtime-chunks").textContent = state.chunks || 0;
    const seconds = Math.floor(state.elapsed || 0);
    $("runtime-elapsed").textContent = `${Math.floor(seconds/60).toString().padStart(2,"0")}:${(seconds%60).toString().padStart(2,"0")}`;
    $("record-description").textContent = running && state.record_path ? `正在保存 · ${state.record_frames || 0} 帧` : config.record ? "执行时保存视频、观测与动作；停止后自动完成保存。" : "关闭时不保存视频、观测和动作数据。";
    const executionHint = state.model_state === "ready" && !devices.confirmed && !running
      ? "模型已加载；执行前在设备连接区确认本次用途。闲置机械臂不用分配。" : "";
    $("model-message").textContent = state.model_state === "ready" && !match ? "设备配置或 checkpoint 已改变，请重新加载。" : executionHint || state.message || "等待加载";
    const message = notice || (running ? state.job.message : executionHint || state.message) || "等待模型加载";
    $("inference-message").textContent = message;
    $("inference-message").title = message;
    const modelLabel = config.model === "fastwam" ? "FastWAM" : "π0.5";
    $("load-model").disabled = !online || busy || devices.jog?.active || devices.motion_active || !config.path || !["pi05", "fastwam"].includes(config.model);
    $("load-model").textContent = state.model_state === "loading" ? `正在加载 ${modelLabel}…` : state.model_state === "ready" ? `重新加载 ${modelLabel}` : `加载 ${modelLabel}`;
    $("unload-model").disabled = !online || busy || state.model_state === "empty";
    const enabled = online && devices.online && devices.confirmed && !devices.jog?.active && !busy && !state.job?.recovery;
    $("home-motion").disabled = !enabled;
    $("single-motion").disabled = $("continuous-motion").disabled = !enabled || state.model_state !== "ready" || !match || !config.task.trim();
    $("stop-motion").disabled = !(pending || running || state.model_state === "loading" || devices.jog?.active);
    for (const id of ["model", "checkpoint-source", "checkpoint", "choose-checkpoint", "refresh-checkpoints", "task", "record", "rtc-enabled", "dataset-name", "output-dir"]) {
      // Preserve catalog's disabled state when idle; only remove our own lock.
      const node = $(id);
      if (busy && !node.dataset.executionLock) {
        node.dataset.executionLock = node.disabled ? "disabled" : "enabled";
        node.disabled = true;
      } else if (!busy && node.dataset.executionLock) {
        node.disabled = node.dataset.executionLock === "disabled";
        delete node.dataset.executionLock;
      } else if (busy) node.disabled = true;
    }
    if (config.model !== "pi05") $("rtc-enabled").disabled = true;
    $("record-result").hidden = !state.record_path;
    $("record-result").textContent = state.record_path ? `${running ? "本次保存目录" : "本次数据"}：${state.record_path}（${state.record_frames || 0} 帧）` : "";
    if (lastBusy !== busy) {
      lastBusy = busy;
      window.dispatchEvent(new CustomEvent("inference-busy", {detail: busy}));
    }
  }
  function accept(value) {
    state = value;
    online = value.version === 1;
    if (localJob && (!state.job?.active || state.job.id !== localJob)) localJob = null;
    render();
  }
  async function request(action, data = {}, keepalive = false) {
    const response = await fetch(`/api/inference/${action}`, {
      method: "POST", headers: {"Content-Type":"application/json", "X-Platform-Request":"1"},
      body: JSON.stringify(data), keepalive, signal: AbortSignal.timeout(10000),
    });
    if (response.status === 404) throw new Error("推理服务已更新，请重启平台后刷新页面。");
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "操作失败");
    return result;
  }
  async function operation(action) {
    if (pending) return;
    pending = true;
    notice = "";
    const version = ++serial;
    render();
    try {
      const result = await request(action, {revision: state.revision, settings: settings()});
      if (version === serial) {
        if (["single", "continuous", "home"].includes(action)) localJob = result.job?.id || null;
        accept(result);
      }
    } catch (error) { if (version === serial) notice = error.message; }
    finally { pending = false; render(); }
  }
  async function stop() {
    localJob = null;
    const version = ++serial;
    window.dispatchEvent(new CustomEvent("stop-all-motion"));
    try {
      const result = await request("stop", {}, true);
      if (version === serial) accept(result);
    } catch (_) {
      notice = "停止请求未送达，已停止心跳；请检查实物，必要时使用实体急停。";
    }
    render();
  }
  async function poll() {
    const version = serial;
    try {
      const response = await fetch("/api/inference", {signal: AbortSignal.timeout(2000)});
      if (!response.ok) throw new Error("服务不可用");
      const value = await response.json();
      if (version === serial && !pending) accept(value);
    } catch (_) { online = false; render(); }
    setTimeout(poll, 350);
  }
  async function heartbeat() {
    if (localJob && !document.hidden) {
      try { await request("heartbeat", {job: localJob}); } catch (_) { /* Lease expiry stops hardware. */ }
    }
    setTimeout(heartbeat, 300);
  }
  for (const [id, action] of Object.entries({"load-model":"load", "unload-model":"unload", "single-motion":"single", "continuous-motion":"continuous", "home-motion":"home"})) {
    $(id).addEventListener("click", () => operation(action));
  }
  $("stop-motion").addEventListener("click", stop);
  window.addEventListener("inference-config-change", render);
  window.addEventListener("inference-devices", event => {devices = event.detail; render();});
  document.addEventListener("keydown", event => {if (event.key === "Escape") stop();});
  window.addEventListener("pagehide", () => {if (localJob || pending) stop();});
  document.addEventListener("visibilitychange", () => {if (document.hidden && (localJob || pending)) stop();});
  poll(); heartbeat();
})();
