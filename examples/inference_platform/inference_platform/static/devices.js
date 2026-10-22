"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  if (!document.querySelector('meta[name="platform-ui-version"][content="5"]')) {
    const notice = document.createElement("p");
    notice.className = "config-issue";
    notice.textContent = "设备界面已更新。请在启动终端按 Ctrl+C，重新启动平台后刷新页面。";
    document.body.prepend(notice);
    return;
  }
  const dialog = $("device-dialog");
  const jogDialog = $("jog-dialog");
  const disableDialog = $("disable-dialog");
  const measureDialog = $("measure-dialog");
  const phases = {closed: "未连接", connecting: "连接中", ready: "已连接", waiting: "等待反馈",
    error: "连接异常", closing: "断开中", release_failed: "释放失败", jogging: "微动中"};
  const roleNames = {left_leader: "左主臂", right_leader: "右主臂",
    left_follower: "左从臂", right_follower: "右从臂"};
  let state = {devices: [], revision: "", active: false, scanning: false, confirmed: false};
  let online = false;
  let busy = false;
  let inferenceBusy = false;
  let listKey = "";
  let localJob = null;
  let pendingJog = null;
  let startingJog = false;
  let pendingDisable = null;
  let pendingMeasure = null;
  let mutation = 0;
  let serviceError = "";
  let configError = "";
  const cards = new Map();
  const frames = new Map();
  const jogErrors = new Map();
  const autoOpened = new Set();
  const currentConfig = () => ({
    robot: $("robot").value, arms: $("arms").value,
    space: document.querySelector('[data-space][aria-pressed="true"]').dataset.space,
  });
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function error(message) {
    $("device-error").textContent = message;
    $("device-error").hidden = !message;
    $("device-confirm-error").textContent = message;
    $("device-confirm-error").hidden = !message;
  }
  function accept(next) {
    if (state.revision !== next.revision) $("acknowledge-devices").checked = false;
    state = next;
    if (localJob && (!state.jog?.active || state.jog.id !== localJob)) localJob = null;
    online = next.schema_version === 4;
    serviceError = online ? "" : "设备服务版本较旧，请在启动终端按 Ctrl+C，重新启动平台后刷新页面。";
    if (serviceError) error(serviceError);
    render();
    window.dispatchEvent(new CustomEvent("inference-devices", {detail: {...state, online}}));
    autoOpenCameras();
  }
  function autoOpenCameras() {
    if (busy || !online || state.scanning || !state.config) return;
    const cameraIds = new Set(state.devices.filter((item) => item.kind === "camera").map((item) => item.id));
    for (const id of [...autoOpened]) if (!cameraIds.has(id)) autoOpened.delete(id);
    for (const item of state.devices) {
      if (item.kind !== "camera" || autoOpened.has(item.id)) continue;
      if (item.phase === "closed" || item.phase === "release_failed") {
        autoOpened.add(item.id);
        operation("open", {id: item.id, config: state.config});
        return;
      }
    }
  }
  async function operation(action, extra = {}) {
    if (busy) return;
    busy = true;
    const serial = ++mutation;
    if (action === "jog") startingJog = true;
    if (action === "jog") jogErrors.delete(extra.id);
    if (action === "configure") configError = "";
    error("");
    render();
    try {
      const response = await fetch(`/api/devices/${action}`, {
        method: "POST", headers: {"Content-Type": "application/json", "X-Platform-Request": "1"},
        body: JSON.stringify({revision: state.revision, ...extra}),
        signal: AbortSignal.timeout(10000),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "操作失败");
      if (serial === mutation) {
        if (action === "jog") localJob = data.jog?.id || null;
        accept(data);
      }
    } catch (failure) {
      if (serial === mutation) {
        error(failure.message);
        if (action === "jog") jogErrors.set(extra.id, failure.message);
        if (action === "configure") configError = failure.message;
      }
    } finally {
      if (action === "jog") startingJog = false;
      busy = false;
      render();
    }
  }
  window.addEventListener("inference-busy", (event) => { inferenceBusy = event.detail; render(); });
  window.addEventListener("stop-all-motion", () => { if (state.jog?.active || startingJog) stopJog(); });
  async function stopJog() {
    localJob = null;
    const serial = ++mutation;
    try {
      const response = await fetch("/api/devices/jog-stop", {
        method: "POST", headers: {"Content-Type": "application/json", "X-Platform-Request": "1"},
        body: "{}", keepalive: true, signal: AbortSignal.timeout(2000),
      });
      if (!response.ok) throw new Error("停止请求未送达");
      const data = await response.json();
      if (serial === mutation) accept(data);
    } catch (_) {
      error("停止请求未送达，已停止页面心跳；请检查实物，必要时使用实体急停。");
    }
  }
  async function heartbeat() {
    const job = localJob;
    if (job && !document.hidden && dialog.open) {
      try {
        await fetch("/api/devices/jog-heartbeat", {
          method: "POST", headers: {"Content-Type": "application/json", "X-Platform-Request": "1"},
          body: JSON.stringify({job}), signal: AbortSignal.timeout(800),
        });
      } catch (_) { /* Missing renewals cancel the motion on the server. */ }
    }
    setTimeout(heartbeat, 300);
  }
  async function startMeasure() {
    if (!pendingMeasure) return;
    $("measure-start").hidden = true;
    $("measure-stop").hidden = false;
    $("measure-status").textContent = "正在使能并准备拖动…";
    try {
      await operation("measure", {id: pendingMeasure.id});
      $("measure-status").textContent = "拖动中…请移动各关节到两端（不要硬顶机械限位），完成后点停止测量并保存。";
      pollMeasure();
    } catch (_) {
      $("measure-start").hidden = false;
      $("measure-stop").hidden = true;
    }
  }
  async function stopMeasure() {
    if (!pendingMeasure) return;
    $("measure-status").textContent = "正在停止并保存…";
    try {
      await operation("measure-stop", {id: pendingMeasure.id});
    } catch (_) {}
    pollMeasure();
  }
  async function pollMeasure() {
    if (!pendingMeasure) return;
    const dialogNode = measureDialog;
    if (!dialogNode.open) { pendingMeasure = null; return; }
    try {
      const response = await fetch("/api/devices");
      const data = await response.json();
      const device = (data.devices || []).find((entry) => entry.id === pendingMeasure.id);
      const measure = device?.measure;
      if (measure?.report?.ok) {
        $("measure-status").textContent = `已保存：${measure.report.label}（${measure.report.samples} 样本）→ ${measure.report.path}`;
        $("measure-stop").hidden = true;
        $("measure-start").hidden = false;
        pendingMeasure = null;
        return;
      }
      if (measure?.report?.error) {
        $("measure-status").textContent = "测量失败：" + measure.report.error;
        $("measure-stop").hidden = true;
        $("measure-start").hidden = false;
        pendingMeasure = null;
        return;
      }
    } catch (_) {}
    if (pendingMeasure) setTimeout(pollMeasure, 1000);
  }
  function buildCards() {
    const key = JSON.stringify([state.config, state.scanning, state.devices.map((item) => [item.id, item.kind])]);
    if (key === listKey) return;
    listKey = key;
    cards.clear();
    $("arm-list").replaceChildren();
    $("camera-list").replaceChildren();
    let cameraNumber = 0;
    for (const item of state.devices) {
      const card = element("article", "device-card");
      const title = element("strong", "device-title", item.kind === "arm" ? item.id
        : `相机 ${++cameraNumber} · ${item.type === "rgbd" ? "RGBD" : "USB"}`);
      title.title = item.id;
      const badge = element("span", "small-tag");
      const heading = element("div", "device-card-heading");
      heading.append(title, badge);
      const description = element("p", "device-identity", item.kind === "arm"
        ? `USB ${item.identity?.serial || "无序列号"} · ${item.up ? "CAN 已启动" : "CAN 未启动"}`
        : `${item.backend} · ${item.serial || item.path || item.id}`);
      const feedback = element("p", "device-feedback");
      const preview = element("img", "device-thumbnail");
      preview.alt = `${item.id} 预览`;
      preview.hidden = true;
      const placeholder = element("p", "thumbnail-placeholder", "打开预览后核对相机视角");
      const label = element("label", "role-label", "设备用途");
      const role = element("select");
      role.setAttribute("aria-label", `${item.id} 设备用途`);
      role.add(new Option("未分配", ""));
      const roles = item.kind === "camera" ? ["head", "left", "right"]
        : Object.keys(roleNames);
      for (const value of roles) {
        role.add(new Option(item.kind === "camera"
          ? ({head: "头部相机", left: "左腕相机", right: "右腕相机"})[value] : roleNames[value], value));
      }
      role.addEventListener("change", () => operation("assign", {id: item.id, role: role.value}));
      label.append(role);
      const connect = element("button", "button compact");
      connect.type = "button";
      connect.addEventListener("click", () => {
        const live = state.devices.find((entry) => entry.id === item.id);
        const connected = ["connecting", "ready", "waiting", "closing", "error", "jogging"].includes(live.phase);
        operation(connected ? "disconnect" : "open", {id: item.id, config: currentConfig()});
      });
      const message = element("p", "device-card-message");
      const jogFeedback = element("p", "jog-feedback");
      jogFeedback.setAttribute("role", "status");
      jogFeedback.setAttribute("aria-live", "polite");
      jogFeedback.hidden = true;
      const actions = element("div", "device-actions");
      actions.append(connect);
      let jog = null;
      let disable = null;
      let measure = null;
      if (item.kind === "arm") {
        jog = element("button", "button compact", "微动测试");
        jog.type = "button";
        jog.setAttribute("aria-label", `${item.id} 微动测试`);
        jog.addEventListener("click", () => {
          pendingJog = {id: item.id, revision: state.revision};
          const joint = state.config.robot === "nero" ? "J7" : "J6";
          const live = state.devices.find((entry) => entry.id === item.id);
          $("jog-target").textContent = `${state.config.robot.toUpperCase()} · ${item.id} · `
            + `${roleNames[live.role] || "用途未分配"}：${joint} 向限位中间方向微动 5°，然后返回测试前位置。`;
          render();
          jogDialog.showModal();
        });
        actions.append(jog);
        measure = element("button", "button compact", "关节限位测量");
        measure.type = "button";
        measure.setAttribute("aria-label", `${item.id} 关节限位测量`);
        measure.addEventListener("click", () => {
          pendingMeasure = {id: item.id, revision: state.revision};
          const live = state.devices.find((entry) => entry.id === item.id);
          $("measure-target").textContent = `${state.config.robot.toUpperCase()} · ${item.id} · `
            + `${roleNames[live.role] || "用途未分配"}`;
          $("measure-status").textContent = "";
          $("measure-start").hidden = false;
          $("measure-stop").hidden = true;
          measureDialog.showModal();
        });
        actions.append(measure);
        disable = element("button", "button compact stop", "失能");
        disable.type = "button";
        disable.setAttribute("aria-label", `${item.id} 失能`);
        disable.addEventListener("click", () => {
          pendingDisable = {id: item.id, revision: state.revision};
          const live = state.devices.find((entry) => entry.id === item.id);
          $("disable-target").textContent = `${state.config.robot.toUpperCase()} · ${item.id} · `
            + `${roleNames[live.role] || "用途未分配"}：断开关节电机与夹爪动力。`;
          render();
          disableDialog.showModal();
        });
        actions.append(disable);
      }
      card.append(heading, description);
      if (item.kind === "camera") card.append(preview, placeholder);
      card.append(feedback, label, actions, message);
      if (item.kind === "arm") card.append(jogFeedback);
      $(item.kind === "arm" ? "arm-list" : "camera-list").append(card);
      cards.set(item.id, {badge, feedback, role, connect, jog, disable, measure, jogFeedback, message, preview, placeholder});
    }
    for (const kind of ["arm", "camera"]) {
      if (!state.devices.some((item) => item.kind === kind)) {
        $(`${kind}-list`).append(element("p", "field-note", state.scanning ? "正在检测…" : "尚未检测到设备"));
      }
    }
  }
  function assigned(kind, role) {
    return state.devices.find((item) => item.kind === kind && item.role === role);
  }
  function deviceLabel(kind, role) {
    if (!online) return "状态未知";
    const item = assigned(kind, role);
    return item ? `${item.kind === "arm" ? `${item.id} · ` : ""}${phases[item.phase]}` : "未分配";
  }
  function renderArmDisableList(moving, needsScan) {
    const container = $("arm-disable-list");
    if (!container) return;
    const arms = state.devices.filter((item) => item.kind === "arm");
    container.replaceChildren();
    if (!arms.length) {
      container.append(element("p", "field-note", "尚未检测到机械臂。"));
      return;
    }
    for (const item of arms) {
      const connected = ["ready", "waiting", "error"].includes(item.phase);
      const btn = element("button", "button full", `${item.id.toUpperCase()} 机械臂失能`);
      btn.type = "button";
      btn.disabled = moving || busy || !online || needsScan || state.scanning || !connected;
      btn.title = item.role
        ? `${roleNames[item.role]} · ${connected ? "点击断开该臂动力" : "先连接该机械臂"}`
        : `${connected ? "点击断开该臂动力" : "先连接该机械臂"}`;
      btn.addEventListener("click", () => {
        pendingDisable = {id: item.id, revision: state.revision};
        const live = state.devices.find((entry) => entry.id === item.id);
        $("disable-target").textContent = `${state.config.robot.toUpperCase()} · ${item.id} · `
          + `${roleNames[live.role] || "用途未分配"}：断开关节电机与夹爪动力。`;
        render();
        disableDialog.showModal();
      });
      container.append(btn);
    }
  }

  function render() {
    const jogging = Boolean(state.jog?.active || startingJog);
    const moving = jogging || inferenceBusy || state.motion_active;
    if (state.config && !busy) {
      const local = currentConfig();
      if (Object.keys(local).some((key) => local[key] !== state.config[key])) {
        window.dispatchEvent(new CustomEvent("device-config", {detail: state.config}));
      }
    }
    for (const node of [$("robot"), $("arms"), ...document.querySelectorAll("[data-space]")]) {
      node.disabled = busy || moving || state.scanning || !online;
    }
    $("device-config-message").textContent = configError || serviceError || (moving
      ? "执行中，停止并停稳后可修改配置。"
      : "连接后仍可修改；单/双臂、关节/EEF 保留连接。切换型号仅需重连机械臂，用途分配保留。");
    $("device-config-message").className = configError || serviceError ? "config-issue" : "field-note";
    const status = !online ? "服务连接中断" : moving ? "运行中" : state.confirmed ? "用途已确认"
      : state.active ? "待确认用途" : "未连接";
    $("device-status").textContent = status;
    $("runtime-devices").textContent = status;
    $("connection-summary").textContent = status;
    $("device-message").textContent = online
      ? state.jog && !state.confirmed ? `${state.jog.device} · ${state.jog.message}` : state.message
      : serviceError || "无法读取设备状态，请检查平台终端";
    $("device-message").dataset.phase = !state.confirmed ? state.jog?.phase || "" : "";
    $("jog-status").hidden = !state.jog;
    $("jog-status").textContent = state.jog ? `${state.jog.device} · ${state.jog.message}` : "";
    $("jog-status").dataset.phase = state.jog?.phase || "";
    $("stop-jog").disabled = !jogging;
    $("start-jog").disabled = busy || !online || moving || state.jog?.recovery
      || pendingJog?.revision !== state.revision;
    for (const role of Object.keys(roleNames)) {
      $(`arm-${role.replace("_", "-")}-status`).textContent = deviceLabel("arm", role);
    }
    $("arm-role-hint").textContent = (currentConfig().arms === "single"
      ? "单臂推理：选择左从臂或右从臂其中一个。"
      : "双臂推理：分配左从臂和右从臂。")
      + "主臂可选；用途记录实物身份，不会自动切换硬件主从模式。";
    for (const role of ["head", "left", "right"]) {
      $(`camera-${role}-status`).textContent = deviceLabel("camera", role);
    }
    const cameras = state.devices.filter((item) => item.kind === "camera" && item.phase === "ready" && item.role);
    $("preview-status").textContent = online && cameras.length ? `${cameras.length} 路实时预览 · RGB` : "等待相机连接";
    $("disconnect-all").disabled = moving || busy || !online || !state.active || state.scanning;
    $("dialog-disconnect").disabled = $("disconnect-all").disabled;
    $("scan-devices").disabled = busy || !online || state.active || state.scanning;
    $("scan-devices").textContent = state.scanning ? "检测中…" : "重新检测设备";
    const needsScan = state.config && Object.keys(currentConfig()).some((key) => currentConfig()[key] !== state.config[key]);
    const selected = state.devices.filter(item => item.role);
    const followers = selected.filter(item => item.kind === "arm" && item.role.endsWith("_follower"));
    const unusedArms = state.devices.filter(item => item.kind === "arm" && !item.role).length;
    const cameraNames = {head: "头部相机", left: "左腕相机", right: "右腕相机"};
    const selection = selected.map(item => item.kind === "arm" ? `${roleNames[item.role]} ${item.id}` : cameraNames[item.role]);
    $("current-device-selection").textContent = (selection.length ? `本次分配：${selection.join("、")}。` : "请先分配本次使用的从臂与相机。")
      + (unusedArms ? `另 ${unusedArms} 条机械臂未分配，正常忽略。` : "")
      + "模型只控制从臂。";
    const followerCountReady = currentConfig().arms === "single" ? followers.length === 1 : followers.length === 2;
    const mappingReady = followerCountReady && cameras.length > 0 && selected.every(item => item.phase === "ready");
    $("confirm-current-devices").disabled = moving || busy || !online || needsScan || state.scanning || state.confirmed || !mappingReady;
    $("confirm-current-devices").textContent = state.confirmed ? "本次设备用途已确认" : "确认本次设备用途";
    $("scan-hint").textContent = state.active ? "修改配置无需重新检测；如需重新枚举插拔的设备，请先断开全部设备。"
      : needsScan ? "配置已变化，请重新检测设备。" : "检测只枚举设备；点击连接后才打开通信。";
    $("confirm-devices").disabled = moving || busy || !online || needsScan || state.scanning || !$("acknowledge-devices").checked;
    $("confirmation-message").textContent = state.confirmed
      ? "本次设备用途已确认。可以关闭面板，在主界面查看画面。"
      : "只确认本次分配的设备；未分配设备正常忽略。单臂需一个从臂，双臂需左右从臂，主臂可选，至少一台相机。";
    buildCards();
    for (const item of state.devices) {
      const card = cards.get(item.id);
      card.badge.textContent = phases[item.phase];
      card.badge.dataset.phase = item.phase;
      card.message.textContent = item.message;
      card.role.value = item.role;
      card.role.disabled = moving || busy || !online || needsScan || state.scanning;
      card.feedback.textContent = item.kind === "arm" && item.sample.joint_degrees
        ? `${item.sample.source} · ${item.sample.joint_degrees.map((value, i) => `J${i + 1} ${value}°`).join(" / ")}`
          + (item.sample.gripper_m === null ? "" : ` · 夹爪 ${(item.sample.gripper_m * 1000).toFixed(1)} mm`)
        : "";
      const connected = ["connecting", "ready", "waiting", "closing", "error", "jogging"].includes(item.phase);
      card.connect.textContent = connected ? "断开连接" : item.kind === "arm" ? "连接并读取反馈" : "打开预览";
      card.connect.disabled = moving || busy || !online || needsScan || state.scanning || item.phase === "closing"
        || item.phase === "release_failed" || (item.kind === "arm" && !item.up);
      if (moving && !connected) card.connect.disabled = true;
      if (card.jog) {
        card.jog.disabled = moving || busy || !online || needsScan || state.scanning
          || item.phase !== "ready" || Boolean(state.jog?.recovery);
        card.jog.title = state.jog?.recovery ? "请检查实物并处理故障后重启平台"
          : item.phase !== "ready" ? "先连接并等待普通关节和夹爪反馈正常" : "腕关节微动 5° 后返回";
        const job = state.jog?.device === item.id ? state.jog : null;
        const requestError = jogErrors.get(item.id);
        card.jogFeedback.textContent = requestError || (startingJog && pendingJog?.id === item.id
          ? "正在提交微动请求…" : job?.message || "");
        card.jogFeedback.hidden = !card.jogFeedback.textContent;
        card.jogFeedback.dataset.phase = requestError ? "error" : job?.phase || "";
        card.jog.textContent = job?.active ? "微动中…" : "微动测试";
      }
      if (card.measure) {
        card.measure.disabled = moving || busy || !online || needsScan || state.scanning
          || item.phase !== "ready" || !item.role;
        card.measure.title = item.phase !== "ready" ? "先连接并等待普通关节和夹爪反馈正常"
          : !item.role ? "请先分配用途（左/右主从臂）再测量" : "使能并切到拖动模式，测量各关节限位";
        card.measure.textContent = item.measure?.measuring ? "测量中…" : "关节限位测量";
      }
      if (card.disable) {
        card.disable.disabled = moving || busy || !online || needsScan || state.scanning
          || !["ready", "waiting", "error"].includes(item.phase);
        card.disable.title = ["ready", "waiting", "error"].includes(item.phase)
          ? "断开该机械臂的关节与夹爪动力" : "先连接该机械臂再失能";
      }
    }
    const results = Object.values(state.detection || {});
    $("detection-details").hidden = !results.length;
    $("detection-message").textContent = results.map((item) => `${item.label}：${item.message}`).join("；");
    renderArmDisableList(moving, needsScan);
    paintFrames();
  }
  function putImage(image, placeholder, entry) {
    const ready = online && entry && Date.now() - entry.time < 600 && entry.revision === state.revision;
    image.hidden = !ready;
    placeholder.hidden = Boolean(ready);
    if (ready && image.getAttribute("src") !== entry.url) image.src = entry.url;
    if (!ready) image.removeAttribute("src");
  }
  function paintFrames() {
    for (const item of state.devices.filter((entry) => entry.kind === "camera")) {
      const card = cards.get(item.id);
      putImage(card.preview, card.placeholder, item.phase === "ready" ? frames.get(item.id) : null);
    }
    for (const figure of document.querySelectorAll("[data-camera-role]")) {
      const item = assigned("camera", figure.dataset.cameraRole);
      const frame = item?.phase === "ready" ? frames.get(item.id) : null;
      putImage(figure.querySelector("img"), figure.querySelector(".camera-empty"), frame);
      figure.querySelector(".frame-status").textContent = !online ? "状态未知" : item?.phase === "ready"
        && frame && Date.now() - frame.time < 600 ? `${item.sample.width} × ${item.sample.height} · 实时`
        : item ? phases[item.phase] : "未分配";
    }
  }
  async function pollStatus() {
    // Do not let an older GET overwrite a completed mutation.
    if (!busy) {
      const revision = state.revision;
      try {
        const response = await fetch("/api/devices", {signal: AbortSignal.timeout(3000)});
        if (!response.ok) throw new Error("状态读取失败");
        const next = await response.json();
        if (!busy && state.revision === revision) accept(next);
      } catch (_) {
        if (!busy) { online = false; $("acknowledge-devices").checked = false; render(); }
      }
    }
    setTimeout(pollStatus, 500);
  }
  async function pollFrames() {
    const revision = state.revision;
    const cameras = online && !document.hidden ? state.devices.filter((item) =>
      item.kind === "camera" && item.phase === "ready" && (dialog.open || item.role)) : [];
    await Promise.all(cameras.map(async (item) => {
      try {
        const query = new URLSearchParams({id: item.id, revision});
        const response = await fetch(`/api/devices/frame?${query}`, {signal: AbortSignal.timeout(1500)});
        if (response.status !== 200) throw new Error("画面不可用");
        const blob = await response.blob();
        if (state.revision !== revision) return;
        const previous = frames.get(item.id);
        frames.set(item.id, {url: URL.createObjectURL(blob), time: Date.now(), revision});
        paintFrames();
        if (previous) URL.revokeObjectURL(previous.url);
      } catch (_) {
        const previous = frames.get(item.id);
        if (previous) URL.revokeObjectURL(previous.url);
        frames.delete(item.id);
      }
    }));
    for (const [id, value] of frames) {
      if (!cameras.some((item) => item.id === id)) {
        URL.revokeObjectURL(value.url);
        frames.delete(id);
      }
    }
    paintFrames();
    setTimeout(pollFrames, 120);
  }
  for (const id of ["manage-arms", "manage-cameras"]) $(id).addEventListener("click", () => {
    dialog.showModal();
    if (state.devices.length) $(id === "manage-arms" ? "arm-list-heading" : "camera-list-heading")
      .scrollIntoView({block: "start"});
  });
  $("close-devices").addEventListener("click", () => {
    if (localJob || startingJog) stopJog();
    dialog.close();
  });
  $("cancel-jog").addEventListener("click", () => jogDialog.close());
  $("start-jog").addEventListener("click", () => {
    if (!pendingJog || pendingJog.revision !== state.revision) return;
    jogDialog.close();
    operation("jog", {...pendingJog, acknowledged: true});
  });
  $("cancel-disable").addEventListener("click", () => disableDialog.close());
  $("start-disable").addEventListener("click", () => {
    if (!pendingDisable || pendingDisable.revision !== state.revision) return;
    disableDialog.close();
    operation("disable", {...pendingDisable, acknowledged: true});
  });
  $("cancel-measure").addEventListener("click", () => { pendingMeasure = null; measureDialog.close(); });
  $("measure-start").addEventListener("click", startMeasure);
  $("measure-stop").addEventListener("click", stopMeasure);
  jogDialog.addEventListener("close", () => render());
  disableDialog.addEventListener("close", () => render());
  measureDialog.addEventListener("close", () => render());
  dialog.addEventListener("cancel", (event) => {
    if (state.jog?.active || startingJog) { event.preventDefault(); stopJog(); }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && (state.jog?.active || startingJog)) {
      event.preventDefault();
      stopJog();
    }
  });
  for (const id of ["stop-jog"]) $(id).addEventListener("click", stopJog);
  window.addEventListener("pagehide", () => { if (localJob || startingJog) stopJog(); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && (localJob || startingJog)) stopJog();
  });
  $("scan-devices").addEventListener("click", () => operation("scan", {config: currentConfig()}));
  window.addEventListener("platform-config-change", (event) => {
    if (!online || busy || state.scanning || state.jog?.active) { render(); return; }
    operation("configure", {config: event.detail});
  });
  $("disconnect-all").addEventListener("click", () => operation("disconnect"));
  $("dialog-disconnect").addEventListener("click", () => operation("disconnect"));
  $("acknowledge-devices").addEventListener("change", render);
  $("confirm-devices").addEventListener("click", () => operation("confirm", {acknowledged: $("acknowledge-devices").checked}));
  $("confirm-current-devices").addEventListener("click", () => operation("confirm", {acknowledged: true}));
  pollStatus();
  pollFrames();
  heartbeat();
  setInterval(paintFrames, 250);
})();
