"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  if (!document.querySelector('meta[name="checkpoint-ui-version"][content="2"]')) {
    $("checkpoint-detail").textContent = "模型选择界面已更新，请重启平台后刷新页面。";
    return;
  }
  const storageKey = "nero-piper-inference-config-v1";
  const defaults = {
    version: 2, robot: "nero", arms: "dual", space: "eef", model: "pi05",
    checkpointSource: "outputs",
    checkpoint: "", customPath: "", task: "", record: false, datasetName: "", outputDir: "", duration: 60,
  };
  let config = {...defaults};
  let checkpoints = [];
  let catalogError = "";
  let dirty = false;
  let storageError = "";
  let conflicts = [];
  let catalogLoaded = false;
  let scanning = false;
  let manualCheckpoint = null;
  let selectionBusy = false;
  let selectionMessage = "";
  let selectionError = false;

  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if ([1, 2].includes(saved?.version)) {
      for (const key of Object.keys(defaults)) {
        if (key !== "version" && typeof saved[key] === typeof defaults[key]) config[key] = saved[key];
      }
      if (saved.version === 1 && saved.checkpoint === "custom") {
        config.checkpointSource = "manual";
        config.checkpoint = "";
      }
    }
  } catch (_) {
    storageError = "浏览器配置不可用，本次可继续选择。";
  }
  for (const [key, values] of Object.entries({
    robot: ["nero", "piper"], arms: ["single", "dual"], space: ["joint", "eef"], model: ["pi05"],
    checkpointSource: ["outputs", "manual"],
  })) {
    if (!values.includes(config[key])) config[key] = defaults[key];
  }
  const names = {nero: "NERO", piper: "Piper", single: "单臂", dual: "双臂", joint: "关节空间", eef: "EEF"};
  const option = (text, value) => new Option(text, value);

  function populateCheckpoints() {
    const select = $("checkpoint");
    const available = checkpoints.filter((entry) => entry.model === config.model);
    select.replaceChildren();
    for (const item of available) {
      const space = item.space ? names[item.space] : "空间待核对";
      select.add(option(`${item.name} · ${space}`, item.path));
    }
    if (!available.length) {
      select.add(option(catalogLoaded ? "outputs 中没有此模型的 checkpoint" : "正在扫描 outputs…", ""));
    }
    if (catalogLoaded && !available.some((entry) => entry.path === config.checkpoint)) {
      if (config.checkpoint) dirty = true;
      config.checkpoint = available[0]?.path || "";
    }
    select.value = config.checkpoint;
    select.disabled = selectionBusy || scanning || !available.length;
  }

  window.platformSettings = () => ({...config, path: config.checkpointSource === "manual" ? config.customPath : config.checkpoint});

  function render() {
    const summary = `${names[config.robot]} · ${names[config.arms]} · ${names[config.space]}`;
    $("scope-summary").textContent = summary;
    $("observation-title").textContent = `${names[config.robot]} · ${names[config.arms]}`;
    $("observation-space").textContent = config.space === "eef" ? "EEF · 末端空间" : "关节空间";
    $("connection-note").textContent = config.arms === "dual"
      ? "双臂推理使用左右从臂；主臂可选。请按实物分配身份。"
      : "单主从采集训练对应单臂推理：选一个从臂；主臂可选，其余保持未分配。";
    for (const button of document.querySelectorAll("[data-space]")) {
      button.setAttribute("aria-pressed", String(button.dataset.space === config.space));
    }
    $("space-description").textContent = config.space === "eef"
      ? "模型输出末端位置、姿态和夹爪动作。"
      : `模型输出关节与夹爪动作；${names[config.robot]} 每臂 ${config.robot === "nero" ? 7 : 6} 个关节。`;
    const manual = config.checkpointSource === "manual";
    $("checkpoint-source").value = config.checkpointSource;
    $("checkpoint-source").disabled = selectionBusy;
    $("outputs-checkpoint-field").hidden = manual;
    $("manual-checkpoint-field").hidden = !manual;
    $("manual-checkpoint-path").textContent = config.customPath || "尚未选择文件夹";
    $("choose-checkpoint").disabled = selectionBusy;
    $("choose-checkpoint").textContent = selectionBusy ? "请在文件夹窗口中选择…" : "选择 checkpoint 文件夹…";
    $("checkpoint-selection-message").textContent = selectionMessage;
    $("checkpoint-selection-message").hidden = !selectionMessage;
    $("checkpoint-selection-message").className = selectionError ? "config-issue" : "field-note";
    $("model").disabled = selectionBusy;
    $("save-config").disabled = selectionBusy;
    $("refresh-checkpoints").disabled = scanning || selectionBusy;
    $("checkpoint").disabled = scanning || selectionBusy || !checkpoints.some((entry) => entry.model === config.model);
    $("record-fields").hidden = !config.record;
    $("record-description").textContent = config.record
      ? "执行时保存视频、观测与动作；停止后自动完成保存。"
      : "关闭时不保存视频、观测和动作数据。";
    $("dataset-name").placeholder = `${config.robot}/inference_pick01`;
    conflicts = [];
    const item = manual ? manualCheckpoint : checkpoints.find((entry) => entry.path === config.checkpoint);
    const notes = [];
    if (item) {
      const details = [item.space ? names[item.space] : "动作空间待核对"];
      if (item.action_dim) details.push(`${item.action_dim} 维动作`);
      if (item.cameras.length) details.push(`${item.cameras.length} 路图像`);
      if (!item.robot) details.push("机械臂型号待核对");
      $("checkpoint-detail").textContent = details.join(" · ");
      $("checkpoint-detail").title = item.path;
      $("checkpoint").title = manual ? "" : item.path;
      if (item.model !== config.model) conflicts.push("Checkpoint 与所选基座模型不一致，请重新选择。");
      if (item.space && item.space !== config.space) conflicts.push(`Checkpoint 使用${names[item.space]}，与当前选择不一致。`);
      if (item.robot && item.robot !== config.robot) conflicts.push(`Checkpoint 标注为 ${names[item.robot]}，请核对机械臂类型。`);
      if (item.arms && item.arms !== config.arms) conflicts.push(`Checkpoint 为${names[item.arms]}配置，与当前使用方式不一致。`);
      if (item.joint_count && item.joint_count !== (config.robot === "nero" ? 7 : 6)) {
        conflicts.push(`Checkpoint 每臂 ${item.joint_count} 关节，与所选机械臂不一致。`);
      }
      if (item.error) conflicts.push(`配置读取失败：${item.error}`);
      if (!item.weights_present) notes.push("未发现预期的权重文件，加载前需补齐。");
      if (!item.space) notes.push("动作空间无法从元数据确定，加载前需核对。");
    } else {
      $("checkpoint-detail").title = "";
      $("checkpoint").title = "";
      $("checkpoint-detail").textContent = manual
        ? "请选择包含 config.json 或 config.yaml 的 checkpoint 文件夹。"
        : "只读取 outputs 内 checkpoint 的配置，不加载权重。";
    }
    if (catalogError && !manual) notes.push(catalogError);
    const message = [...conflicts, ...notes].join(" ");
    $("config-issue").textContent = message;
    $("config-issue").hidden = !message;
    window.dispatchEvent(new CustomEvent("inference-config-change"));
    $("save-status").textContent = storageError || (dirty ? "配置有修改 · 尚未保存" : "配置仅保存在当前浏览器");
  }

  function syncInputs() {
    for (const id of ["robot", "arms", "model", "task"]) $(id).value = config[id];
    $("duration").value = config.duration;
    $("record").checked = config.record;
    $("dataset-name").value = config.datasetName;
    $("output-dir").value = config.outputDir;
    populateCheckpoints();
    render();
  }

  function change(key, value) {
    config[key] = value;
    if (key === "model") {
      config.checkpoint = "";
      config.customPath = "";
      manualCheckpoint = null;
      selectionMessage = "";
      populateCheckpoints();
    }
    dirty = true;
    storageError = "";
    render();
    if (["robot", "arms", "space"].includes(key)) {
      window.dispatchEvent(new CustomEvent("platform-config-change", {
        detail: {robot: config.robot, arms: config.arms, space: config.space},
      }));
    }
  }
  for (const id of ["robot", "arms", "model", "checkpoint"]) {
    $(id).addEventListener("change", () => change(id, $(id).value));
  }
  for (const [id, key] of Object.entries({
    task: "task", "dataset-name": "datasetName", "output-dir": "outputDir",
  })) $(id).addEventListener("input", () => change(key, $(id).value));
  $("duration").addEventListener("input", () => {
    const value = parseInt($("duration").value, 10);
    change("duration", Number.isFinite(value) ? value : 60);
  });
  $("record").addEventListener("change", () => change("record", $("record").checked));
  for (const button of document.querySelectorAll("[data-space]")) {
    button.addEventListener("click", () => change("space", button.dataset.space));
  }
  $("config-form").addEventListener("submit", (event) => event.preventDefault());
  $("save-config").addEventListener("click", () => {
    if (conflicts.length) {
      $("save-status").textContent = "请先处理 checkpoint 配置不匹配。";
      $("config-issue").scrollIntoView({behavior: "smooth", block: "nearest"});
      return;
    }
    if (config.checkpointSource === "manual" && !manualCheckpoint) {
      $("save-status").textContent = "请先选择有效的 checkpoint 文件夹。";
      $("choose-checkpoint").focus();
      return;
    }
    try {
      localStorage.setItem(storageKey, JSON.stringify(config));
      dirty = false;
      storageError = "";
      render();
      $("save-status").textContent = "配置已保存到当前浏览器";
    } catch (_) {
      $("save-status").textContent = "保存失败：浏览器存储不可用。";
    }
  });
  async function refreshCheckpoints() {
    if (scanning) return;
    scanning = true;
    $("refresh-checkpoints").disabled = true;
    $("refresh-checkpoints").textContent = "扫描中…";
    try {
      const response = await fetch("/api/platform/checkpoints");
      if (!response.ok) throw new Error("目录读取失败");
      const data = await response.json();
      if (data.selection_version !== 2) throw new Error("扫描服务已更新，请重启平台后刷新页面。");
      if (!Array.isArray(data.checkpoints)) throw new Error("目录格式错误");
      checkpoints = data.checkpoints;
      $("checkpoint-scan-hint").title = data.directory;
      catalogError = "";
    } catch (failure) {
      checkpoints = [];
      catalogError = `${failure.message} 可重新扫描或手动选择文件夹。`;
    } finally {
      scanning = false;
      catalogLoaded = true;
      populateCheckpoints();
      render();
      $("refresh-checkpoints").textContent = "重新扫描";
    }
  }
  async function checkpointRequest(action, data = {}) {
    const response = await fetch(`/api/platform/checkpoints/${action}`, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Platform-Request": "1"},
      body: JSON.stringify(data), signal: AbortSignal.timeout(action === "pick" ? 125000 : 10000),
    });
    if (response.status === 404) throw new Error("文件夹选择服务已更新，请重启平台后刷新页面。");
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "文件夹选择失败");
    return result;
  }
  async function chooseManualCheckpoint() {
    if (selectionBusy) return;
    const previousSource = config.checkpointSource;
    config.checkpointSource = "manual";
    selectionBusy = true;
    selectionError = false;
    selectionMessage = "已打开本机文件夹窗口，请选择 checkpoint 文件夹。";
    render();
    try {
      const data = await checkpointRequest("pick");
      if (data.cancelled) {
        config.checkpointSource = previousSource;
        selectionMessage = "已取消选择，保留原 checkpoint。";
      } else {
        manualCheckpoint = data.checkpoint;
        config.customPath = manualCheckpoint.path;
        selectionMessage = "文件夹已选择，已读取模型配置。";
        dirty = true;
      }
    } catch (failure) {
      config.checkpointSource = previousSource;
      selectionError = true;
      selectionMessage = failure.message;
    } finally {
      selectionBusy = false;
      render();
    }
  }
  async function restoreManualCheckpoint() {
    if (config.checkpointSource !== "manual" || !config.customPath) return;
    selectionBusy = true;
    render();
    try {
      const data = await checkpointRequest("inspect", {path: config.customPath});
      manualCheckpoint = data.checkpoint;
      config.customPath = manualCheckpoint.path;
    } catch (failure) {
      selectionError = true;
      selectionMessage = `已保存的文件夹不可用：${failure.message}`;
    } finally {
      selectionBusy = false;
      render();
    }
  }
  $("refresh-checkpoints").addEventListener("click", refreshCheckpoints);
  $("checkpoint-source").addEventListener("change", () => {
    if ($("checkpoint-source").value === "manual") {
      chooseManualCheckpoint();
    } else {
      config.checkpointSource = "outputs";
      dirty = true;
      selectionMessage = "";
      render();
    }
  });
  $("choose-checkpoint").addEventListener("click", chooseManualCheckpoint);
  window.addEventListener("device-config", (event) => {
    for (const key of ["robot", "arms", "space"]) {
      if (config[key] !== event.detail[key]) dirty = true;
      config[key] = event.detail[key];
    }
    syncInputs();
  });
  syncInputs();
  const incomingCheckpoint = new URLSearchParams(window.location.search).get("checkpoint");
  refreshCheckpoints();
  if (incomingCheckpoint) {
    selectionBusy = true;
    render();
    checkpointRequest("inspect", {path: incomingCheckpoint}).then((data) => {
      manualCheckpoint = data.checkpoint;
      config.checkpointSource = "manual";
      config.customPath = manualCheckpoint.path;
      dirty = true;
      selectionMessage = "已带入训练 checkpoint，请核对机械臂、动作空间与相机后手动加载。";
    }).catch((failure) => {
      selectionError = true;
      selectionMessage = `无法带入此模型：${failure.message}`;
    }).finally(() => {
      selectionBusy = false;
      render();
    });
  } else {
    restoreManualCheckpoint();
  }
})();
