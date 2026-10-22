/* The browser owns a draft; the server owns immutable running configurations. */
"use strict";
const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const state = {models: [], jobs: [], current: null, config: null, dataset: null, checkpoint: null,
  checkpoints: [], mode: "new", busy: false, online: false, editor: null, tab: "common", command: ""};
const statuses = {draft: "草稿", running: "训练中", stopping: "保存并停止中", stopped: "已停止",
  completed: "已完成", failed: "异常退出", exited: "已退出"};
const active = (job) => ["running", "stopping", "starting"].includes(job?.status);
const number = (value) => value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value) : null;
const format = (value, digits = 0) => number(value) === null ? "—" : Number(value).toLocaleString("zh-CN", {maximumFractionDigits: digits});
const modelConfig = (config = state.config) => config?.policy || config?.reward_model || {};
const modelKind = (config = state.config) => config?.reward_model ? "reward_model" : "policy";
const modelEntry = (config = state.config) => state.models.find((entry) => entry.id === modelConfig(config).type && entry.kind === modelKind(config));
function error(message, id = "global-error") { $(id).textContent = message || ""; $(id).hidden = !message; }
async function api(path, method = "GET", body) {
  const options = {method, headers: {"X-Platform-Request": "1"}};
  if (body !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
  return data;
}
async function action(callback) {
  if (state.busy) return;
  state.busy = true; error(""); renderControls();
  try { await callback(); } catch (exception) { error(exception.message); }
  finally { state.busy = false; renderControls(); }
}
function text(tag, content, className) { const element = document.createElement(tag); element.textContent = content; if (className) element.className = className; return element; }
function option(select, value, label) { const node = text("option", label); node.value = value; select.append(node); return node; }
function detail(container, label, value) { const row = document.createElement("div"); row.append(text("dt", label), text("dd", String(value))); container.append(row); }
function pathGet(object, path) { return path.split(".").reduce((item, key) => item?.[key], object); }
function pathSet(object, path, value) { const keys = path.split("."); const key = keys.pop(); let target = object; for (const part of keys) { if (!target[part] || typeof target[part] !== "object") target[part] = {}; target = target[part]; } target[key] = value; }
function syncSidebar() {
  if (!state.config) return;
  state.config.dataset ||= {};
  state.config.dataset.repo_id = $("dataset-repo").value.trim();
  const model = modelConfig();
  if (state.mode !== "resume") model.pretrained_path = $("pretrained-path").value.trim() || null;
}
function fillSidebar() {
  if (!state.config) return;
  const model = modelConfig();
  $("model-select").value = `${modelKind()}:${model.type}`;
  $("dataset-repo").value = state.config.dataset?.repo_id || "";
  $("pretrained-path").value = model.pretrained_path || "";
  $("training-mode").value = state.mode;
  $("dataset-select").value = state.config.dataset?.root || "";
  const checkpointSelect = $("checkpoint-select");
  if (state.checkpoint?.path && !Array.from(checkpointSelect.options).some((item) => item.value === state.checkpoint.path)) {
    option(checkpointSelect, state.checkpoint.path, `当前续训起点 · ${format(state.checkpoint.step)} 步 · ${state.checkpoint.path}`);
  }
  checkpointSelect.value = state.checkpoint?.path || "";
  renderConfig(); renderDataset(); renderControls();
}
function renderConfig() {
  const config = state.config;
  const entry = modelEntry();
  $("model-description").textContent = entry ? `${entry.kind === "reward_model" ? "奖励模型" : "策略模型"} · ${entry.available && entry.trainable !== false ? "使用原生 LeRobot 训练入口" : entry.reason || "当前模型需要专用训练流程"}` : "选择当前 LeRobot 支持的训练模型。";
  $("model-description").classList.toggle("warning", Boolean(entry && (!entry.available || entry.trainable === false)));
  $("pretrained-group").hidden = state.mode === "resume";
  $("resume-group").hidden = state.mode !== "resume";
  $("config-summary").replaceChildren();
  if (config) {
    detail($("config-summary"), "目标总步数", format(config.steps));
    detail($("config-summary"), "Batch size", format(config.batch_size));
    detail($("config-summary"), "保存间隔", `${format(config.save_freq)} 步`);
    const frames = number(state.dataset?.frames ?? state.dataset?.total_frames);
    const dataParallel = (config.parallelism?.dp_replicate || 1) * (config.parallelism?.dp_shard || 1);
    if (frames > 0 && number(config.batch_size) > 0) detail($("config-summary"), "目标折合 epoch", format(config.steps * config.batch_size * dataParallel / frames, 2));
    detail($("config-summary"), "设备", modelConfig().device || "原生自动选择");
  }
  if (state.checkpoint) {
    $("resume-info").textContent = `从 ${format(state.checkpoint.step)} 步继续；${state.checkpoint.resumable ? "包含完整训练状态" : "缺少完整训练状态，只能作为新训练的模型起点"}。`;
    $("resume-info").classList.toggle("warning", !state.checkpoint.resumable);
  }
}
function renderDataset() {
  const data = state.dataset;
  $("dataset-details").replaceChildren();
  if (data) {
    detail($("dataset-details"), "Episode / 帧数", `${format(data.episodes ?? data.total_episodes)} / ${format(data.frames ?? data.total_frames)}`);
    detail($("dataset-details"), "机械臂类型", data.robot_type || "未标注");
    detail($("dataset-details"), "动作空间 / 臂数", `${({joint: "关节空间", eef: "EEF"})[data.space] || "未标注"} / ${({single: "单臂", dual: "双臂"})[data.arms] || "未标注"}`);
    detail($("dataset-details"), "相机", Array.isArray(data.cameras) ? data.cameras.join("、") || "无图像" : "—");
    $("dataset-path").textContent = data.path || state.config?.dataset?.root || "";
  } else {
    detail($("dataset-details"), "数据状态", state.config?.dataset?.repo_id ? "尚未读取本地元数据" : "尚未选择");
    $("dataset-path").textContent = state.config?.dataset?.root || "";
  }
}
function renderControls() {
  const job = state.current;
  const running = active(job);
  const entry = modelEntry();
  const blocked = !state.online || state.busy || !state.config;
  $("start-training").disabled = blocked || running || job?.readonly || (entry && (!entry.available || entry.trainable === false)) || (state.mode === "resume" && !state.checkpoint?.resumable);
  $("stop-training").disabled = !state.online || state.busy || !running || job?.readonly || !job?.managed || job?.status === "stopping";
  $("save-draft").disabled = blocked;
  $("edit-config").disabled = blocked;
  $("validate-config").disabled = blocked;
  $("new-job").disabled = state.busy || !state.online;
  ["pick-dataset", "pick-model", "pick-checkpoint", "model-select", "training-mode", "dataset-select"].forEach((id) => { $(id).disabled = state.busy || !state.online; });
  $("start-training").textContent = state.mode === "resume" ? "▶ 继续训练" : "▶ 开始训练";
  $("footer-title").textContent = state.busy ? "正在处理…" : job ? statuses[job.status] || job.status : "准备新的训练";
  $("status-light").className = running ? "running" : "";
  $("footer-message").textContent = job?.readonly ? "已有终端任务 · 只读监控；启动与停止请使用原终端" : running ? "关闭网页不影响后台训练；修改配置仅保存为下一次草稿" : state.mode === "resume" ? "恢复完整训练状态；目标步数为累计总步数" : "使用当前草稿启动原生 LeRobot 训练";
}
function metric(metrics, ...keys) { for (const key of keys) if (number(metrics?.[key]) !== null) return Number(metrics[key]); return null; }
function renderJob() {
  const job = state.current;
  const metrics = job?.metrics || {};
  const step = number(job?.step); const total = number(job?.total_steps ?? job?.config?.steps);
  const progress = number(job?.progress) ?? (total > 0 && step !== null ? step / total : 0);
  $("job-title").textContent = job?.name || "准备新的训练";
  $("job-status").textContent = job ? `${statuses[job.status] || job.status}${job.readonly ? " · 只读" : ""}` : "草稿";
  $("job-status").className = `status-pill ${job?.status || "draft"}`;
  $("job-description").textContent = job?.error || (job?.readonly ? "从现有训练进程与日志读取；该任务由原终端管理。" : job?.draft_config && active(job) ? "本次使用启动时配置。已保存的修改将在下次运行生效。" : job ? `${modelConfig(job.config).type || "LeRobot"} · ${job.config?.dataset?.repo_id || "数据集未指定"}` : "选择数据与模型，编辑配置后启动训练。");
  $("job-description").classList.toggle("warning", Boolean(job?.error));
  $("step-value").replaceChildren(document.createTextNode(format(step)), text("span", ` / ${format(total)} steps`));
  $("progress-value").textContent = step === null ? "尚未开始" : `${format(Math.max(0, Math.min(1, progress)) * 100, 1)}%`;
  $("progress-bar").style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  $("epoch-value").textContent = `折合 epoch ${format(job?.epoch, 3)}`;
  $("resume-value").textContent = job?.start_step ? `本次追加 ${format(Math.max(0, (step || 0) - job.start_step))} 步 · 起点 ${format(job.start_step)}` : "";
  const secondsPerStep = metric(metrics, "step_s", "updt_s", "update_s");
  const speed = metric(metrics, "steps_per_second", "steps_per_s") ?? (secondsPerStep > 0 ? 1 / secondsPerStep : null);
  const remaining = speed > 0 && total !== null && step !== null && active(job) ? Math.max(0, total - step) / speed : null;
  $("eta-value").textContent = `预计剩余 ${remaining === null ? "—" : remaining >= 3600 ? `${format(remaining / 3600, 1)} 小时` : `${format(remaining / 60, 0)} 分钟`}`;
  $("metric-loss").textContent = format(metric(metrics, "loss", "train_loss", "train/loss"), 5);
  $("metric-speed").textContent = format(speed, 2);
  const lr = metric(metrics, "lr", "learning_rate");
  $("metric-lr").textContent = lr === null ? "—" : lr.toExponential(2);
  $("metric-memory").textContent = format(metric(metrics, "mem_gb", "memory_gb", "gpu_memory_gb", "max_memory_gb"), 2);
  const logs = Array.isArray(job?.logs) ? job.logs.join("\n") : job?.logs || "尚无训练日志。";
  if ($("log-output").textContent !== logs) { $("log-output").textContent = logs; if ($("follow-logs").checked) $("log-output").scrollTop = $("log-output").scrollHeight; }
  $("log-file").textContent = job?.log_path || "日志保存在本地";
  $("log-file").title = job?.log_path || "";
  $("output-path").textContent = job?.output_dir || "";
  renderChartOptions(); drawChart(); renderCheckpoints(); renderControls();
}
function renderJobs() {
  const list = $("job-list"); list.replaceChildren();
  if (!state.jobs.length) list.append(text("p", "尚无训练任务", "empty"));
  for (const job of state.jobs) {
    const card = text("button", "", `job-card${job.id === state.current?.id ? " selected" : ""}`);
    const row = document.createElement("div");
    row.append(text("strong", job.name || job.id), text("span", statuses[job.status] || job.status, `status-pill ${job.status}`));
    card.append(row, text("p", `${modelConfig(job.config).type || "LeRobot"} · ${format(job.step)} / ${format(job.total_steps ?? job.config?.steps)} 步${job.readonly ? " · 外部任务" : ""}`));
    card.addEventListener("click", () => action(() => selectJob(job.id)));
    list.append(card);
  }
}
function renderCheckpoints() {
  const checkpoints = state.current?.checkpoints || [];
  $("checkpoint-count").textContent = `${checkpoints.length} 个 checkpoint`;
  const list = $("checkpoint-list"); list.replaceChildren();
  if (!checkpoints.length) list.append(text("p", "尚无已保存的 checkpoint", "empty"));
  for (const checkpoint of checkpoints.slice().reverse()) {
    const card = text("div", "", "checkpoint-card");
    const header = document.createElement("div");
    header.append(text("strong", `Step ${format(checkpoint.step)}`), text("span", checkpoint.resumable ? "可续训" : "模型权重", "tag"));
    card.append(header, text("p", checkpoint.path, "path"));
    const actions = text("div", "", "card-actions");
    const resume = text("button", "设为续训起点", "text-button");
    resume.disabled = !checkpoint.resumable;
    resume.addEventListener("click", () => action(() => chooseCheckpoint(checkpoint.path, true)));
    const infer = text("button", "前往推理", "text-button");
    infer.addEventListener("click", () => action(async () => {
      const entry = modelConfig(state.current?.config).type;
      if (entry !== "pi05") throw new Error("当前推理平台仅接入 π0.5。此 checkpoint 已保留，可在对应模型推理接入后使用。");
      const url = new URL(document.body.dataset.inferenceUrl, location.href);
      url.searchParams.set("checkpoint", checkpoint.path);
      location.href = url.href;
    }));
    actions.append(resume, infer); card.append(actions); list.append(card);
  }
}
function renderChartOptions() {
  const keys = new Set(["loss"]);
  for (const point of state.current?.history || []) for (const [key, value] of Object.entries(point)) if (!["step", "epoch", "timestamp", "samples", "time"].includes(key) && number(value) !== null) keys.add(key);
  const selected = $("chart-metric").value;
  if (Array.from($("chart-metric").options).map((item) => item.value).join("|") !== [...keys].join("|")) { $("chart-metric").replaceChildren(); for (const key of keys) option($("chart-metric"), key, key); $("chart-metric").value = keys.has(selected) ? selected : "loss"; }
}
function drawChart() {
  const canvas = $("loss-chart"); const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const context = canvas.getContext("2d"); context.scale(ratio, ratio);
  const width = rect.width, height = rect.height; const margin = {left: 48, top: 18, right: 15, bottom: 28};
  const key = $("chart-metric").value;
  const points = (state.current?.history || []).filter((point) => number(point.step) !== null && number(point[key]) !== null);
  $("chart-empty").hidden = points.length > 0;
  let minimum = Math.min(...points.map((point) => Number(point[key]))), maximum = Math.max(...points.map((point) => Number(point[key])));
  if (!points.length) { minimum = 0; maximum = 1; }
  const span = maximum - minimum || Math.max(Math.abs(maximum) * .2, .01);
  minimum -= span * .08; maximum += span * .08;
  const first = points.length ? Math.min(...points.map((point) => Number(point.step))) : 0;
  const last = points.length ? Math.max(...points.map((point) => Number(point.step))) : 1;
  const x = (value) => margin.left + (value - first) / (last - first || 1) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (value - minimum) / (maximum - minimum) * (height - margin.top - margin.bottom);
  context.font = "9px ui-monospace, monospace"; context.strokeStyle = "#2a394d"; context.fillStyle = "#6d85a3"; context.lineWidth = 1;
  for (let index = 0; index <= 4; index++) {
    const value = minimum + (maximum - minimum) * index / 4;
    const position = y(value); context.beginPath(); context.moveTo(margin.left, position); context.lineTo(width - margin.right, position); context.stroke();
    if (points.length) { context.textAlign = "right"; context.fillText(Math.abs(value) < .01 && value !== 0 ? value.toExponential(1) : Number(value.toPrecision(3)).toString(), margin.left - 8, position + 3); }
  }
  if (!points.length) return;
  context.textAlign = "left"; context.fillText(format(first), margin.left, height - 7);
  context.textAlign = "right"; context.fillText(`${format(last)} steps`, width - margin.right, height - 7);
  context.beginPath(); points.forEach((point, index) => index ? context.lineTo(x(point.step), y(point[key])) : context.moveTo(x(point.step), y(point[key])));
  context.strokeStyle = "#6eb5fc"; context.lineWidth = 1.6; context.lineJoin = "round"; context.stroke();
  if (points.length === 1) { context.beginPath(); context.arc(x(points[0].step), y(points[0][key]), 3, 0, Math.PI * 2); context.fillStyle = "#6eb5fc"; context.fill(); }
}
async function selectJob(id) {
  const {job} = await api(`/api/jobs/${encodeURIComponent(id)}`);
  state.current = job; state.config = clone(job.draft_config || job.config);
  const resumeSource = Object.hasOwn(job, "draft_checkpoint") ? job.draft_checkpoint : job.resume_from;
  state.mode = resumeSource ? "resume" : "new";
  state.checkpoint = resumeSource ? {path: resumeSource, step: job.start_step, resumable: true} : null;
  if (resumeSource) {
    try { state.checkpoint = (await api("/api/checkpoints/inspect", "POST", {path: resumeSource})).checkpoint; }
    catch (_) { /* Keep the source path visible even when an old checkpoint is unavailable. */ }
  }
  state.dataset = null; localStorage.setItem("lerobot-training-current-job", id);
  const datasetSource = state.config?.dataset?.root || state.config?.dataset?.repo_id;
  if (datasetSource) {
    try { state.dataset = (await api("/api/datasets/inspect", "POST", {path: datasetSource})).dataset; } catch (_) { /* Remote/unavailable metadata must not hide a running job. */ }
  }
  fillSidebar(); renderJob(); renderJobs();
}
async function newDraft() {
  const entry = modelEntry() || state.models.find((item) => item.available && item.trainable !== false) || state.models[0];
  if (!entry) throw new Error("没有读取到 LeRobot 模型，请检查服务端环境。");
  state.config = (await api(`/api/defaults?model=${encodeURIComponent(entry.id)}&kind=${encodeURIComponent(entry.kind)}`)).config;
  state.current = null; state.dataset = null; state.checkpoint = null; state.mode = "new";
  localStorage.removeItem("lerobot-training-current-job");
  $("validation-status").textContent = "启动前会校验配置与所需依赖。";
  fillSidebar(); renderJob(); renderJobs();
}
async function chooseDataset(path) {
  const {dataset} = await api("/api/datasets/inspect", "POST", {path});
  state.dataset = dataset; state.config.dataset ||= {};
  state.config.dataset.root = dataset.path || path; state.config.dataset.repo_id = dataset.repo_id || state.config.dataset.repo_id;
  fillSidebar();
}
async function chooseCheckpoint(path, newRecord = false) {
  const {checkpoint} = await api("/api/checkpoints/inspect", "POST", {path});
  if (!checkpoint.resumable) throw new Error(`此目录缺少完整训练状态，不能断点续训。${(checkpoint.resume_errors || []).join("；")} 可在“新建训练”的模型起点中使用权重。`);
  state.checkpoint = checkpoint; state.mode = "resume";
  if (checkpoint.train_config) state.config = clone(checkpoint.train_config);
  if (newRecord) { state.current = null; localStorage.removeItem("lerobot-training-current-job"); }
  state.config.resume = true;
  fillSidebar(); renderJob(); renderJobs();
  $("validation-status").textContent = `已选择第 ${format(checkpoint.step)} 步 checkpoint。请在“编辑配置”中设置新的累计目标步数。`;
}
async function pick(kind) {
  const result = await api("/api/pick", "POST", {kind});
  if (result.cancelled || !result.path) return;
  if (kind === "dataset") await chooseDataset(result.path);
  else if (kind === "checkpoint") await chooseCheckpoint(result.path);
  else { modelConfig().pretrained_path = result.path; fillSidebar(); }
}
async function validate(config = state.config) {
  const result = await api("/api/validate", "POST", {config, mode: state.mode, checkpoint: state.checkpoint?.path || null});
  const errors = result.errors || [];
  return {...result, message: errors.length ? errors.join("\n") : (result.warnings || []).join("\n") || "配置校验通过。"};
}
async function saveDraft() {
  syncSidebar();
  const payload = {config: state.config, checkpoint: state.mode === "resume" ? state.checkpoint?.path || null : null};
  const result = state.current ? await api(`/api/jobs/${encodeURIComponent(state.current.id)}`, "PUT", payload) : await api("/api/jobs", "POST", {...payload, name: state.config.job_name || undefined});
  state.current = result.job;
  localStorage.setItem("lerobot-training-current-job", result.job.id);
  $("validation-status").textContent = active(result.job) ? "已保存下一次训练草稿；当前运行配置保持不变。" : "训练草稿已保存。";
  await refresh(); return result.job;
}
const commonFields = [
  ["job_name", "任务名称", "string"], ["output_dir", "输出目录", "string"],
  ["steps", "目标总步数", "number"], ["batch_size", "Batch size", "number"],
  ["save_freq", "Checkpoint 保存间隔", "number"], ["log_freq", "日志间隔", "number"],
  ["num_workers", "数据加载进程数", "number"], ["seed", "随机种子", "number"],
  ["dataset.repo_id", "数据集 Repo ID", "string"], ["dataset.root", "本地数据集目录", "string"],
  ["save_checkpoint", "保存 checkpoint", "boolean"], ["wandb.enable", "启用 Weights & Biases", "boolean"]
];
function field(container, path, label, value, explicitType) {
  const block = text("div", "", "field-block");
  const labelNode = text("label", label); const id = `field-${path.replaceAll(".", "-")}`; labelNode.htmlFor = id;
  const type = explicitType || (value !== null && typeof value === "object" ? "json" : typeof value);
  let input;
  if (type === "boolean") { input = document.createElement("select"); option(input, "true", "开启 / true"); option(input, "false", "关闭 / false"); input.value = String(Boolean(value)); }
  else if (type === "json" || (!explicitType && (value === null || value === undefined))) { input = document.createElement("textarea"); input.value = JSON.stringify(value ?? null, null, 2); input.dataset.type = "json"; }
  else { input = document.createElement("input"); input.type = type === "number" ? "number" : "text"; input.value = value ?? ""; if (type === "number") input.step = "any"; }
  input.id = id; input.dataset.path = path; input.dataset.type ||= type;
  block.append(labelNode, input, text("p", path, "hint")); container.append(block);
}
function renderEditor() {
  $("common-fields").replaceChildren(); $("model-fields").replaceChildren();
  for (const [path, label, type] of commonFields) field($("common-fields"), path, label, pathGet(state.editor, path), type);
  const kind = modelKind(state.editor); const model = modelConfig(state.editor);
  for (const [key, value] of Object.entries(model)) {
    if (key === "type") continue;
    field($("model-fields"), `${kind}.${key}`, key, value);
  }
  $("config-json").value = JSON.stringify(state.editor, null, 2);
  $("actual-config").textContent = JSON.stringify(state.current?.config || state.editor, null, 2);
}
function readEditor() {
  if (state.tab === "json") {
    const config = JSON.parse($("config-json").value);
    if (!config || Array.isArray(config) || typeof config !== "object") throw new Error("完整配置必须是 JSON 对象。");
    state.editor = config;
  } else if (state.tab !== "actual") {
    const container = $(state.tab === "common" ? "common-fields" : "model-fields");
    for (const input of container.querySelectorAll("[data-path]")) {
      let value = input.value;
      if (input.dataset.type === "number") { if (!value.trim()) throw new Error(`${input.dataset.path} 不能为空。`); value = Number(value); if (!Number.isFinite(value)) throw new Error(`${input.dataset.path} 必须为数字。`); }
      else if (input.dataset.type === "boolean") value = value === "true";
      else if (input.dataset.type === "json") { try { value = JSON.parse(value); } catch (_) { throw new Error(`${input.dataset.path} 不是有效 JSON。`); } }
      else if (value === "" && ["output_dir", "job_name", "dataset.root"].includes(input.dataset.path)) value = null;
      pathSet(state.editor, input.dataset.path, value);
    }
  }
  return state.editor;
}
function changeTab(tab) {
  try { readEditor(); renderEditor(); error("", "config-error"); }
  catch (exception) { error(exception.message, "config-error"); return; }
  state.tab = tab;
  for (const button of document.querySelectorAll("[data-tab]")) { button.setAttribute("aria-selected", String(button.dataset.tab === tab)); $( `tab-${button.dataset.tab}`).hidden = button.dataset.tab !== tab; }
}
function openEditor() {
  syncSidebar(); state.editor = clone(state.config); state.tab = "common";
  $("edit-notice").textContent = active(state.current) ? "当前任务正在运行。编辑后保存为下一次训练草稿，不会修改本次运行。" : "修改保存至训练草稿。启动时将由 LeRobot 校验并使用此配置。";
  document.querySelector('[data-tab="actual"]').hidden = !state.current;
  error("", "config-error"); $("config-result").textContent = ""; renderEditor();
  for (const button of document.querySelectorAll("[data-tab]")) { button.setAttribute("aria-selected", String(button.dataset.tab === "common")); $(`tab-${button.dataset.tab}`).hidden = button.dataset.tab !== "common"; }
  $("config-dialog").showModal();
}
async function editorAction(callback) {
  error("", "config-error");
  try { await callback(); } catch (exception) { error(exception.message, "config-error"); }
}
function shellCommand(command) {
  if (typeof command === "string") return command;
  return (command || []).map((part) => /^[a-zA-Z0-9_./:=+-]+$/.test(String(part)) ? String(part) : `'${String(part).replaceAll("'", "'\\''")}'`).join(" \\\n  ");
}
async function refresh() {
  const result = await api("/api/jobs"); state.jobs = result.jobs || []; state.online = true;
  $("connection").className = "connection online"; $("connection").replaceChildren(document.createElement("i"), document.createTextNode("本地服务在线"));
  const currentId = state.current?.id;
  if (currentId) {
    const {job} = await api(`/api/jobs/${encodeURIComponent(currentId)}`);
    if (state.current?.id === currentId) state.current = job;
  }
  renderJobs(); renderJob();
}
function bind() {
  $("new-job").addEventListener("click", () => action(newDraft));
  $("model-select").addEventListener("change", () => action(async () => {
    syncSidebar(); const [kind, model] = $("model-select").value.split(":");
    const config = (await api(`/api/defaults?model=${encodeURIComponent(model)}&kind=${encodeURIComponent(kind)}`)).config;
    config.dataset = clone(state.config?.dataset || config.dataset); state.config = config; state.mode = "new"; state.checkpoint = null; fillSidebar();
  }));
  $("training-mode").addEventListener("change", () => { syncSidebar(); state.mode = $("training-mode").value; state.config.resume = state.mode === "resume"; renderConfig(); renderControls(); });
  $("dataset-select").addEventListener("change", () => { if ($("dataset-select").value) action(() => chooseDataset($("dataset-select").value)); });
  $("checkpoint-select").addEventListener("change", () => { if ($("checkpoint-select").value) action(() => chooseCheckpoint($("checkpoint-select").value)); });
  $("dataset-repo").addEventListener("change", () => { syncSidebar(); state.dataset = null; state.config.dataset.root = null; renderDataset(); renderConfig(); });
  $("pretrained-path").addEventListener("change", syncSidebar);
  for (const kind of ["dataset", "checkpoint", "model"]) $(`pick-${kind}`).addEventListener("click", () => action(() => pick(kind)));
  $("save-draft").addEventListener("click", () => action(saveDraft));
  $("validate-config").addEventListener("click", () => action(async () => { syncSidebar(); const result = await validate(); $("validation-status").textContent = result.message; $("validation-status").classList.toggle("warning", !result.valid); }));
  $("start-training").addEventListener("click", () => action(async () => {
    syncSidebar(); const result = await validate();
    if (!result.valid) throw new Error(result.message);
    const job = await saveDraft();
    const launched = await api(`/api/jobs/${encodeURIComponent(job.id)}/start`, "POST", {checkpoint: state.mode === "resume" ? state.checkpoint?.path : undefined});
    await selectJob(launched.job.id); await refresh();
  }));
  $("stop-training").addEventListener("click", () => action(async () => { state.current = (await api(`/api/jobs/${encodeURIComponent(state.current.id)}/stop`, "POST", {})).job; await refresh(); }));
  $("refresh-jobs").addEventListener("click", () => action(refresh));
  $("chart-metric").addEventListener("change", drawChart);
  new ResizeObserver(drawChart).observe($("loss-chart"));
  $("edit-config").addEventListener("click", openEditor);
  for (const button of document.querySelectorAll("[data-close]")) button.addEventListener("click", () => $(button.dataset.close).close());
  for (const button of document.querySelectorAll("[data-tab]")) button.addEventListener("click", () => changeTab(button.dataset.tab));
  $("check-editor").addEventListener("click", () => editorAction(async () => { const result = await validate(readEditor()); $("config-result").textContent = result.message; if (!result.valid) error(result.message, "config-error"); }));
  $("save-config").addEventListener("click", () => editorAction(async () => { state.config = clone(readEditor()); fillSidebar(); await saveDraft(); $("config-dialog").close(); }));
  $("reset-config").addEventListener("click", () => editorAction(async () => {
    const entry = modelEntry(state.editor); if (!entry) throw new Error("请选择有效模型。");
    const config = (await api(`/api/defaults?model=${encodeURIComponent(entry.id)}&kind=${encodeURIComponent(entry.kind)}`)).config;
    config.dataset = clone(state.editor.dataset || config.dataset); state.editor = config; renderEditor(); $("config-result").textContent = "已恢复该模型默认配置，并保留数据集。保存后生效。";
  }));
  $("import-config").addEventListener("click", () => $("import-file").click());
  $("import-file").addEventListener("change", () => editorAction(async () => {
    const file = $("import-file").files[0]; if (!file) return; if (file.size > 2 * 1024 * 1024) throw new Error("配置文件不能超过 2 MB。");
    const config = JSON.parse(await file.text()); if (!config || Array.isArray(config) || typeof config !== "object") throw new Error("配置必须是 JSON 对象。");
    state.editor = config; renderEditor(); $("import-file").value = ""; $("config-result").textContent = "已导入，请校验后保存。";
  }));
  $("export-config").addEventListener("click", () => editorAction(async () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(readEditor(), null, 2)], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = "train_config.json"; link.click(); URL.revokeObjectURL(url);
  }));
  $("view-command").addEventListener("click", () => action(async () => {
    let command = active(state.current) ? state.current.command : null;
    if (!command?.length) {
      const job = await saveDraft();
      command = (await api(`/api/jobs/${encodeURIComponent(job.id)}/command`, "POST", {checkpoint: state.mode === "resume" ? state.checkpoint?.path : undefined})).command;
    }
    state.command = shellCommand(command); $("command-text").textContent = state.command || "先保存并校验配置，生成启动命令。"; $("copy-result").textContent = ""; $("command-dialog").showModal();
  }));
  $("copy-command").addEventListener("click", async () => { try { await navigator.clipboard.writeText(state.command); $("copy-result").textContent = "已复制"; } catch (_) { $("copy-result").textContent = "无法访问剪贴板，请选择上方命令手动复制。"; } });
}
async function initialize() {
  bind();
  try {
    const results = await Promise.allSettled([api("/api/catalog"), api("/api/datasets"), api("/api/checkpoints"), api("/api/jobs")]);
    if (results[0].status === "rejected") throw results[0].reason;
    state.models = results[0].value.models || [];
    $("model-select").replaceChildren();
    for (const entry of state.models) option($("model-select"), `${entry.kind}:${entry.id}`, `${entry.label || entry.id}${entry.kind === "reward_model" ? " · 奖励模型" : ""}${entry.trainable === false ? " · 专用流程" : !entry.available ? " · 需配置" : ""}`);
    if (results[1].status === "fulfilled") for (const data of results[1].value.datasets || []) option($("dataset-select"), data.path, data.repo_id || data.name || data.path);
    if (results[2].status === "fulfilled") {
      state.checkpoints = results[2].value.checkpoints || [];
      for (const checkpoint of state.checkpoints) option($("checkpoint-select"), checkpoint.path, `${checkpoint.model || "模型"} · ${format(checkpoint.step)} 步 · ${checkpoint.path}`);
    }
    if (results[3].status === "fulfilled") state.jobs = results[3].value.jobs || [];
    const remembered = localStorage.getItem("lerobot-training-current-job");
    const selected = state.jobs.find((job) => job.id === remembered) || state.jobs.find(active) || state.jobs[0];
    if (selected) await selectJob(selected.id); else await newDraft();
    await refresh();
    const unavailable = results.slice(1).filter((result) => result.status === "rejected");
    if (unavailable.length) error(`部分列表暂不可用：${unavailable.map((result) => result.reason.message).join("；")}`);
  } catch (exception) { error(exception.message); }
  window.setInterval(async () => {
    if (state.busy || document.hidden) return;
    try { await refresh(); } catch (_) { state.online = false; $("connection").className = "connection"; $("connection").replaceChildren(document.createElement("i"), document.createTextNode("服务连接中断")); renderControls(); }
  }, 2000);
}
initialize();
