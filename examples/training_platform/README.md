# LeRobot 本地训练平台

在 LeRobot 项目内提供模型选择、配置编辑、训练监控、后台任务和 checkpoint 管理。
训练复用当前 checkout 的 LeRobot 配置注册表、数据处理器和 `lerobot_train`，不复制训练循环。
NERO 是机器人与数据集接入；训练界面没有限制机器人型号。真机运行使用已有推理平台。

## 启动

```bash
conda activate lerobot
cd /home/jushen/Documents/Robot/lerobot
python examples/training_platform/run.py
```

打开 http://127.0.0.1:7863 。依赖已有 LeRobot 训练环境和 Flask；无需新装训练框架。
默认采集入口为 `http://127.0.0.1:5000`，推理入口为 `http://127.0.0.1:7862`。
可用 `--port`、`--collection-url`、`--inference-url` 修改入口。

## 使用

1. 已有终端启动的 `lerobot-train` 会被发现并只读监控，读取配置、重定向日志与已保存 checkpoint。
   未重定向到文件的终端任务无法补回日志；缺失指标留空。平台不停止或接管这些进程。
2. 新建训练，选择本地数据集或填写 Hugging Face 数据集 ID，选择模型与模型起点。
   留空模型起点沿用该模型原生默认初始化；需要基座的模型仍需提供基座。
3. “编辑配置”包含常用参数、模型专属参数和完整 JSON，可导入、导出与校验。
   运行中修改仅保存下一次草稿，“本次运行配置”保留实际参数；恢复默认值也只作用于草稿。
4. 开始训练前检查类型、依赖、本地数据特征与输出目录。新输出目录不得覆盖现有训练。
   本地托管一次运行一个训练任务；已有终端训练时阻止新任务抢占显存。
5. “保存并停止”请求在完成训练更新、到达梯度累积边界后触发原生 checkpoint 保存，
   优化器、RNG、预处理器与 EMA（如启用）写完后退出。加载模型期间请求会等待进入该边界。
   保存失败会显示失败，不伪装成安全停止。关闭网页或平台服务不会停止后台训练。
6. 选完整 checkpoint 续训，填写**累计目标总步数**。续训创建新任务记录和独立输出目录，
   保留原 checkpoint；恢复原生训练状态。模型结构、动作表示、归一化或数据划分变化应新建微调。
7. checkpoint 可送到已有推理平台；目前推理页仅接入 π0.5，其他模型的权重照常保存。
   送入只设置模型路径，不自动加载、连接设备或执行动作。

“查看启动命令”提供当前实际命令，或为已保存草稿提供 `run.py --start TASK_ID` 命令。
该命令经过相同校验与任务管理后启动原生训练，避免续训时丢失嵌套配置。

## 模型兼容范围

模型和字段从当前安装的 LeRobot 注册表动态获取，包含策略与奖励模型，不固定为 π 系列。
通用入口支持原生 `lerobot-train` 的离线可训练模型。缺失可选依赖时显示原因；
包版本检查不是每个模型的实际训练验证，初始化错误会保留在任务日志中。

当前 checkout 的 GaussianActor 属于专用 RL 流程；TOPReward 是零样本模型，Robometer 当前实现仅推理。
它们会显示对应原因，不能作为离线训练任务启动。在线 RL 的环境/actor/learner 调度不包含在此页面。
托管启动暂限本机单进程；完整配置仍可编辑，分布式或云训练使用 LeRobot 原生入口。

日志中的不同模型 loss 含义不同，不能跨模型直接比较大小。界面不按 loss 自动判定最佳模型或过拟合。
epoch 按真实样本数、batch 和数据并行数换算；训练 step 的含义沿用当前 LeRobot（micro-batch）。
显存为训练日志提供的峰值；预计剩余时间按最近 step 耗时估算。曲线是已有日志，未记录的历史不补画。

## 文件与结构

- `training_platform/configuration.py`：注册表、原生配置校验、数据集和 checkpoint 元数据。
- `training_platform/jobs.py`：任务记录、独立进程、增量日志、外部任务发现。
- `training_platform/worker.py`：原生训练入口和 checkpoint 边界停止。
- `training_platform/app.py`：本机 HTTP API；`templates/` 和 `static/` 是原生 HTML/CSS/JS。

任务配置、草稿、日志、PID 身份和结果默认在 `outputs/training_platform/<task-id>/`；
训练权重默认在 `outputs/train/<task-id>/`。`--state-dir` 可指定任务记录目录。
目录选择窗口复用推理平台的 `DirectoryPicker`，需要桌面会话与 zenity；没有桌面时可编辑路径。

仅包含两项快速运行层测试（进度解析与保存停止边界），不下载模型、不执行真机动作：

```bash
PYTHONPATH=examples/training_platform python -m unittest discover -s examples/training_platform/tests -q
```
