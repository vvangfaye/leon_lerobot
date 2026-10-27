# 用 Codex 直接操作 NERO

此入口复用本仓库的 NERO FK/IK、相机与硬件状态检查，不需要训练策略，也不需要单独设置 OpenAI API key。推理由连接它的 Codex 会话负责。

## 快速开始

已有本仓库 `.venv` 和 `pyAgxArm` 环境时，在仓库根目录运行：

```bash
# 与本机已注册的服务使用同一个缓存；首次自检会下载 MCP SDK。
export UV_CACHE_DIR=/tmp/leon-uv-cache

# 完整 MCP 协议自检：只使用模拟机械臂，不连接 CAN 或真实相机。
uv run --no-sync --with 'mcp==1.30.0' python examples/nero_to_nero/smoke_mcp.py

# 注册真实硬件的观察模式：可看相机和状态，不能使能或移动。
codex mcp add nero --env UV_CACHE_DIR=/tmp/leon-uv-cache -- bash "$PWD/examples/nero_to_nero/run_nero_mcp.sh"
```

重启 Codex 会话以加载工具。第一条提示可以是：

> 调用 nero 的 get_robot_info，检查从左和从右的映射，再读取左右臂状态，显示 head、left、right 三路相机。仅观察，不准备机械臂、不移动。

若想先在 Codex 里使用模拟机械臂，注册时在脚本后加 `--mock --allow-motion`；模拟图像上会写明 MOCK，所有结构化结果也带 `mock: true`。

## 开启真实小步动作

结束数采工作台、遥操作、策略推理和其他占用 CAN 的控制程序。停止上述程序时沿用它们现有的保持姿态流程。由现场操作者检查两条从臂和工作空间后，再更换 MCP 配置：

```bash
codex mcp remove nero
codex mcp add nero --env UV_CACHE_DIR=/tmp/leon-uv-cache -- bash "$PWD/examples/nero_to_nero/run_nero_mcp.sh" --allow-motion
```

重启 Codex，再请求一个具体动作，例如：

> 先看 head、right 相机并读取 right 状态。现场已准备好，仅准备右从臂；保持末端姿态，在右臂基坐标系 z 方向上移 1 mm，2 秒完成，再读取状态并显示相机。左臂不动；失败就停止，不扩大步长或改限位。

这只是首个接口验证动作，不代表已经完成抓取。夹爪默认使用 3 N 力；若控制器报告夹爪尚未使能，先在已有硬件控制台完成准备，本服务不会偷偷切换或复位夹爪模式。

启动脚本使用离线缓存，避免 Codex 启动时等待网络。如果系统清理了 `/tmp`，重新运行上面的自检即可下载依赖。`get_robot_state` 的 `outside_conservative_limits` 会列出当前超出保守限位的关节编号；此时先停止该臂的动作，不自动扩大限位。

## 工具与边界

| 工具 | 功能 |
| --- | --- |
| `get_robot_info` | 当前模式、CAN 映射、相机和限制；不连接硬件 |
| `get_robot_state(side)` | 新鲜的关节角（度）、夹爪开度（mm）、末端位姿 |
| `look(camera)` | head / left / right 的新鲜 RGB 图像 |
| `prepare_arm(side)` | 显式准备左或右从臂；不回零；需 `--allow-motion` |
| `move_joints(side, target_degrees, duration_s)` | 七个绝对关节角；每次最多改变 5°，速度最多 5°/s |
| `move_ee_by(side, dx_mm, dy_mm, dz_mm, duration_s)` | 单次最多平移 10 mm，保持姿态；IK 误差上限 1 mm / 3° |
| `set_gripper(side, width_mm)` | 0–70 mm，单次改变最多 10 mm |
| `stop_motion()` | 取消进行中的动作，尝试在新鲜反馈位置保持；再次动作前需重新准备 |

默认从 `collect_config.platform.json` 按角色读取映射：当前从左 `can0`、从右 `can1`；不会控制主左 `can2`、主右 `can3`。可用 `--config /absolute/path.json` 指定另一份平台配置。不会修改原配置、校准、数据集或策略权重。

一次只执行一条臂的动作。局部轨迹按约 50 Hz 下发，完成后检查实测到位；模型不参与高频循环。拒绝越界、非有限数、反馈不完整或过期的动作。连接和退出不自动复位或失能。

**坐标是每条臂自己的基坐标系，不是相机或统一世界坐标系。** 目前没有手眼标定、桌面/障碍物避碰或双臂碰撞规划。看到图像不等于已经知道物体的毫米坐标。首先验证小步运动与方向，再测量桌面、抓取点和相机外参。

当前 `eef_xyz_m` 和 `eef_rotation_6d` 来自 SDK 的 **link7 坐标系**，尚未包含实际夹爪抓取中心的 TCP 标定。不能将它们直接当成指尖或物体接触点的位置；包装、插入等任务需先标定工具偏移。

`stop_motion` 是软件取消，不是硬件急停；已发给固件的动作只能在通信和反馈正常时尝试覆盖为保持目标。反馈丢失时不会猜测位置或发零位目标，应由现场操作者使用设备急停。

进程锁只防止本入口的多个实例竞争；它不能阻止其他 SocketCAN 控制程序。真实使用前仍须关闭其他控制源。

## 验证

```bash
uv run --no-sync python -m unittest discover -s examples/nero_to_nero -p 'test_agent_*.py' -v
uv run --no-sync --with 'mcp==1.30.0' python examples/nero_to_nero/smoke_mcp.py
```

协议自检包含初始化、8 个工具、模拟图像、关节移动、末端移动、夹爪和停止后的拒绝动作。以上均是离线验证，不能代替 NERO 真机验收。
