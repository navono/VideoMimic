# Simulation 训练架构

## 四个训练阶段详解

### Stage 1: MoCap Pre-training (g1_deepmimic)

- **脚本**: `train_stage_1_mcpt.sh`
- **输入**: AMASS 动捕数据 (`use_amass=True`，使用 `lafan_walk_and_dance/*.pkl`)
  - walk (5 种变体, 8 个序列), dance (2 种变体, 7 个序列), run (1 个序列), 共 19 个序列
- **训练**: 从零开始，用多样化动捕数据训练通用的"动作跟踪器"——让机器人学会跟着参考轨迹动
- **输出 checkpoint**: `20250410_063030_g1_deepmimic_mcpt/model_300000.pt`
- **无需前置 checkpoint**

### Stage 2: Terrain RL (g1_deepmimic_proj_heightfield)

- **脚本**: `train_stage_2_terrain_rl.sh`
- **输入**: 人类视频数据 (`use_human_videos=True`, `use_amass=False`，123 个动作，主要为楼梯地形)
- **加载**: Stage 1 的 checkpoint (`--load_run 20250410_063030_g1_deepmimic_mcpt --resume`)
- **训练**: 在真实地形网格上做 RL 微调，学习适应地形
- **输出 checkpoint**: `20250414_170842_g1_deepmimic_terrain/model_42250.pt`
- **观测**: 包含过去 5 帧的历史观测（torso_real, torso_xy_rel, torso_yaw_rel 等），在仿真中表现好但不适合真机部署

### Stage 3: Distillation (g1_deepmimic_root_heightfield_no_history_dagger)

- **脚本**: `train_stage_3_distillation.sh`
- **输入**: 人类视频数据 + Stage 2 的策略作为 teacher
- **加载**: Stage 2 的 checkpoint 作为 `policy_to_clone`（DAgger 蒸馏）
- **训练**: 将 Stage 2 的 teacher 策略蒸馏到无历史观测的学生策略
- **输出 checkpoint**: `20250502_124756_g1_deepmimic_distill/model_5250.pt`

**为什么要去掉 obs_history？**

Stage 2 的策略输入包含过去 5 帧的历史观测，让策略在仿真中表现更好（能从历史推断速度、趋势），但部署到真机上有问题：
- 真机控制循环有时序约束，维护多帧历史增加延迟
- 传感器噪声/丢帧会导致历史数据不准
- 更大的输入 → 更大的网络 → 更慢的推理（Jetson 算力有限）

Stage 3 蒸馏后，学生策略只使用当前帧观测，用更小的输入换取可部署性。

### Stage 4: RL Finetuning (g1_deepmimic_root_heightfield_no_history_dagger)

- **脚本**: `train_stage_4_rl_finetune.sh`
- **输入**: AMASS 动捕数据 + 人类视频数据（两种同时使用，`use_amass=True` AND `use_human_videos=True`）
- **加载**: Stage 3 的 checkpoint (`--resume --load_run 20250502_124756_g1_deepmimic_distill`)
- **训练**: 在蒸馏策略基础上做纯 RL 微调，关闭 BC loss (`bc_loss_coef=0.0`)
- **输出**: 需要运行后生成（当前磁盘上没有 Stage 4 的 checkpoint）

> **已知问题**: Stage 4 脚本中 `amass_replay_data_path=lafan_walk/*.pkl`，但 `lafan_walk/` 目录不存在。实际可用的目录是 `lafan_walk_and_dance/` 或 `lafan_replay_data/`。

**为什么要关闭 BC loss，只用纯 RL？**

Stage 3 蒸馏时使用 BC（行为克隆）+ RL，策略在模仿 teacher 的同时优化 reward。但蒸馏后学生和 teacher 之间有差距。Stage 4 关闭 BC loss 意味着：
- 不再模仿 teacher，完全靠 RL reward 信号自我优化
- 策略可以超越 teacher，找到比蒸馏结果更好的动作方式
- 弥补蒸馏造成的性能损失

简单说：**Stage 2→3 是"学着做"，Stage 4 是"自己做，做得更好"**。

> **Checkpoint 路径**: 所有 checkpoint 存储在 `simulation/videomimic_gym/logs/g1_deepmimic/` 下，目录名格式为 `<时间戳>_<run_name>`。

## 数据流总结

```
AMASS 动捕数据 → [Stage 1] → 20250410_063030_g1_deepmimic_mcpt (MCPT 策略)
                                      ↓ resume
人类视频 + 地形  → [Stage 2] → 20250414_170842_g1_deepmimic_terrain (地形 RL 策略, teacher)
                                      ↓ policy_to_clone (DAgger 蒸馏)
人类视频 + 地形  → [Stage 3] → 20250502_124756_g1_deepmimic_distill (无历史学生策略, 可部署)
                                      ↓ resume
AMASS + 人类视频 → [Stage 4] → (待训练) (最终 RL 微调策略)
```

## 判断是否需要重训的标准

| 新动作类型 | 是否需要重训 | 原因 |
|---|---|---|
| 不同地形的走路（上坡、碎石等） | 不需要 | 和训练数据同类 |
| 不同的走路风格（快走、慢走） | 不需要 | 策略已见过各种 walk |
| 上下楼梯 | 不需要 | 训练数据已包含 stairs |
| 坐 / 站 | 建议重训 | 训练数据中没有这种大幅度重心变化 |
| 蹲下 / 起立 | 建议重训 | 同上 |
| 跳跃 | 需要重训 | 完全没见过 |
| 跑步 | 可能不需要 | AMASS 有 run2 |
| 舞蹈动作 | 可能不需要 | AMASS 有 dance |

**简单判断方法**: 看新动作的关节运动模式和重心变化是否在训练数据覆盖范围内。如果差异大（比如坐/站涉及从 0.78m 高度降到很低），策略没学过这种模式，直接 play 效果会差，加入训练数据重训 Stage 3-4 会好很多。

## 只加新技能（如 sitting_standing）的流程

需要通过 real2sim 流程生成新的 `retarget_poses_g1.h5` + `background_mesh.obj`，然后：

1. 创建新的 YAML 配置文件（参考 `sitting_standing_motion.yaml`）
2. 从 Stage 2 或 Stage 3 开始重训即可，不需要从 Stage 1 开始

### 重训各阶段的输入

```
                        Stage 1 (不需要重训)
                        已有: 20250410_063030_g1_deepmimic_mcpt
                                      ↓ resume
┌─────────────────────────────────────┐
│ Stage 2: 地形 RL (需重训)            │
│                                     │
│ 输入1: Stage 1 checkpoint (已有)     │
│ 输入2: 合并数据配置                   │
│   - 原 123 motions (楼梯/走路)       │
│   + sitting_standing 1 motion (新增) │
│   = 124 motions 的新 YAML           │
│                                     │
│ 输出: 新的 teacher checkpoint        │
└─────────────────────────────────────┘
                                      ↓ policy_to_clone (DAgger 蒸馏)
┌─────────────────────────────────────┐
│ Stage 3: 蒸馏 (需重训)               │
│                                     │
│ 输入1: 新 Stage 2 的 teacher        │
│ 输入2: 同一份 124 motions 数据配置    │
│                                     │
│ 输出: 新的无历史学生 checkpoint       │
└─────────────────────────────────────┘
                                      ↓ resume
┌─────────────────────────────────────┐
│ Stage 4: RL 微调 (需重训)            │
│                                     │
│ 输入1: 新 Stage 3 的 checkpoint      │
│ 输入2: 124 motions 数据配置          │
│ 输入3: 现有 AMASS 动捕数据(同Stage1) │
│   (需修复 lafan_walk 路径问题)       │
│                                     │
│ 输出: 最终可部署策略                  │
└─────────────────────────────────────┘
```

### 每个新动作的数据需求

一个新动作需要 2 个文件（均由 real2sim pipeline 产出）：

| 文件 | 格式 | 大小 | 内容 |
|---|---|---|---|
| `retarget_poses_g1.h5` | HDF5 | ~131 KB | 机器人关节轨迹（见下表） |
| `background_mesh.obj` | Wavefront OBJ | ~690 KB | 3D 环境地形网格 |

**`retarget_poses_g1.h5` 内部结构**（以 sitting_standing 为例，55 帧）：

| 字段 | 形状 | 说明 |
|---|---|---|
| `root_pos` | (55, 3) | 根节点位置 (x, y, z) |
| `root_quat` | (55, 4) | 根节点朝向 (四元数) |
| `joints` | (55, 23) | 23 个关节角度 |
| `link_pos` | (55, 37, 3) | 37 个连杆的 3D 位置 |
| `link_quat` | (55, 37, 4) | 37 个连杆的朝向 |
| `contacts/left_foot` | (55,) | 左脚接触标志 |
| `contacts/right_foot` | (55,) | 右脚接触标志 |

**`background_mesh.obj`**: 3D 环境点云转网格（NKSR），约 10K 顶点 + 20K 面，描述机器人脚下的地形。

两个文件总计不到 1 MB，数据量很小。
