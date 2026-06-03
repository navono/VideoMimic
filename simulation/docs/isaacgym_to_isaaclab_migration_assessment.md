# IsaacGym → IsaacLab 迁移可行性评估

## 结论

迁移**可行**，推荐使用 **DirectRLEnv** 路径（而非 ManagerBasedRLEnv），预估工作量 **8-12 周** 完成 MVP（sitting/standing 单任务训练跑通），**16-20 周** 完成完整 4 阶段训练管线对齐。

**背景:** H100 (sm_90) 不支持 IsaacGym Preview 4 的 GPU PhysX（`libPhysXGpu_64.so` 只编译了 sm_80/sm_86 的 cubin），且 IsaacGym 已停止维护。Isaac Lab v2.3+ 支持 H100，内置 G1 velocity locomotion 配置可作为参考，但 VideoMimic 的 DeepMimic 框架远超标准 locomotion，需要大量自定义代码。

---

## 一、代码规模与 IsaacGym 依赖度

| 类别 | 代码量 | IsaacGym 依赖程度 | 迁移难度 |
|------|--------|-------------------|----------|
| `base_task.py` (仿真生命周期) | 137 行 | **极高** — gymapi.acquire_gym, create_sim, prepare_sim, viewer | 中 |
| `legged_robot.py` (核心环境) | 1516 行 | **极高** — 40+ gymapi/gymtorch 调用 | 高 |
| `robot_deepmimic.py` (DeepMimic) | 1039 行 | **高** — gymtorch.unwrap_tensor 写回状态 | 中高 |
| `g1_deepmimic.py` (G1 特化) | 513 行 | **中** — 继承链间接依赖 | 中 |
| `g1_config.py` / `g1_deepmimic_config.py` | 1566 行 | **低** — 纯配置 dataclass | 低 |
| `terrain.py` (地形生成) | 157 行 | **中** — isaacgym.terrain_utils | 中 |
| `deepmimic_terrain.py` | 369 行 | **低** — numpy only, 不依赖 gymapi | 低 |
| `helpers.py` / `isaacgym_utils.py` | 480 行 | **中** — gymapi.SimParams, gymutil | 中 |
| `viser_visualizer.py` | 1628 行 | **中低** — 读取 torch tensor 状态，需要 adapter | 中低 |
| `raycaster/` (传感器) | 1147 行 | **低** — Warp + torch, 不用 gymapi | 低 |
| `rsl_rl/` (RL 算法) | 2171 行 | **极低** — 只依赖 env 接口，不依赖 gym | 低 |
| `replay_data.py` (数据加载) | 1223 行 | **极低** — numpy/torch only | 极低 |
| **总计** | **~10K 行** | | |

**可直接复用的代码 (~4.5K 行，45%)**: rsl_rl, replay_data, raycaster, deepmimic_terrain, 配置 dataclass

**需要重写的代码 (~5.5K 行，55%)**: base_task, legged_robot (仿真交互部分), robot_deepmimic (状态读写), helpers, terrain

---

## 二、IsaacGym API 依赖清单

### 仿真生命周期 (6 处)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gymapi.acquire_gym()` | base_task:12 | `SimulationContext(cfg)` | 低 |
| `gym.create_sim()` | legged_robot:363 | 自动 via config | 低 |
| `gym.prepare_sim()` | base_task:59 | 自动 via SceneCfg | 低 |
| `gym.simulate()` | legged_robot:140 | `sim.step(render=False)` | 低 |
| `gym.fetch_results()` | base_task:129, legged_robot:148 | `scene.update(dt)` 自动 | 低 |
| `gym.create_viewer()` | base_task:68-73 | Isaac Sim viewport / headless | 低 |

### Tensor API — 状态读取 (7 处)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gym.acquire_actor_root_state_tensor()` | legged_robot:768 | `articulation.data.root_pos_w`, `root_quat_w` 等 | 中 |
| `gym.acquire_rigid_body_state_tensor()` | legged_robot:769, g1_env:37 | `articulation.data.body_pos_w` 等 | 中 |
| `gym.acquire_dof_state_tensor()` | legged_robot:770 | `articulation.data.joint_pos`, `joint_vel` | 低 |
| `gym.acquire_net_contact_force_tensor()` | legged_robot:771 | `ContactSensor.data.net_forces_w` | 中 |
| `gymtorch.wrap_tensor()` | legged_robot:777-788 | 直接从 articulation/sensor 读取 | 中 |
| `gym.refresh_*_tensor()` | legged_robot:172-174 | `scene.update(dt)` 自动同步 | 低 |
| `gym.refresh_rigid_body_state_tensor()` | g1_env:50 | `scene.update(dt)` 自动同步 | 低 |

### Tensor API — 状态写入 (6 处, 关键!)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gym.set_actor_root_state_tensor_indexed()` | legged_robot:692-694, 721-723, robot_deepmimic:307-309, 183-185, 1037-1039 | `articulation.write_root_state_to_sim(root_state, env_ids)` | **高** — 需逐项对齐 root_state 的 13 维格式 |
| `gym.set_dof_state_tensor_indexed()` | legged_robot:671-673, robot_deepmimic:286-288 | `articulation.write_joint_state_to_sim(pos, vel, env_ids)` | **高** — 同上，需对齐格式 |
| `gym.set_dof_actuation_force_tensor()` | legged_robot:136 | `JointEffortAction` 或直接写 | 中 |
| `gym.set_dof_position_target_tensor()` | legged_robot:139 | `JointPositionAction` 或 `ImplicitActuator` | 中 |

### 资产与环境创建 (10+ 处)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gym.load_asset()` | legged_robot:961 | `ArticulationCfg(prim_path, usd_path)` | 低 |
| `gymapi.AssetOptions` (15+ 参数) | legged_robot:946-959 | `ArticulationCfg` 属性 | 中 |
| `gym.create_env()` | legged_robot:1019 | 自动 via SceneCfg | 低 |
| `gym.create_actor()` | legged_robot:1026 | 自动 via SceneCfg | 低 |
| `gym.set_actor_rigid_shape_properties()` | legged_robot:1027 | EventManager `mode="prestartup"` | 中 |
| `gym.set_actor_dof_properties()` | legged_robot:1029 | `ActuatorCfg` | 中 |
| `gym.set_actor_rigid_body_properties()` | legged_robot:1032 | EventManager `mode="prestartup"` | 中 |
| `gym.find_actor_rigid_body_handle()` | legged_robot:1038-1046 | body name mapping | 低 |
| `gymapi.AssetOptions.DOF_MODE_POS/EFFORT` | legged_robot:987 | `ImplicitActuatorCfg` / `ExplicitActuatorCfg` | 中 |
| per-env 循环创建 (1026-1034) | legged_robot:1017-1034 | 批量自动创建, 不需要逐 env 循环 | **利好** |

### 地形 (3 处)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gym.add_ground()` + `PlaneParams` | legged_robot:893-898 | `GroundPlaneCfg` | 低 |
| `gym.add_triangle_mesh()` + `TriangleMeshParams` | legged_robot:906-922 | 自定义 `TerrainImporterCfg` | **高** — DeepMimicTerrain mesh 加载需自定义 |
| `isaacgym.terrain_utils` | terrain.py:5, 40-43, 70-77, 95-109 | `isaaclab.terrains` | 中 |

### 渲染/Viewer (4 处)
| API | 调用位置 | IsaacLab 对应 | 难度 |
|------|---------|--------------|------|
| `gym.create_viewer()` | base_task:68 | Isaac Sim viewport | 低 |
| `gym.subscribe_viewer_keyboard_event()` | base_task:70-73 | 不迁移, 用 Viser 替代 | 低 |
| `gym.viewer_camera_look_at()` | legged_robot:382-384 | viewport API | 低 |
| `gym.step_graphics/draw_viewer()` | base_task:133-134 | Isaac Sim 自动 | 低 |

---

## 三、关键风险与难点

### 1. root_state / dof_state 格式对齐 (风险: 高)

IsaacGym 的 root_state 是 13 维 `(pos_xyz, quat_wxyz, lin_vel_xyz, ang_vel_xyz)` flat tensor, indexed by env_ids。IsaacLab 没有统一的 13 维 root_state tensor，而是分散在 `root_pos_w`, `root_quat_w`, `lin_vel_w`, `ang_vel_w` 等属性中。

VideoMimic 的 DeepMimic reset 逻辑大量使用 `root_states[env_ids, :3]`, `root_states[env_ids, 3:7]` 等 slice 操作。迁移时需要：
- 创建一个 adapter 把 IsaacLab 的分散属性组合回 13 维格式
- 或重写所有 slice 操作改为独立属性操作

推荐方案：创建 `DeepMimicStateAdapter` 提供 13 维兼容格式，避免逐行修改 DeepMimic 代码。

### 2. DeepMimic 地形 mesh 加载 (风险: 中高)

IsaacGym 用 `gym.add_triangle_mesh()` 直接从 numpy vertices/triangles 加载地形。Isaac Lab 的地形系统是声明式的 (`TerrainImporterCfg`)，不支持直接传入 numpy mesh。

需要：
- 把 DeepMimicTerrain 的 vertices/triangles 保存为 USD/OBJ 文件
- 或自定义 `TerrainImporterCfg` 子类支持 numpy mesh
- Isaac Lab 的 `RayCasterCfg` 对自定义 mesh 的 raycast 需要注册 mesh 为可 raycast 的 prim

### 3. 自定义 Raycaster 传感器 (风险: 中)

当前 raycaster (DepthCameraSensor, HeightfieldSensor, MultiLinkHeightSensor) 基于 Warp mesh raycast，不依赖 IsaacGym。理论上可以直接复用，但需要：
- 验证 rigid body state 数据源从 IsaacGym tensor 改为 IsaacLab `articulation.data.body_pos_w` 后格式一致
- raycast mesh 来源从 IsaacGym terrain vertices 改为 Isaac Lab mesh prim 注册

### 4. DAgger / Teacher-Student 蒸馏 (风险: 中高)

videomimic_rl 是 fork 版 rsl_rl，含 `policy_to_clone`, BC loss, teacher obs, history obs 等自定义逻辑。Isaac Lab 内置 rsl_rl 集成，但接口不同：
- `policy_to_clone` 加载 frozen teacher checkpoint → 需适配 Isaac Lab 的 runner 配置
- BC loss 开关 (Stage 3 开, Stage 4 关) → 需在 Isaac Lab runner 中复刻
- teacher obs / student obs 切换 → 需适配 Isaac Lab 的 observation pipeline
- history obs (`HistoryHandler`) → Isaac Lab 内置 `history_length` 但接口不同

### 5. Domain Randomization (风险: 中)

IsaacGym 用逐 env 循环随机化 (摩擦/质量/COM/PD增益/DOF摩擦/collision filter)。Isaac Lab 有 `EventManager` 原生支持，但需要用声明式配置重建相同行为。碰撞 filter (`dont_collide_groups`) 在 Isaac Lab 中需要 USD-level collision group 配置。

### 6. 性能回归 (风险: 中)

Isaac Lab 基于 Isaac Sim，比 IsaacGym 重。在 4096+ parallel env 下：
- FPS 可能低于 IsaacGym (Isaac Sim 启动慢, 渲染开销大)
- 显存占用更高
- 需实测 gate: 1/128/1024/4096 env 的 FPS 和显存

---

## 四、推荐迁移策略: DirectRLEnv 路径

### 为什么选 DirectRLEnv 而非 ManagerBasedRLEnv

| 因素 | DirectRLEnv | ManagerBasedRLEnv |
|------|-------------|-------------------|
| 与现有代码结构相似度 | **高** — `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones` 与 LeggedRobot/RobotDeepMimic 的 step/post_physics_step 模式一致 | 低 — 需把所有逻辑拆成独立 term |
| DeepMimic 自定义逻辑保留 | **容易** — 可直接把 DeepMimic reward/obs/contact 逻辑放在 `_get_rewards`/`_get_observations` 中 | **困难** — 需拆成几十个 RewardTermCfg/ObservationTermCfg |
| 迁移工作量 | 8-12 周 MVP | 12-16 周 MVP |
| 后续重构空间 | 后续可渐进式迁移到 Manager 模式 | 一步到位 |

### 分阶段计划

**阶段 0: 环境准备 (3-5 天)**
- 安装 Isaac Lab v2.3+ (支持 H100, Python 3.11, PyTorch 2.7+cu128)
- 将 G1 URDF 转换为 USD (Isaac Lab 需要 USD 格式)
- 验证 G1 USD 加载、joint/body names、joint limits 与 IsaacGym baseline 对齐
- Gate: `articulation.data.joint_pos` shape 和数值范围与旧环境一致

**阶段 1: 核心环境迁移 (2-3 周)**
- 创建 `G1DeepMimicDirectEnv(DirectRLEnv)`
- 实现仿真生命周期: `_setup_scene`, `_pre_physics_step`, `_step_impl`
- 实现状态读取: 从 `scene["robot"].data.*` 读取 root/joint/body/contact
- 创建 `DeepMimicStateAdapter` 提供 13 维 root_state 和 flat tensor 兼容格式
- 实现 PD controller / DEEPMIMIC_DELTA controller
- Gate: 单 env flat terrain 下 G1 能站立不倒

**阶段 2: DeepMimic 框架移植 (2-3 周)**
- 移植 `ReplayDataLoader` (可直接复用, ~0 改动)
- 移植 `DeepMimicTerrain` mesh 加载 → 自定义 TerrainImporter + mesh 文件保存
- 移植 `env_offsets` 和 env/world frame 转换逻辑
- 移植所有 DeepMimic reward 函数 (15+ 个, 大部分是纯 torch 运算, 可直接复用)
- 移植所有 DeepMimic observation 函数
- 移植 contact matching / termination / link_pos_error
- Gate: sitting/standing motion clip 的 reward 曲线与 IsaacGym baseline 可 diff (固定 seed)

**阶段 3: 传感器移植 (1 周)**
- 移植 raycaster 传感器 (Warp + torch, 改数据源为 Isaac Lab)
- 或改用 Isaac Lab 内置 `RayCasterCfg` (需验证输出一致)
- 移植 MultiLinkHeightSensor
- Gate: depth/height/link_height 输出 shape/dtype 与旧环境一致

**阶段 4: 训练管线适配 (1.5-2 周)**
- 适配 videomimic_rl 的 rsl_rl 到 Isaac Lab 的 env 接口
- 实现 DAgger teacher-student 蒸馏 (policy_to_clone 加载, BC loss 开关)
- 适配 history obs pipeline
- 适配 4 阶段训练脚本
- Gate: sitting/standing distillation 能跑通至少 100 iterations

**阶段 5: Domain Randomization + 验证 (1-2 周)**
- 用 Isaac Lab `EventManager` 重建 friction/mass/COM/PD/DOF/collision randomization
- 性能基准: FPS/显存 vs IsaacGym baseline
- 对齐 reward 曲线和物理行为
- Gate: 训练结果无明显退化

**阶段 5.5: Viser 可视化 (3-5 天)**
- 创建 state adapter 让 Viser 读取 Isaac Lab 状态
- 验证 Viser 能显示 robot pose/joint/contact/replay keypoints

---

## 五、工作量总结

| 阶段 | 工作量 | 风险 | 可并行 |
|------|--------|------|--------|
| 0. 环境准备 | 3-5 天 | 低 | — |
| 1. 核心环境 | 2-3 周 | 中 | 与阶段 0 串行 |
| 2. DeepMimic | 2-3 周 | 中高 | 与阶段 1 串行 |
| 3. 传感器 | 1 周 | 中 | 可与 2 并行 |
| 4. 训练管线 | 1.5-2 周 | 中高 | 与 2/3 串行 |
| 5. Randomization + 验证 | 1-2 周 | 中 | — |
| 5.5. Viser | 3-5 天 | 低 | 可与 5 并行 |
| **MVP (sitting/standing 单任务)** | **8-12 周** | | |
| **完整 4 阶段管线** | **16-20 周** | | |

---

## 六、Isaac Lab 现有资源可借鉴

- **G1 velocity locomotion 配置**: Isaac Lab 已内置 G1 flat/rough locomotion env, 可作为 URDF/USD 转换和基本 articulation 设置的参考
- **G1 locomanipulation (v2.3+)**: 上下半身分离控制, 可参考 upper body joint 配置
- **ContactSensor**: 比 IsaacGym 的 net_contact_force_tensor 更简洁, 支持按 body name 过滤
- **EventManager**: 原生 domain randomization, 比 IsaacGym 逐 env 循环更高效
- **rsl_rl 集成**: Isaac Lab 内置 rsl_rl runner, 但 videomimic_rl 的 DAgger 扩展需要适配
