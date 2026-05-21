# IsaacGym → IsaacLab 迁移：可行性分析与技术方案

## 一、当前 IsaacGym 依赖全景

### 涉及文件与 API 使用

| 文件 | 核心依赖 |
|------|----------|
| `base_task.py` | `gymapi.acquire_gym()`, `create_sim()`, `prepare_sim()`, viewer 创建与键盘事件 |
| `legged_robot.py` | `gymtorch.wrap_tensor()`, `gymapi.AssetOptions`, URDF 加载, actor 创建, DOF/root state 设置, contact force 获取, 地面/trimesh 添加 |
| `robot_deepmimic.py` | 继承 `LeggedRobot`, 重写 reset/state, 依赖 `gymtorch.unwrap_tensor()` 进行状态回写 |
| `terrain.py` | `isaacgym.terrain_utils` — SubTerrain, heightfield→trimesh 转换 |
| `isaacgym_utils.py` | `gymutil.parse_device_str()`, `gymapi.SIM_PHYSX` |
| `train.py`/`play.py` | 入口脚本, 环境实例化 |

### 关键 IsaacGym API 调用分类

**1. 仿真生命周期** (3 处)
- `gymapi.acquire_gym()` → `create_sim()` → `prepare_sim()` → `simulate()` → `fetch_results()`

**2. 资产与环境** (6 处)
- `load_asset()` + `AssetOptions` (密度、阻尼、armature 等 15+ 参数)
- `create_env()` → `create_actor()` (每个 env 独立创建, 含 per-env 随机化)
- `set_actor_rigid_shape_properties()`, `set_actor_dof_properties()`, `set_actor_rigid_body_properties()`

**3. Tensor API** (核心, 约 20 处)
- `acquire_*_tensor()` + `gymtorch.wrap_tensor()` → 所有 GPU 状态读取
- `set_dof_state_tensor_indexed()`, `set_actor_root_state_tensor_indexed()` → 状态回写
- `set_dof_actuation_force_tensor()`, `set_dof_position_target_tensor()` → 控制输入
- `refresh_*_tensor()` → 手动同步

**4. 地形** (2 处)
- `add_ground()`, `add_triangle_mesh()` + `TriangleMeshParams`
- `isaacgym.terrain_utils` 的 procedural terrain 生成

**5. 渲染/Viewer** (4 处)
- `create_viewer()`, `subscribe_viewer_keyboard_event()`, `step_graphics()`, `draw_viewer()`

**6. Contact** (1 处)
- `acquire_net_contact_force_tensor()` → 全局接触力

---

## 二、IsaacLab 概况与关键差异

### IsaacLab 是什么

IsaacLab (原 Isaac Orbit) 是 NVIDIA 基于 Isaac Sim 构建的 RL 框架, 目前已取代 IsaacGym 成为官方推荐方案。它基于 Omniverse/Isaac Sim。注意: 当前 IsaacLab 官方 API 使用 `isaaclab.*` 命名空间, 旧资料中的 `omni.isaac.lab.*` 已不适合作为新迁移代码的 import 依据。

### 核心 API 映射

| IsaacGym | IsaacLab | 变化程度 |
|----------|----------|----------|
| `gymapi.acquire_gym()` + `create_sim()` | `gym.make()` / `ManagerBasedRLEnv` / `DirectRLEnv` | **高** — 完全不同的初始化模式 |
| `gymtorch.wrap_tensor()` | `scene["robot"].data.*`, env buffers | **中** — 数据直接在 asset/env 对象上, 但字段 shape/坐标系需逐项对齐 |
| `load_asset()` + `AssetOptions` | `isaaclab.assets.ArticulationCfg` + dataclass 配置 | **高** — 声明式配置替代命令式 API |
| `create_env()` + `create_actor()` | `ManagerBasedEnv` 自动处理 | **高** — 批量创建, 无逐 env 循环 |
| `set_dof_state_tensor_indexed()` | `scene["robot"].write_root_state_to_sim()` | **中** — 方法名和调用模式不同 |
| `isaacgym.terrain_utils` | `isaaclab.terrains` | **中** — 接口相似但重写 |
| Viewer + 键盘事件 | Isaac Sim Window / headless web viewer | **中** — 不同渲染管线 |
| `acquire_net_contact_force_tensor()` | `scene["robot"].data.net_contact_forces_w` | **低** — 数据更易访问 |
| Domain Randomization | 手动 → 内置 `EventManager` | **利好** — 原生支持 |

### 架构差异

IsaacLab 的核心架构变化：

- **从命令式到声明式**: IsaacGym 在 Python 代码中逐步构建环境; IsaacLab 使用 dataclass 配置 (Cfg) 声明环境, 框架自动构建
- **从单文件到 Manager 模式**: 观测、奖励、动作、事件各自由 Manager 管理, 可组合
- **从逐 env 循环到批量操作**: IsaacGym 的 `for i in range(num_envs): create_actor()` 在 IsaacLab 中不需要
- **训练循环**: IsaacLab 内置 `RLTaskEnv` + `ManagerBasedRLEnv`, 可直接对接 `rsl_rl`、`skrl`、`rl_games`

---

## 三、可行性评估

### 可行性: MVP 中等偏高, 完整替换风险中高

结论: 迁移方向可行, 但不应按“替换 IsaacGym API”的方式直接改写当前 `BaseTask`/`LeggedRobot`。更稳妥的路线是在 IsaacLab 中新建 simulation backend, 先跑通 G1 最小任务, 再逐步接入 DeepMimic、terrain、sensor、training pipeline。当前 IsaacGym 实现必须保留为 baseline, 用于行为和性能对齐。

| 维度 | 评估 | 说明 |
|------|------|------|
| **物理引擎兼容性** | ⚠️ 中 | 都是 PhysX 系列, 但接触、关节限位、PD/effort 控制、reset 初始速度和随机 push 需要实测对齐 |
| **Tensor 数据等价性** | ⚠️ 中 | root/joint/body/contact 数据都有对应来源, 但 shape、frame、刷新时机和 env-local/world 坐标需要逐项验收 |
| **URDF/MJCF 支持** | ✅ 高 | IsaacLab 完整支持 |
| **Contact Force** | ⚠️ 中 | API 更简洁, 但当前 termination/reward/contact matching 对阈值和 body index 敏感 |
| **Terrain 生成** | ⚠️ 中 | 需要重写, 但 IsaacLab 有内置 terrain 生成器 |
| **Raycaster 传感器** | ❌ 低 | 当前自定义 Warp raycaster 需要完全重写、适配 `RayCasterCfg`, 或封装为 IsaacLab sensor |
| **DeepMimic 框架** | ⚠️ 中高 | ReplayDataLoader 可复用, 但 reward/obs/contact/terrain offset 与仿真状态字段强耦合 |
| **Domain Randomization** | ✅ 高 | IsaacLab EventManager 原生支持, 可能更简洁 |
| **Viser 可视化** | ⚠️ 中低 | Viser web viewer 可保留, 但需要 state adapter 提供 `root_states`、`dof_pos`、`rigid_body_pos`、`contact_forces` 等旧字段 |
| **rsl_rl 集成** | ⚠️ 中 | IsaacLab 原生支持 rsl_rl, 但接口需要适配 |

### 主要风险

1. **自定义 Raycaster 传感器**: 当前 `DepthCameraSensor`、`HeightfieldSensor`、`MultiLinkHeightSensor` 是基于 Warp mesh 和 rigid body state tensor 的自定义实现。迁移不能只替换为 Camera sensor, 必须验证输出 shape、单位、坐标系、噪声、延迟、update frequency、uint8/float 归一化是否一致。
2. **DeepMimic 坐标系与 terrain offset**: 当前 `RobotDeepMimic` 通过 `env_offsets` 在 world frame 与 env frame 间转换, 并把 replay root/link/contact 与 terrain mesh 对齐。IsaacLab terrain/env origin 机制不同, 这是训练质量风险而不是简单 API 风险。
3. **Per-env 属性随机化**: IsaacGym 用循环逐 env 设置不同 friction/mass/COM/DOF friction/collision filter; IsaacLab 支持随机化, 但需要用 EventManager 或 asset API 重建相同行为。
4. **物理行为和性能回归**: IsaacLab 基于 Isaac Sim, 启动、渲染、内存和大规模 parallel env 开销通常高于 IsaacGym; 4096+ env 的 FPS 和显存必须作为 gate。
5. **训练与 checkpoint 兼容**: 模型权重可能可复用, 但 observation dictionary、normalization、teacher obs、history obs、contact threshold 变化都可能导致 checkpoint 不能直接接续训练。
6. **多 GPU 训练**: 当前用 `torchrun`; IsaacLab 也支持多 GPU, 但启动脚本、配置传递和 runner 对接方式不同。

---

## 四、技术方案

### 阶段 0: 环境准备与 API 固化 (2-4 天)

- 安装 IsaacLab (依赖 Isaac Sim/Omniverse, 版本需固定)
- 以当前官方 `isaaclab.*` API 为准, 不使用旧 `omni.isaac.lab.*` import
- 验证 G1 URDF 在 IsaacLab 中正确加载
- 搭建最小可运行 demo: 单个 G1 站立
- 记录 G1 joint names、body names、default joint pose、joint limits、actuator drive mode 与 IsaacGym baseline 的 diff

### 阶段 1: 核心环境迁移 (2-3 周)

将 `base_task.py` + `legged_robot.py` 的核心仿真逻辑迁移到 IsaacLab 的 `ManagerBasedRLEnv` 模式。

**1.1 配置声明**

```python
# 旧: 命令式
asset_options = gymapi.AssetOptions()
asset_options.density = cfg.asset.density
...

# 新: 声明式
@configclass
class G1AssetCfg(AssetBaseCfg):
    prim_path = "{ENV_REGEX_NS}/Robot"
    spawn = UsdFileCfg(
        usd_path=urdf_to_usd(cfg.asset.file),  # 或直接使用 UrdfFileCfg, 取决于 IsaacLab 版本和资产格式
        rigid_props=RigidBodyPropertiesCfg(density=cfg.asset.density, ...),
        articulation_props=ArticulationRootPropertiesCfg(...),
    )
```

**1.2 环境构建**

```python
# 旧: 逐 env 创建
for i in range(self.num_envs):
    env_handle = self.gym.create_env(...)
    actor_handle = self.gym.create_actor(env_handle, ...)
    self.gym.set_actor_rigid_shape_properties(...)

# 新: 批量, 配置驱动
@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    scene = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations = ObservationsCfg(...)  # Manager-based
    actions = ActionsCfg(...)
    rewards = RewardsCfg(...)
    events = EventsCfg(...)  # Domain randomization
```

**1.3 Tensor 访问**

```python
# 旧
root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
self.root_states = gymtorch.wrap_tensor(root_state)

# 新
self.root_states = self.scene["robot"].data.root_pos_w  # 直接访问
self.dof_pos = self.scene["robot"].data.joint_pos
self.contact_forces = self.scene["robot"].data.net_contact_forces_w
```

**1.4 状态回写**

```python
# 旧
self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), ...)

# 新
self.scene["robot"].write_root_state_to_sim(root_states, env_ids)
self.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel, env_ids)
```

### 阶段 2: DeepMimic 框架迁移 (2-4 周)

`robot_deepmimic.py` 中的 `ReplayDataLoader` 和部分数学函数可以复用, 但不能假设所有 reward/obs 直接无改动迁移。当前 DeepMimic 逻辑依赖 IsaacGym 风格状态字段、body index、contact force、sensor output、terrain offset。需要先做 state adapter, 再逐项迁移。

| 需重写 | 可复用 |
|--------|--------|
| `_reset_dofs()` 中的 `gym.set_dof_state_tensor_indexed` | `ReplayDataLoader` 数据读取、clip sampling、episode index 管理 |
| `_reset_root_states()` 中的 `gym.set_actor_root_state_tensor_indexed` | quaternion/math helper 函数 |
| `viz_replay_data()` 中的 state 回写 | 与仿真状态无关的数据预处理 |
| Terrain offset 管理 (与 IsaacLab terrain/env origins 对接) | 部分 tracking reward 公式, 但输入字段需重新验证 |
| body/link/contact index 解析 | contact matching 目标数据, 但阈值和 actual contact 需要重新标定 |

阶段 2 的验收标准:

- 同一 replay clip、同一 reset seed 下, IsaacLab 的 root/joint/body 初始状态与 IsaacGym baseline 数值接近。
- `env_frame_to_world_frame()` / `world_frame_to_env_frame()` 等价性通过单元测试。
- `deepmimic`、`torso_xy_rel`、`torso_yaw_rel`、`target_joints` 等关键 obs 输出 shape 完全一致, 数值差异有解释。
- contact reward/termination 在固定 replay 片段上的触发帧与 IsaacGym baseline 可对齐或有明确阈值调整记录。

### 阶段 3: 传感器迁移 (1-2 周)

当前自定义 raycaster 传感器需要适配:

| 当前传感器 | IsaacLab 替代方案 |
|-----------|-------------------|
| `DepthCameraSensor` (自定义 raycaster) | `isaaclab.sensors.CameraCfg` (基于渲染的深度) 或 `RayCasterCfg` |
| `HeightfieldSensor` (自定义 raycaster) | `RayCasterCfg` 或自定义 `SensorBase` 子类 |
| `MultiLinkHeightSensor` | `RayCasterCfg` + multi-origin |

**关键决策**: IsaacLab 的 Camera 是基于渲染的 (更真实但更慢), 而当前 raycaster 是纯数学计算 (更快但不渲染图像)。如果只需要深度图用于 RL 观测, 建议使用 `RayCasterCfg`; 如果需要 RGB 图像, 使用 `CameraCfg`。

传感器迁移必须先定义兼容层:

- 输出 shape 与 dtype: `DepthCameraSensor` 当前为 `[num_envs, H, W]` float, `HeightfieldSensor` 可为 uint8 或 float, `MultiLinkHeightSensor` 为 `[num_envs, num_links]`。
- 坐标系: 当前 ray origin 来自 `rigid_body_pos/quat`, 可选 `only_heading`。
- 噪声与时序: 当前支持 orientation noise、white noise、bad distance、offset noise、max delay、随机 update frequency。
- mesh 来源: 当前 raycast mesh 直接使用 `terrain.vertices/triangles`, IsaacLab 版本需要确认能否直接对 imported mesh 做 GPU raycast, 或保留 Warp raycast 作为自定义 sensor。

### 阶段 4: 训练管线适配 (1 周)

- 将 `videomimic_rl/rsl_rl` 适配到 IsaacLab 的 `rsl_rl`/Gymnasium env 接口
- IsaacLab 已内置 `OnPolicyRunnerCfg` 集成, 需要适配 4 阶段训练脚本
- 验证 checkpoint 兼容性 (模型权重通用, 但 env 依赖的 obs normalization 需要重新计算)
- 验证 dict observation、teacher observation、history observation 和 privileged observation 在 runner 中的接口兼容性

### 阶段 5: 地形系统迁移 (1 周)

- `isaacgym.terrain_utils` → `isaaclab.terrains`
- `DeepMimicTerrain` (自定义, 加载真实场景 mesh) → 需要自定义 `TerrainImporterCfg` 加载外部 mesh
- Curriculum terrain 需要用 IsaacLab 的 `SubTerrainCfg` 重写

### 阶段 5.5: Viser Web 可视化适配 (3-5 天)

当前运行时可通过 Viser web 查看仿真, Viser 本身可以保留, 但它读取的是 IsaacGym 风格 robot 状态字段。迁移后需要提供 simulator-neutral state adapter:

| 当前 Viser 依赖字段 | IsaacLab adapter 来源 |
|--------------------|----------------------|
| `root_states` | `scene["robot"].data.root_state_w` 或 root pos/quat/vel 拼接 |
| `dof_pos` | `scene["robot"].data.joint_pos` |
| `rigid_body_pos`, `rigid_body_quat` | `scene["robot"].data.body_pos_w`, `body_quat_w` |
| `contact_forces` | contact sensor 或 articulation contact data |
| `body_names`, `dof_names` | articulation metadata / asset joint-body name mapping |

建议把 `init_isaacgym_robot()` 改名或包一层为 `init_robot_state_source()`, 避免 Viser 与 IsaacGym 命名继续耦合。验收标准是: IsaacLab 单 env 和多 env 下, Viser 能显示 robot pose、joint pose、contact force 和 replay keypoints, 且不依赖 IsaacGym viewer。

### 阶段 6: 验证与对齐 (1-2 周)

- 对比 IsaacGym vs IsaacLab 的训练曲线
- 验证物理行为一致性 (接触力、关节限位、阻尼)
- 性能基准测试 (FPS, GPU 显存)
- Sim-to-real 迁移验证 (在 real2sim 数据上对比策略质量)

最低验收 gate:

1. **Asset gate**: G1 URDF/USD 加载成功, joint/body names、joint limits、default pose、actuator mode 与 IsaacGym baseline 对齐。
2. **State gate**: reset 后 root/joint/body/contact tensor shape 与旧环境一致, 坐标系差异有明确 adapter。
3. **Sensor gate**: depth/height/link height 输出 shape、dtype、数值范围、噪声、延迟与旧环境一致或有显式变更。
4. **DeepMimic gate**: 同一 replay clip 的关键 obs/reward/contact 在固定 seed 下可 diff。
5. **Performance gate**: 记录 1/128/1024/4096 env 的 FPS、step time、显存、启动时间, 与 IsaacGym baseline 比较。
6. **Training gate**: 至少一个 flat walking 和一个 DeepMimic 任务能完成短训 smoke test, reward 曲线没有明显退化。
7. **Viser gate**: Web viewer 可显示 IsaacLab state, 不依赖 IsaacGym viewer。

---

## 五、工作量与时间线总结

| 阶段 | 工作量 | 风险 |
|------|--------|------|
| 0. 环境准备与 API 固化 | 2-4 天 | 低 |
| 1. 核心环境迁移 | 2-3 周 | 中 — API 差异大 |
| 2. DeepMimic 框架 | 2-4 周 | 中高 — obs/reward/contact/terrain offset 与状态字段强耦合 |
| 3. 传感器迁移 | 1-2 周 | **高** — raycaster 重写 |
| 4. 训练管线 | 1 周 | 中 — 需要适配 |
| 5. 地形系统 | 1 周 | 中 — DeepMimicTerrain 自定义 |
| 5.5. Viser Web 可视化 | 3-5 天 | 中低 — 需要 state adapter |
| 6. 验证对齐 | 1-2 周 | 中 — 行为差异排查 |
| **MVP 总计** | **8-12 周** | 单 G1 + 基础训练 + 部分传感器 |
| **完整替换总计** | **12-20 周** | 四阶段训练、DeepMimic、terrain、传感器、性能和 sim-to-real 全部对齐 |

---

## 六、建议

1. **渐进式迁移**: 不要试图一次性迁移全部代码。先在 IsaacLab 中跑通最简单的 G1 walking 任务, 再逐步添加 DeepMimic、传感器、地形等
2. **保留 IsaacGym 分支**: 迁移期间保持双轨运行, 确保可以随时回退
3. **传感器是最大风险**: 建议优先验证 `RayCasterCfg` 能否满足 depth/heightfield 需求, 如果不行需要保留自定义实现
4. **先做 state adapter**: 不要让 Viser、DeepMimic、sensor、runner 同时直接依赖 IsaacLab 内部字段。先提供兼容旧字段的 adapter, 再逐步重构。
5. **考虑 IsaacLab 的 Manager 模式红利**: 迁移后的代码会更模块化、更易维护; Domain Randomization 从手动循环变成声明式配置, 长期收益大
6. **性能验证很关键**: IsaacLab 基于 Isaac Sim, 比 IsaacGym 重; 在大规模并行训练 (4096+ envs) 下可能有性能损失, 需要实测
7. **不要默认 checkpoint 可续训**: 权重结构可能兼容, 但 observation normalization、history、teacher obs 和 contact/sensor 分布变化后, 更现实的目标是迁移策略结构并重新标定/再训练。
