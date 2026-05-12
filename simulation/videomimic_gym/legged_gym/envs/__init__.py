from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_deepmimic_config import G1DeepMimicCfg, G1DeepMimicCfgPPO, G1DeepMimicCfgDagger

from legged_gym.utils.task_registry import task_registry


def _import_env_classes():
    """Import environment classes lazily to avoid requiring Isaac Sim at import time."""
    from legged_gym.envs.g1.g1_env import G1RobotEnv
    from legged_gym.envs.g1.g1_deepmimic import G1DeepMimic
    return G1RobotEnv, G1DeepMimic


def _ensure_registered(task_name=None):
    """Ensure env classes are registered, importing them lazily if needed.

    Args:
        task_name: If provided, only ensure this specific task is registered.
    """
    if task_name and task_name in task_registry.task_classes:
        return

    if not task_registry.task_classes:
        G1RobotEnv, G1DeepMimic = _import_env_classes()
        task_registry.register("g1", G1RobotEnv, G1RoughCfg(), G1RoughCfgPPO())
        task_registry.register("g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
        task_registry.register("g1_deepmimic_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgDagger())

        # Also register tasks from g1_deepmimic_config that aren't already registered
        from legged_gym.envs.g1 import g1_deepmimic_config
        if hasattr(g1_deepmimic_config, '_register_all'):
            g1_deepmimic_config._register_all(G1DeepMimic)


# Try eager registration; fall back to lazy if IsaacLab is unavailable
try:
    G1RobotEnv, G1DeepMimic = _import_env_classes()

    task_registry.register("g1", G1RobotEnv, G1RoughCfg(), G1RoughCfgPPO())
    task_registry.register("g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
    task_registry.register("g1_deepmimic_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgDagger())
except ImportError:
    # IsaacLab not available (e.g. Isaac Sim not installed) — registration deferred
    # Call _ensure_registered() before using task_registry.make_env()
    pass
