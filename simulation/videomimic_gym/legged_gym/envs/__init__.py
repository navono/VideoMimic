from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_deepmimic import G1DeepMimic
from legged_gym.envs.g1.g1_deepmimic_config import G1DeepMimicCfg, G1DeepMimicCfgPPO, G1DeepMimicCfgDagger

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
task_registry.register( "g1_deepmimic_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgDagger())