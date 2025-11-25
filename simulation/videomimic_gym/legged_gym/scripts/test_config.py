import isaacgym
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgV2
# from legged_gym.envs.g1.g1_deepmimic_config import G1DeepMimicCfg, G1DeepMimicCfgV2
from legged_gym.envs.base.legged_robot_config import  LeggedRobotCfg as LeggedNew
from legged_gym.envs.base.legged_robot_config_old import  LeggedRobotCfg as LeggedOld
from legged_gym.envs.base.base_config_old import class_to_dict

def compare_dicts_recursive(dict1, dict2, path=""):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    only_in_first = []
    only_in_second = []
    value_changes = []
    
    # Keys only in first dict
    for k in (keys1 - keys2):
        only_in_first.append((f"{path}.{k}" if path else k, dict1[k]))
    
    # Keys only in second dict
    for k in (keys2 - keys1):
        only_in_second.append((f"{path}.{k}" if path else k, dict2[k]))
    
    # Compare common keys
    for k in (keys1 & keys2):
        new_path = f"{path}.{k}" if path else k
        if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
            # Recursive call for nested dicts
            sub_first, sub_second, sub_changes = compare_dicts_recursive(dict1[k], dict2[k], new_path)
            only_in_first.extend(sub_first)
            only_in_second.extend(sub_second)
            value_changes.extend(sub_changes)
        elif dict1[k] != dict2[k]:
            value_changes.append((new_path, dict1[k], dict2[k]))
    
    return only_in_first, only_in_second, value_changes

if __name__ == "__main__":
    cfg = class_to_dict(LeggedNew())
    cfg2 = class_to_dict(LeggedOld())
    # cfg = class_to_dict(LeggedRobotCfg())
    # cfg2 = LeggedRobotCfgV2().to_dict()
    # cfg = class_to_dict(G1DeepMimicCfg())
    # cfg2 = G1DeepMimicCfgV2().to_dict()
    print("=== Differences between LeggedRobotCfg and LeggedRobotCfgV2 ===")
    only_in_first, only_in_second, value_changes = compare_dicts_recursive(cfg, cfg2)
    
    if only_in_first or only_in_second or value_changes:
        if only_in_first:
            print("\nKeys only in LeggedRobotCfg:")
            for path, value in sorted(only_in_first):
                print(f"  - {path}: {value}")
                
        if only_in_second:
            print("\nKeys only in LeggedRobotCfgV2:")
            for path, value in sorted(only_in_second):
                print(f"  - {path}: {value}")
                
        if value_changes:
            print("\nKeys with different values:")
            for path, val1, val2 in sorted(value_changes):
                print(f"  - {path}:")
                print(f"    LeggedRobotCfg   : {val1}")
                print(f"    LeggedRobotCfgV2 : {val2}")
    else:
        print("No differences found between the configurations.")
