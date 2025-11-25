import os
import numpy as np
import random
import torch
import argparse
from isaacgym.gymutil import parse_device_str
from isaacgym import gymapi

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def parse_arguments_modified(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    """Modefied from gymutil.parse_arguments to also allow unknown argument parsing."""
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args, unknown = parser.parse_known_args()

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args, unknown