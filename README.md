# VideoMimic
**CoRL 2025, Best Student Paper Award.**


[[project page]](https://www.videomimic.net/) [[arxiv]](https://arxiv.org/pdf/2505.03729)  [[proceedings]](https://proceedings.mlr.press/v305/allshire25a.html) 

**Visual Imitation Enables Contextual Humanoid Control.**
    
<div style="background-color: #333; padding: 16px 20px; border-radius: 8px; color: #eee; font-family: sans-serif; line-height: 1.6;">
  <div style="font-size: 14px; margin-bottom: 12px;">
    Arthur Allshire<sup>*</sup>, Hongsuk Choi<sup>*</sup>, Junyi Zhang<sup>*</sup>, David McAllister<sup>*</sup>, 
    Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, Angjoo Kanazawa (*Equal contribution) 
  </div>    
  <div style="font-size: 14px;">
    <i>University of California, Berkeley</i>
  </div>
</div>

## Updates

- **Sep 30, 2025:** Videomimic won the Best Student Paper Award. 
- **Sep 15, 2025:** Simulation code and preliminary sim2real code released.
- **Jul 6, 2025:** Initial real-to-sim pipeline release. 

## VideoMimic Real-to-Sim

VideoMimic’s [real-to-sim pipeline](real2sim/README.md) reconstructs 3D environments and human motion from single-camera videos and retargets the motion to humanoid robots for imitation learning. It extracts human poses in world coordinates, maps them to robot configurations, and reconstructs environments as pointclouds later converted to meshes.

## VideoMimic Simulation

Provides sim training pipeline. See [readme](simulation/README.md) for details. It proceeds in 4 stages including motion capture pretraining, scene-conditioned tracking, distillation, and RL finetuning.

## VideoMimic Sim-to-Real

Provides real world deployment pipeline. See [readme](sim2real/README.md) for details. We provide a C++ file which you can compile to a binary to run on your real robot using torchscript-exported checkpoints.

## Video Dataset

 Uploaded [here](https://drive.google.com/file/d/1lQWmxebQX9Yu_KBX_tpyg3CuP61hZrvP/view?usp=sharing). Note that individual videos are provided as sequences of jpegs rather than encoded mp4s.


**[BibTex]**
```
@inproceedings{allshire2025visual,
  title={Visual Imitation Enables Contextual Humanoid Control},
  author={Allshire, Arthur and Choi, Hongsuk and Zhang, Junyi and McAllister, David and Zhang, Anthony and Kim, Chung Min and Darrell, Trevor and Abbeel, Pieter and Malik, Jitendra and Kanazawa, Angjoo},
  booktitle={Proceedings of The Conference on Robot Learning},
  series={Proceedings of Machine Learning Research},
  year={2025}
}
```
