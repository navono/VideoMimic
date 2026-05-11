from setuptools import find_packages, setup

setup(name='videomimic_gym',
      version='0.0.1',
      author='Arthur Allshire',
      license="MIT",
      packages=find_packages(),
      author_email='arthur@allshire.org',
      description='VideoMimic Gym',
      python_requires='>=3.10',
      install_requires=['rsl-rl', 'matplotlib', 'tensorboard', 'mujoco>=3.2.3', 'pyyaml', 'plotly', 'wandb', 'trimesh', 'numpy<2', 'hydra-core', 'warp-lang', 'tqdm', 'yourdfpy', 'viser', 'pyliblzfse', 'robot_descriptions', 'h5py', 'dm_control', 'rtree'],
      extras_require={
          'isaacgym': ['isaacgym'],
      })
