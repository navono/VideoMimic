# 环境安装

建议使用 `Claude Code`

# 视频输入
## 拍摄要求
  - 相机要移动（有视差），静态相机 3D 重建效果差
  - 人保持在画面中心，避免遮挡
  - 场景要有纹理（避免白墙、纯色平面）
  - 尽量拍全场景表面，mesh 更完整

## 视频规格
  - 格式：MP4/MOV 均可，高码率优先
  - 帧数：50-200 帧最佳，不要超过 300 帧（GPU 显存限制）
  - FPS：30fps，配合 STRIDE=2 降采样
  - 分辨率：无硬性要求，内部会 resize 到 512 处理

## 硬件
  - MegaSAM 重建 ~300 帧需 24GB+ 显存
  - 4070 Ti Super (16GB) 建议 100-150 帧 + STRIDE=2

```sh
  # 提取前 150 帧
  make extract-frames VIDEO_NAME=path/to/video.mp4 VIDEO_PATH=path/to/video.mp4 EXTRACT_END=150

  # 跑 pipeline
  make pipeline VIDEO_NAME=path/to/video.mp4 STRIDE=2 HEIGHT=1.7
```

- STRIDE=2 — 帧采样间隔。从视频中每隔 2 帧取 1 帧（取第 1、3、5、7...帧）。30fps 视频用 STRIDE=2 等效于
   15fps，计算量减半。

- HEIGHT=1.7 — 视频中人物的身高（1.7 米）。用于 SMPL
  体型优化，让重建出的人体模型和场景的比例更准确。可选值：
  - -1（默认）：从视频自动检测人体身高，按检测到的身高优化 SMPL 体型
  - 0：跳过人体身高拟合，直接用 G1 机器人的体型参数（--use-g1-shape）来优化 SMPL。相当于把人"缩"成机器人的体型比例再 retarget，误差更小
  - 1.7 等具体数值：手动指定视频中人物的身高（米），用于 SMPL 体型缩放

# Real2Sim

## 单次运行

运行
```sh
cd real2sim

export HF_TOKEN=hf
make extract-frames VIDEO_NAME=assets/sitting_standing.mp4
```

验证：
```sh
make visualize VIDEO_NAME=assets/sitting_standing.mp4
```

## HTTP 服务

配合 `benchverse` 使用

```sh
cd real2sim

make serve-install
make serve
```


# Simulation

