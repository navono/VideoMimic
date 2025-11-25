#! /bin/bash

# if downloading on mac
if [ -d "__MACOSX" ]; then
    rm -rf __MACOSX
fi

bash download_gdrive_data.sh 11zjLwc-gV_xsNe2ON-XSvj4L1qk78tNO videomimic_captures.zip

bash download_gdrive_data.sh 1_9UMOzfAJi6HMcd_hLOIsuFhrrnnKMbZ unitree_lafan.zip

bash download_gdrive_data.sh 1rzwgQ5cf0amGM7e8qXMEgiOIvgowSAmI checkpoints.zip

mkdir -p ../videomimic_gym/logs/g1_deepmimic
cp -r checkpoints/* ../videomimic_gym/logs/g1_deepmimic/

if [ -d "__MACOSX" ]; then
    rm -rf __MACOSX
fi