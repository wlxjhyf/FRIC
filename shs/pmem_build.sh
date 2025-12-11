#!/bin/bash

MOUNT_POINT="/mnt/data/xujiahao/beaver/moudels/register_pm/mnt"
NAMESPACE="namespace0.0"
REGION="region0"
PMEM_DEV="/dev/pmem0"

CURRENT_MODE=$(ndctl list -N -r "$REGION" | jq -r ".[0].mode")

if [ "$CURRENT_MODE" == "fsdax" ]; then
    echo "Namespace is already in fsdax mode, skipping unmount/destroy/create."
    exit 0
fi

# 1. 检查是否已挂载
if mountpoint -q "$MOUNT_POINT"; then
    echo "Unmounting $MOUNT_POINT ..."
    sudo umount "$MOUNT_POINT"
    if [ $? -ne 0 ]; then
        echo "Failed to unmount $MOUNT_POINT"
        exit 1
    fi
else
    echo "$MOUNT_POINT is not mounted, skipping umount"
fi

# 2. 销毁 namespace
echo "Destroying namespace $NAMESPACE ..."
sudo ndctl destroy-namespace "$NAMESPACE" -f
if [ $? -ne 0 ]; then
    echo "Failed to destroy namespace $NAMESPACE"
    exit 1
fi

# 3. 创建 fsdax namespace
echo "Creating fsdax namespace ..."
sudo ndctl create-namespace --region="$REGION" --mode=fsdax
if [ $? -ne 0 ]; then
    echo "Failed to create fsdax namespace"
    exit 1
fi

# 4. 挂载 pmem0
echo "Mounting $PMEM_DEV to $MOUNT_POINT with dax ..."
sudo mount -o dax -t ext4 "$PMEM_DEV" "$MOUNT_POINT"
if [ $? -ne 0 ]; then
    echo "Failed to mount $PMEM_DEV to $MOUNT_POINT"
    exit 1
fi

echo "Done!"