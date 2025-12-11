#!/bin/bash
MOUNT_POINT="/mnt/data/xujiahao/beaver/moudels/register_pm/mnt"
NAMESPACE="namespace0.0"
REGION="region0"


CURRENT_MODE=$(ndctl list -N -r "$REGION" | jq -r ".[0].mode")

if [ "$CURRENT_MODE" == "devdax" ]; then
    echo "Namespace is already in devdax mode, skipping unmount/destroy/create."
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
echo "Creating dax namespace ..."
sudo ndctl create-namespace --region="$REGION" --mode=devdax
if [ $? -ne 0 ]; then
    echo "Failed to create dax namespace"
    exit 1
fi

echo "Done!"