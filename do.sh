#!/bin/sh

# 配置文件路径
config_file="config.yaml"

# 定义测试的 k 值
k_values="10 100 1000 10000"

# 循环遍历 k 值
for k in $k_values
do
    # 使用 sed 命令修改配置文件中的 k 值
    sed -i "/iFashion:/,/^[^ ]/ s/^\(  k: \)[0-9.]*$/\1$k/" $config_file

    # 输出当前使用的 k 值
    echo "Running with k = $k"

    # 执行 Python 命令
    python train.py -g 0 -m MultiCBR -d iFashion
done