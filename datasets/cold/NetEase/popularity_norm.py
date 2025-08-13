import numpy as np
from collections import defaultdict

def merge_and_scale(file1, file2, output_file="scaled_data.txt"):
    """精确5段20%比例放缩（含80-100%→9-10）"""
    # 1. 合并数据
    merged = defaultdict(float)
    for file in [file1, file2]:
        with open(file, 'r') as f:
            for line in f:
                try:
                    key, val = line.strip().split()
                    merged[int(key)] += float(val)
                except:
                    continue

    if not merged:
        print("错误：无有效数据")
        return

    lesseq5 = {k: v for k, v in merged.items() if v <= 5}
    greater5 = np.array([v for v in merged.values() if v > 5])
    
    if len(greater5) == 0:
        processed = lesseq5
    else:
        # 2. 计算4个20%分位点（严格5段划分）
        quantiles = np.quantile(greater5, [0.2, 0.4, 0.6, 0.8], method='linear')
        max_val = greater5.max()
        zones = [5] + list(quantiles) + [max_val]  # 生成5个区间边界
        scale_ranges = [(5,6), (6,7), (7,8), (8,9), (9,10)]  # 目标范围

        processed = {}
        for k, v in merged.items():
            if v <= 5:
                processed[k] = v
                continue
            
            # 3. 找到所属区间并线性放缩
            for i in range(5):
                if zones[i] < v <= zones[i+1]:
                    # 区间内比例计算
                    zone_start, zone_end = zones[i], zones[i+1]
                    target_start, target_end = scale_ranges[i]
                    proportion = (v - zone_start) / (zone_end - zone_start)
                    scaled = target_start + proportion * (target_end - target_start)
                    processed[k] = round(scaled, 3)  # 保留3位小数
                    break
            else:
                # 理论上不会执行（zones包含max_val）
                processed[k] = 10.0

    # 4. 保存结果
    with open(output_file, 'w') as f:
        for key in sorted(processed.keys()):
            f.write(f"{key} {processed[key]:.3f}\n")
    
    print(f"\n精确5段放缩完成，结果保存到：{output_file}")
    print(f"• 总数据: {len(merged)} → 分段后: {len(processed)}")
    print(f"• 0-5段: {len(lesseq5)} | 5+段: {len(greater5)}")
    print(f"• 分位点: {np.round(quantiles, 2)}（若有>5数据）")

if __name__ == "__main__":
    merge_and_scale(
        "ui_item_counts.txt",
        "weighted_bi_item_counts.txt",
        "scaled_merged.txt"
    )
