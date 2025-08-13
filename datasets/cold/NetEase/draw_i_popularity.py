import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# 全局配置（无多余注释）
plt.rcParams.update({
    'font.family': 'SimSun',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'grid.color': '#e0e0e0'
})

def merge_files(file1, file2):
    """合并两个文件，返回原始数据和合并数据"""
    raw1 = {}
    raw2 = {}
    merged = defaultdict(float)
    
    with open(file1, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                key, val = map(float, line.strip().split())
                raw1[int(key)] = val
                merged[int(key)] += val
            except Exception as e:
                print(f"警告：文件1第{line_num}行解析失败 - {str(e)}")
    
    with open(file2, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                key, val = map(float, line.strip().split())
                raw2[int(key)] = val
                merged[int(key)] += val
            except Exception as e:
                print(f"警告：文件2第{line_num}行解析失败 - {str(e)}")
    
    return raw1, raw2, merged

def plot_and_save(raw1, raw2, merged, save_name="merged_curve.png"):
    """绘制聚焦前90%数据的平滑曲线（无任何图标）"""
    if not merged:
        print("错误：合并后数据为空，无法绘图")
        return
    
    rounded_values = np.array([round(v) for v in merged.values()])
    if len(rounded_values) < 2:
        print("警告：数据点少于2个，无法生成有效图表")
        return
    
    # 计算90%分位数并过滤数据
    cutoff = np.quantile(rounded_values, 0.9)
    filtered_values = [v for v in rounded_values if v <= cutoff]
    
    # 频率统计（仅保留有效数据）
    frequency = defaultdict(int)
    for v in filtered_values:
        frequency[v] += 1
    if not frequency:
        print("警告：过滤后无有效数据，跳过绘图")
        return
    
    sorted_ints = sorted(frequency.keys())
    x = sorted_ints
    y = [frequency[v] for v in sorted_ints]

    # 绘制曲线
    plt.figure(figsize=(10, 5.5))
    plt.plot(x, y, color='#2B6CB0', linewidth=2.2, alpha=0.9)

    # 智能刻度（最多8个整数标签）
    plt.gca().xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, prune='both', nbins=8)
    )
    
    # 聚焦显示区域
    plt.xlim(left=min(sorted_ints)-1, right=cutoff + 2)
    plt.ylim(bottom=0)

    # 图表标签（无多余装饰）
    plt.title('聚焦靠近0的90%数据分布（整数四舍五入）', fontsize=14, pad=12)
    plt.xlabel('四舍五入后的值（显示≤{:.0f}）'.format(cutoff), fontsize=11, labelpad=8)
    plt.ylabel('出现次数', fontsize=11, labelpad=8)

    # 网格和边框
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(pad=3.5)

    # 保存图表（无成功提示图标）
    try:
        plt.savefig(save_name, dpi=300, facecolor='white')
        print(f"图表已保存：{save_name}（尺寸10x5.5英寸，300dpi）")
    except Exception as e:
        print(f"保存失败：{str(e)}")
    finally:
        plt.close()

if __name__ == "__main__":
    FILE1 = "ui_item_counts.txt"
    FILE2 = "weighted_bi_item_counts.txt"
    
    try:
        raw1, raw2, merged = merge_files(FILE1, FILE2)
        print(f"\n合并完成：{len(merged)}个键（文件1：{len(raw1)}，文件2：{len(raw2)}）")
        plot_and_save(raw1, raw2, merged)
    except FileNotFoundError:
        print("错误：请检查文件路径是否正确")
    except Exception as e:
        print(f"发生意外错误：{str(e)}")
