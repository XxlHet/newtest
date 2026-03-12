#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime
import numpy as np

def get_all_csv_files(limit=30):
    """全局搜索 result 文件夹，按生成时间倒序列出最近的 CSV 文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_path = os.path.join(base_dir, 'result', '**', '*.csv')
    
    files = glob.glob(search_path, recursive=True)
    if not files:
        return []
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit]

def generate_multi_comparison_plots():
    print("\n" + "="*60)
    print("   📊 Multi-Agent Swarm Data Comparison Tool")
    print("="*60)

    csv_files = get_all_csv_files()
    if not csv_files:
        print("❌ 未在 result 目录中找到任何 CSV 数据文件！")
        return

    print("\n[*] 发现以下最近的实验数据：")
    for i, f in enumerate(csv_files):
        rel_path = os.path.relpath(f, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"  [{i}] {rel_path}")

    print("\n" + "-"*60)
    sel_input = input(">>> 请输入要对比的文件序号 (用逗号分隔，如 '0, 1, 3'): ")
    
    try:
        indices = [int(x.strip()) for x in sel_input.split(',')]
        selected_files = [csv_files[i] for i in indices]
    except Exception as e:
        print("❌ 输入格式错误或序号越界，程序退出。")
        return

    print("\n" + "-"*60)
    labels = []
    for f in selected_files:
        default_label = os.path.basename(f).replace('.csv', '')
        lbl = input(f">>> 为 '{default_label}' 输入图例名称 (直接回车则使用文件名): ")
        labels.append(lbl if lbl.strip() else default_label)

    print("\n[*] 正在读取数据并生成对比图表...")
    dfs = []
    for f in selected_files:
        dfs.append(pd.read_csv(f))

    # 🎯 核心逻辑：从文件名中侦测 ATO 的专属安全基线
    target_baseline = 0.3 # 默认 baseline
    for f in selected_files:
        fname = os.path.basename(f)
        if "ATO" in fname and "_SD" in fname:
            try:
                # 提取形如 _SD0.40.csv 中的 0.40
                sd_str = fname.split("_SD")[1].replace(".csv", "")
                target_baseline = float(sd_str)
                print(f"[*] 🔍 侦测到 ATO 数据，已强制同步图表安全基线为: {target_baseline}m")
                break # 采纳查找到的第一个 ATO 基线
            except Exception:
                pass

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_target_dir = os.path.join(base_dir, 'result', f"{timestamp}_MultiComparison")
    os.makedirs(new_target_dir, exist_ok=True)
    print(f"[*] 📁 创建图表输出目录: {new_target_dir}")

    metrics = {
        'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)'),
        'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)'),
        'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)')
    }

    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    for col, (title, ylabel) in metrics.items():
        if all(col in df.columns for df in dfs):
            plt.figure(figsize=(10, 6))
            
            global_max = 0.0
            
            for idx, df in enumerate(dfs):
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                linewidth = 2.5 if idx == 0 else 1.8 
                
                plt.plot(df['Time(s)'], df[col], linewidth=linewidth, color=color, 
                         linestyle=linestyle, label=labels[idx], alpha=0.9)
                
                if not df[col].empty:
                    global_max = max(global_max, df[col].max())
            
            if col == 'Target_Error(m)':
                plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal (0.0m)')
            elif col == 'Min_Distance(m)':
                plt.axhline(y=target_baseline, color='black', linestyle='-.', linewidth=1.5, label=f'Safety Limit ({target_baseline}m)')
                plt.axhspan(0, target_baseline, color='gray', alpha=0.15)
                plt.ylim(bottom=max(0.15, target_baseline - 0.05), top=global_max * 1.05)
                
                # ==========================================
                # 🌟 坐标轴刻度劫持：保留原有刻度，追加并加粗放大基线数值（不改变颜色）
                # ==========================================
                ax = plt.gca()
                current_ticks = list(ax.get_yticks())
                
                # 将目标基线强行加入刻度列表，并去重排序
                current_ticks.append(target_baseline)
                new_ticks = sorted(list(set(current_ticks))) 
                ax.set_yticks(new_ticks)
                
                # 遍历所有刻度，只把我们设定的基准线那个刻度加粗并适当放大，不改色
                for label in ax.get_yticklabels():
                    if abs(label.get_position()[1] - target_baseline) < 1e-5:
                        label.set_fontsize(14)
                        label.set_fontweight('bold')
                # ==========================================

            elif col == 'Avg_Velocity(m/s)':
                plt.axhline(y=1.0, color='blue', linestyle=':', alpha=0.5, label='Max Velocity Limit')
                plt.ylim(bottom=-0.05, top=global_max * 1.15 if global_max > 1.0 else 1.1)

            plt.title(title, fontweight='bold', fontsize=15)
            plt.xlabel('Time $t$ (s)', fontsize=13)
            plt.ylabel(ylabel, fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
            
            save_name = f"MultiCompare_{col.split('(')[0]}.png"
            plt.tight_layout()
            
            save_path = os.path.join(new_target_dir, save_name)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"✅ 成功生成图表: {save_name}")
        else:
            print(f"⚠️ 警告: 数据列 '{col}' 在某些选定的 CSV 中缺失，已跳过。")

    print("\n[*] 🎉 所有对比图表均已生成完毕！")
    print(f"[*] 请前往 {new_target_dir} 查看。")

if __name__ == "__main__":
    generate_multi_comparison_plots()