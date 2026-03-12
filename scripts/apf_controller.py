#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment  
import os
import csv
import time
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class APFSwarmController():
    def __init__(self, p_cohesion=1.0, p_seperation=1.0, p_alignment=1.0, max_vel=0.5, min_dist=0.3) -> None:
        self.swarm = None
        self.goals = None
        
        self.min_dist = min_dist
        self.max_vel = max_vel
        
        self.velocities = None
        self.p_separation = p_seperation
        self.p_cohesion = p_cohesion

        # =================================================================
        # 🧠 [ATO 模块引入]: 全局开关
        # 对比 baseline: 原版只有贪心算法，这里引入了基于 LLM 拓扑优化的状态标志。
        # =================================================================
        self.enable_ato = False  
        
        self.log_dir = ""            
        self.current_log_name = ""   
        self.csv_initialized = False 
        self.start_time = 0.0        
        self.last_csv_path = ""      

        # =================================================================
        # 🛡️ [SRM 模块引入]: 安全返航状态机
        # 对比 baseline: 彻底重构了系统的生命周期，增加了统一的返航时间、初始/目标点快照。
        # =================================================================
        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0
        self.return_duration = 5.0
        
        # =================================================================
        # 🚀 [FMS 模块引入]: 动态活跃数量追踪
        # 对比 baseline: 避免了全局遍历，使得天地飞机的状态得以解耦分离。
        # =================================================================
        self.current_shape_num = 0
        self.current_active_num = 0
        self.moving_mask = None 

    # =================================================================
    # 🛡️ [SRM 模块引入]: 核心返航初始化函数
    # =================================================================
    def initiate_safe_return(self, start_poses, home_poses):
        self.is_returning = True
        n = min(len(start_poses), len(home_poses))
        self.return_start_poses = start_poses[:n].copy()
        self.return_home_poses = home_poses[:n].copy()
        
        self.moving_mask = np.zeros(n, dtype=bool)
        m = min(self.current_active_num, n)
        
        if m > 0:
            self.moving_mask[:m] = True
            
            dist_matrix = cdist(self.return_start_poses[:m], self.return_home_poses[:m])
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched_homes = np.zeros_like(self.return_home_poses[:m])
            for r, c in zip(row_ind, col_ind):
                matched_homes[r] = self.return_home_poses[:m][c].copy()
            self.return_home_poses[:m] = matched_homes
            
            max_dist = np.max(np.linalg.norm(self.return_home_poses[:m] - self.return_start_poses[:m], axis=1))
        else:
            max_dist = 0
            
        self.return_duration = max(max_dist / (self.max_vel * 0.45), 6.0) 
        
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy() 
        
        print(f"\n[SRM] Safe Return Activated. {m} active drones returning. Est. time: {self.return_duration:.1f}s")

    # =================================================================
    # 🧠 [ATO + FMS 混合模块]: 重构的目标分配器
    # =================================================================
    def distribute_goals(self, start, goals, shape_num=None, active_num=None):
        if shape_num is None: shape_num = len(goals)
        if active_num is None: active_num = len(goals)
        
        self.current_shape_num = shape_num
        self.current_active_num = active_num

        out_goals = np.copy(goals)

        if active_num > 0:
            active_start = start[:active_num]
            active_goals = goals[:active_num]
            shape_goals = active_goals[:shape_num]
            rtb_goals = active_goals[shape_num:]

            if self.enable_ato and shape_num > 1:
                shape_goals = shape_goals + np.random.normal(0, 1e-3, shape_goals.shape) 
                
                dist_matrix_llm = cdist(shape_goals, shape_goals)
                np.fill_diagonal(dist_matrix_llm, np.inf) 
                
                min_dists = np.min(dist_matrix_llm, axis=1)
                effective_min = np.median(min_dists)
                effective_min = max(effective_min, 0.02) 
                
                # =================================================================
                # 终极炼丹参数 1: 钟摆下压深度 (0.985)
                # 即 0.2955m。目标点被放置在 0.3m 安全线的下方。
                # 强有力的引力会迫使机群突破 0.3m，提供“向下震荡”的物理牵引力。
                # =================================================================
                target_spacing = self.min_dist * 0.985 
                
                # =================================================================
                # 终极炼丹参数 2: 动态体积界限 [0.45, 1.8]
                # =================================================================
                scale = np.clip(target_spacing / effective_min, 0.45, 1.8) 
                
                centroid = np.mean(shape_goals, axis=0)
                scaled_shape_goals = centroid + (shape_goals - centroid) * scale

                for _ in range(60):
                    dists = cdist(scaled_shape_goals, scaled_shape_goals)
                    np.fill_diagonal(dists, np.inf)
                    min_dists = np.min(dists, axis=1)
                    if np.min(min_dists) >= target_spacing * 0.99: break
                        
                    displacement = np.zeros_like(scaled_shape_goals)
                    for i in range(len(scaled_shape_goals)):
                        mask = dists[i] < target_spacing
                        if np.any(mask):
                            vecs = scaled_shape_goals[i] - scaled_shape_goals[mask]
                            ds = dists[i][mask].reshape(-1, 1)
                            ds_safe = np.maximum(ds, 1e-4) 
                            
                            pushes = (vecs / ds_safe) * (target_spacing - ds) * 0.25
                            displacement[i] = np.sum(pushes, axis=0)
                    
                    displacement = np.clip(displacement, -0.08, 0.08)
                    scaled_shape_goals += displacement

                if len(rtb_goals) > 0:
                    scaled_active_goals = np.vstack((scaled_shape_goals, rtb_goals))
                else:
                    scaled_active_goals = scaled_shape_goals

                cost_matrix = cdist(active_start, scaled_active_goals) ** 2.0 
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    out_goals[r] = scaled_active_goals[c]
                print(f"\n[ATO] Adaptive Topology (Scale: {scale:.2f}). Shape drones: {shape_num}")
                
            else:
                dist_matrix = cdist(active_start, active_goals)
                for i in range(active_num):
                    ind = np.argmin(dist_matrix[i])
                    out_goals[i] = active_goals[ind]
                    dist_matrix[i, :] = np.inf
                    dist_matrix[:, ind] = np.inf
                print(f"\n[Baseline] Greedy topology. Shape drones: {shape_num}")

        self.goals = out_goals

    def get_control(self, poses) -> None:
        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]
        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
            
        if self.is_returning:
            elapsed = time.time() - self.return_start_time
            progress = min(elapsed / self.return_duration, 1.0)
            smooth_p = progress * progress * (3 - 2 * progress) 
            bloom_scale = 1.0 + 1.2 * np.sin(np.pi * smooth_p) 
            
            m = min(self.current_active_num, n)
            if m > 0:
                centroid = np.mean(self.return_home_poses[:m], axis=0)
                bloomed_home = centroid + (self.return_home_poses[:m] - centroid) * bloom_scale
                self.goals[:m] = self.return_start_poses[:m] + (bloomed_home - self.return_start_poses[:m]) * smooth_p

        ball_tree = BallTree(poses[:, :2], metric='euclidean')
        control_vels = np.zeros_like(poses)

        self.goals = np.nan_to_num(self.goals)

        error_vec = self.goals[:n] - poses[:n]
        dist_to_goal = np.linalg.norm(error_vec, axis=1, keepdims=True)
        
        # =================================================================
        # 终极炼丹参数 3: 移除人工死区 (0.02m)
        # 将减速带压缩到 2cm，不干涉物理势场的博弈，保留震荡的高频波折感。
        # =================================================================
        scaling = np.where(dist_to_goal < 0.02, dist_to_goal / 0.02, 1.0) 
        vel_cohesion = self.p_cohesion * error_vec * scaling

        for i, pose in enumerate(poses):
            query_pose = pose[:2]
            v_nom = vel_cohesion[i].copy()
            if np.linalg.norm(v_nom) > self.max_vel:
                v_nom = (v_nom / np.linalg.norm(v_nom)) * self.max_vel

            interaction_radius = self.min_dist * 2.0
            nearest_ind = ball_tree.query_radius(query_pose.reshape(1, -1), interaction_radius)[0][1:]
            
            v_rep = np.zeros(3)
            for ind in nearest_ind:
                p_rel = pose - poses[ind]
                dist = np.linalg.norm(p_rel)
                
                if dist < self.min_dist:
                    safe_dist = max(dist, 0.01)
                    
                    # =================================================================
                    # 终极炼丹参数 4: 浅水蹦床 (0.99触发, 1.4x反弹)
                    # 这是实现“向上震荡”的核心！
                    # 飞机被引力拉到 0.297m 时，触发 1.4倍 斥力。这个力道配合惯性，
                    # 刚好能把飞机甩出 0.30m 的水面（但又不会导致小集群发散）。
                    # =================================================================
                    rep_strength = self.p_separation
                    if dist < self.min_dist * 0.99: 
                        rep_strength *= 1.4 
                        
                    repulsive_mag = rep_strength * (1.0 / safe_dist - 1.0 / self.min_dist) / (safe_dist ** 2 + 0.01)
                    v_rep += repulsive_mag * (p_rel / safe_dist)
            
            v_rep[2] = 0
            control_vels[i] = v_nom + v_rep

        if self.is_returning:
            mask = getattr(self, 'moving_mask', np.ones(n, dtype=bool))
            for i in range(n):
                if not mask[i]:
                    control_vels[i] = 0.0
        else:
            for i in range(n):
                if i >= self.current_active_num:
                    control_vels[i] = 0.0

        # =================================================================
        # 终极炼丹参数 5: 钟摆动量保留 (0.80 / 0.20)
        # 给系统 20% 的惯量。它就像钟摆的配重铅块，让飞机在跨越 0.3m 时
        # 能靠着惯性多滑行一小段距离，从而画出完美的对称波浪。
        # =================================================================
        control_vels = 0.80 * control_vels + 0.20 * self.velocities[:n]
        
        for k in range(len(control_vels)):
            speed = np.linalg.norm(control_vels[k])
            if speed > self.max_vel:
                control_vels[k] = (control_vels[k] / speed) * self.max_vel
        self.velocities[:n] = control_vels.copy()

        if self.log_dir and self.current_log_name and self.current_shape_num > 0:
            full_path = os.path.join(self.log_dir, f"{self.current_log_name}.csv")
            if self.last_csv_path != full_path:
                self.csv_initialized = False
                self.last_csv_path = full_path

            if not self.csv_initialized:
                try:
                    with open(full_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Time(s)", "Min_Distance(m)", "Avg_Velocity(m/s)", "Target_Error(m)"])
                    self.start_time = time.time()
                    self.csv_initialized = True
                except Exception:
                    return control_vels

            curr_t = round(time.time() - self.start_time, 2)
            eval_poses = poses[:self.current_shape_num]
            eval_goals = self.goals[:self.current_shape_num]
            eval_vels = control_vels[:self.current_shape_num]
            
            if self.current_shape_num > 1:
                diffs = eval_poses[:, np.newaxis, :] - eval_poses[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_d = round(np.min(dists), 4)
            else:
                min_d = 0.0
                
            avg_v = round(np.mean(np.linalg.norm(eval_vels, axis=1)), 4)
            err = round(np.mean(np.linalg.norm(eval_goals - eval_poses, axis=1)), 4)

            try:
                with open(full_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([curr_t, min_d, avg_v, err])
            except:
                pass 
        return control_vels

    def generate_plots(self):
        if not self.last_csv_path or not os.path.exists(self.last_csv_path): return
        mode_prefix = "ATO" if self.enable_ato else "Base"
        algo_label = "ATO (Ours)" if self.enable_ato else "Baseline"

        print(f"\n[*] Generating plots for [{algo_label}] mode...")
        try:
            df = pd.read_csv(self.last_csv_path)
            metrics = {
                'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)', '#2ECC71' if self.enable_ato else '#E74C3C'),
                'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)', '#2ECC71' if self.enable_ato else '#E74C3C'),
                'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)', '#2ECC71' if self.enable_ato else '#E74C3C')
            }
            for col, (title, ylabel, color) in metrics.items():
                if col in df.columns:
                    plt.figure(figsize=(9, 5.5))
                    plt.plot(df['Time(s)'], df[col], linewidth=2.5 if self.enable_ato else 1.5, 
                             color=color, linestyle='-' if self.enable_ato else '--', 
                             label=algo_label, alpha=0.9)
                    
                    if col == 'Target_Error(m)':
                        plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal')
                    elif col == 'Min_Distance(m)':
                        # 这里动态使用了 self.min_dist，实现了图表随用户输入变化
                        plt.axhline(y=self.min_dist, color='black', linestyle='-.', label=f'Safety Limit ({self.min_dist}m)')
                        plt.axhspan(0, self.min_dist, color='gray', alpha=0.15)
                        plt.ylim(bottom=max(0, self.min_dist - 0.05), top=df[col].max() * 1.05)
                    elif col == 'Avg_Velocity(m/s)':
                        plt.axhline(y=self.max_vel, color='blue', linestyle=':', alpha=0.5, label='Max Velocity')
                        plt.ylim(bottom=-0.05, top=self.max_vel + 0.1)

                    plt.title(title, fontweight='bold', fontsize=14)
                    plt.xlabel('Time $t$ (s)', fontsize=12)
                    plt.ylabel(ylabel, fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
                    
                    img_name = f"{mode_prefix}_{self.current_log_name}_{col.split('(')[0]}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.log_dir, img_name), dpi=300)
                    plt.close()
            print(f"[*] Plots saved: {self.log_dir}")
        except Exception as e:
            print(f"⚠️ Plotting Error: {e}")