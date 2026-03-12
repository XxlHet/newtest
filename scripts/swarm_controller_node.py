#!/usr/bin/env python3

from flock_gpt.msg import Vector3StampedArray
from geometry_msgs.msg import Vector3
import numpy as np
import rospy
from sdf import box, sphere, write_binary_stl, rounded_box, capsule
from sdf import generate
from apf_controller import APFSwarmController
from gpt_sdf import SDFDialog, SDFModel, CURRENT_MODEL
import threading
import os
import time
import sys  # 🌟 新增：用于刷新缓冲区和检测终端状态
from point_distributor import PointDistributer

class SwarmControllerNode():
    def get_input(self, prompt, default_val):
        """🛡️ 工业级输入捕获：防止 roslaunch 后台模式导致的 EOFError 崩溃"""
        sys.stdout.flush()  # 强制刷新缓冲区，确保打印能立刻显示
        try:
            # 检测是否处于无交互终端环境 (例如标准的 roslaunch)
            if not sys.stdin.isatty():
                print(f"{prompt}\n[!] ⚠️ 非交互模式 (roslaunch) 运行中，自动套用默认值: {default_val}")
                sys.stdout.flush()
                return str(default_val)
            
            val = input(prompt)
            return val if val.strip() != "" else str(default_val)
        except (EOFError, KeyboardInterrupt):
            print(f"\n[!] ⚠️ 检测到输入流异常 (可能是 roslaunch 导致)，自动套用默认值: {default_val}")
            sys.stdout.flush()
            return str(default_val)

    def __init__(self, goals=[]) -> None:
        # ⚛️ 替换为 ATOM-UAV 的专属启动界面
        print("\n" + "═"*65)
        print("   ⚛️  ATOM-UAV Swarm Control Framework Initializing...")
        print("   (Adaptive Topology & Operations Management)")
        print("   -------------------------------------------------")
        # 👇 新增下面这一行，动态显示大模型名称
        print(f"   [Loaded] LLM : {CURRENT_MODEL} (Shape Generation Engine)") 
        print("   [Loaded] FMS : Dynamic Fleet Management System")
        print("   [Loaded] SRM : Safe Return & Ground Lock Module")
        print("   [Standby] ATO: Adaptive Trajectory Optimization")
        print("═"*65)
        sys.stdout.flush()
        
        # 🌟 核心理念：设定最大机队容量（无人机池）
        val = self.get_input(">>> [FMS] Enter Maximum Fleet Capacity (e.g. 200, must be >= max shape size): ", "200")
        try:
            self.fleet_capacity = int(val)
        except:
            self.fleet_capacity = 200
            
        # 设置模拟器一次性生成，后续绝不修改此参数，彻底杜绝重启
        rospy.set_param('/swarm_num_drones', self.fleet_capacity)
        
        self.shape_drones = 0       # 当前形状所需无人机数
        self.prev_active_drones = 0 # 记录上一轮飞在空中的无人机数

        print("\n" + "━"*60)
        print("--- ⚙️ Algorithm Module Configuration ---")
        module_input = self.get_input(">>> [ATO] Enable Adaptive Trajectory Optimization? [y/N]: ", "n")
        self.enable_ato = True if module_input.lower() == 'y' else False
        
        # 🎯 动态安全基线设定
        self.safety_baseline = 0.3
        if self.enable_ato:
            sb_input = self.get_input(">>> [ATO] Set Safety Baseline Distance (0.2 - 0.5m, Default 0.3m): ", "0.3")
            try:
                val = float(sb_input)
                if 0.2 <= val <= 0.5:
                    self.safety_baseline = val
                else:
                    print(f"[!] [ATO] Out of bounds ({val}). Defaulting to 0.3m.")
            except ValueError:
                print("[!] [ATO] Invalid input. Defaulting to 0.3m.")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(base_dir, 'result', timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"[*] [ATOM-UAV] Mode: ATO Enabled = {self.enable_ato}")
        print(f"[*] [ATOM-UAV] Safety Baseline : {self.safety_baseline}m")
        print(f"[*] [FMS] Fleet Capacity: {self.fleet_capacity} drones (Simulating...)")
        print("="*60 + "\n")
        sys.stdout.flush()

        self.is_running = False 
        self.goals = []
        self.start_poses = None
        self.home_poses = None  
        self.trigger_return = False 
        
        self.controller = APFSwarmController(max_vel=1, min_dist=self.safety_baseline)
        self.controller.log_dir = self.save_dir 
        self.controller.enable_ato = self.enable_ato
        
        self.model = SDFModel()
        self.dialog = SDFDialog()

        rospy.Subscriber("/swarm/poses", Vector3StampedArray, self.callback_state, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('/swarm/cmd_vel', Vector3StampedArray, queue_size=1)
        
        threading.Thread(target=self.continuous_input_prompt, daemon=True).start()

    def callback_state(self, msg:Vector3StampedArray):
        poses = np.array([[p.x, p.y, p.z] for p in msg.vector])
        
        # 🌟 获取模拟器中全部的机队网格坐标，作为永久的大本营
        if self.home_poses is None and len(poses) >= self.fleet_capacity:
            self.home_poses = poses[:self.fleet_capacity].copy()
            print(f"[*] [FMS] Captured home grid for {self.fleet_capacity} drones. Fleet ready for dispatch.")
            sys.stdout.flush()
            
        if getattr(self, 'trigger_return', False):
            self.controller.initiate_safe_return(poses, self.home_poses)
            self.trigger_return = False
            self.start_poses = poses 
        
        if not self.is_running or self.goals is None or np.array(self.goals).size == 0:
            return
            
        if self.start_poses is None:
            if self.home_poses is None: return # 等待环境加载
            
            # 计算本次调度牵涉到的无人机总数 (当前需要的 vs 原本在天上需要返航的)
            active_drones = max(self.prev_active_drones, self.shape_drones)
            
            self.controller.distribute_goals(
                poses, 
                self.goals, 
                shape_num=self.shape_drones, 
                active_num=active_drones
            ) 
            self.start_poses = poses
            # 记录本次飞行的活跃状态，用于下一轮判断
            self.prev_active_drones = self.shape_drones

        vels = self.controller.get_control(poses)
        cmd_vel = Vector3StampedArray()
        
        for vel in vels:
            vect = Vector3()
            vect.x, vect.y, vect.z = vel[0], vel[1], vel[2]
            cmd_vel.vector.append(vect)
        
        if not rospy.is_shutdown():
            self.cmd_vel_publisher.publish(cmd_vel)

    def process_user_input(self, user_input):
        sdf_code = self.dialog.get_next_sdf_code(user_input)
        if sdf_code:
            local_vars = {"f": None}
            try: 
                f=10
                exec(sdf_code, globals(), local_vars)  
                f = local_vars.get("f")
                if f is not None:
                    pd_dist = PointDistributer(f)
                    points = pd_dist.generate_points(self.shape_drones)
                    
                    # 🌟 池化路由分配：构造全局目标阵列
                    full_goals = np.zeros((self.fleet_capacity, 3))
                    
                    # 1. 构图部队
                    if self.shape_drones > 0:
                        full_goals[:self.shape_drones] = points[:self.shape_drones]
                        
                    # 2. 待命/返航部队
                    if self.fleet_capacity > self.shape_drones:
                        full_goals[self.shape_drones:] = self.home_poses[self.shape_drones:]
                        
                    self.goals = full_goals
            except Exception as e:
                print("Error: ", e)

    def execute_return_sequence(self):
        if self.home_poses is None:
            print("🛑 [ATOM-UAV] Shutting down system...")
            sys.stdout.flush()
            rospy.signal_shutdown("User exit")
            return
            
        print("\n" + "="*60)
        print("[*] 🛬 [SRM] Activating SRM (Safe Return Module)...")
        print("[*] [SRM] Executing global return for all airborne drones.")
        print("="*60)
        sys.stdout.flush()
        
        self.trigger_return = True
        self.controller.current_log_name = "" 
        self.is_running = True
        
        self.get_input("\n>>> [SRM] Press 'Enter' when all drones have landed safely to power off...", "")
        self.is_running = False
        print("🛑 [ATOM-UAV] System powered off successfully.")
        sys.stdout.flush()
        rospy.signal_shutdown("Experiment finished")

    def continuous_input_prompt(self):
        rospy.sleep(1.0)
        
        while not rospy.is_shutdown():
            # 动态请求无人机数量
            num_input = self.get_input(f"\n>>> [FMS] Enter target shape size (Max {self.fleet_capacity}): ", "10")
            try:
                self.shape_drones = min(int(num_input), self.fleet_capacity)
            except:
                print("[!] [FMS] Invalid number. Setting to 10.")
                self.shape_drones = 10
                
            user_input = self.get_input(f">>> [ATOM-UAV] What to build with {self.shape_drones} drones? (e.g., sphere) [type 'exit' to quit]: \n", "exit")
            if user_input.lower() in ['exit', 'quit']:
                self.execute_return_sequence() 
                break
            
            shape_name = user_input.replace(" ", "_")
            mode_str = "ATO" if self.enable_ato else "Base"
            
            # 💡 将安全基线嵌入文件名，供比对脚本读取
            sd_str = f"_SD{self.controller.min_dist:.2f}"
            self.controller.current_log_name = f"data_{shape_name}_{self.shape_drones}drones_{mode_str}{sd_str}"
            
            self.controller.is_returning = False
            self.process_user_input(user_input)
            self.start_poses = None
            self.is_running = True 
            
            # 极其优雅的调度提示
            if self.shape_drones > self.prev_active_drones:
                print(f"[*] [FMS] Scaling UP: {self.prev_active_drones} airborne + {self.shape_drones - self.prev_active_drones} launching from ground.")
            elif self.shape_drones < self.prev_active_drones:
                print(f"[*] [FMS] Scaling DOWN: {self.shape_drones} morphing shape, {self.prev_active_drones - self.shape_drones} automatically returning to base.")
            else:
                print(f"[*] [FMS] Seamless Morphing: All {self.shape_drones} airborne drones transitioning to new shape.")
                
            print(f"[*] [ATOM-UAV] Deploying '{shape_name}' (Log: {self.controller.current_log_name}.csv)...")
            sys.stdout.flush()

            self.get_input("\n>>> [ATOM-UAV] Press 'Enter' when formation is complete to generate individual plots...", "")
            self.is_running = False 
            
            self.controller.generate_plots()
            
            cont = self.get_input("\n>>> [ATOM-UAV] Do you want to try another shape? (y/n): ", "n")
            if cont.lower() != 'y':
                self.execute_return_sequence() 
                break

            print("\n" + "━"*60)
            print("--- ⚙️ Algorithm Module Configuration (New Round) ---")
            module_input = self.get_input(f">>> [ATO] Enable Adaptive Trajectory Optimization? (Current: {self.enable_ato}) [y/N]: ", "n")
            self.enable_ato = True if module_input.lower() == 'y' else False
            self.controller.enable_ato = self.enable_ato

            # 🎯 循环中的动态安全基线更新
            if self.enable_ato:
                sb_input = self.get_input(f">>> [ATO] Set Safety Baseline Distance (0.2 - 0.5m, Current: {self.controller.min_dist}m): ", str(self.controller.min_dist))
                try:
                    val = float(sb_input)
                    if 0.2 <= val <= 0.5:
                        self.controller.min_dist = val
                    else:
                        print(f"[!] [ATO] Out of bounds. Keeping {self.controller.min_dist}m.")
                except ValueError:
                    print(f"[!] [ATO] Invalid input. Keeping {self.controller.min_dist}m.")
            else:
                self.controller.min_dist = 0.3 # Baseline 强制锁死 0.3

            print(f"[*] [ATOM-UAV] Mode updated: ATO Enabled={self.enable_ato}, Safety Baseline={self.controller.min_dist}m")
            sys.stdout.flush()

            self.goals = []          
            self.start_poses = None  
            rospy.sleep(0.5) 

if __name__ == "__main__": 
    try:
        rospy.init_node('swarm_controller_node', anonymous=True)
        controller = SwarmControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass