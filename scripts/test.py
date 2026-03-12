#!/usr/bin/env python3

from flock_gpt.msg import Vector3StampedArray
import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import MarkerArray, Marker

RATE = 100

class SwarmSimulationNode():
    def __init__(self) -> None:
        # =================================================================
        # 🚀 [FMS 模块]: 引擎层容量初始化
        # 对比 baseline: 从固定的 drone_grid_shape 升级为动态数组。
        # =================================================================
        self.num_drones = 0
        self.swarm = np.array([]) 

        self.pose_publisher = rospy.Publisher('/swarm/poses', Vector3StampedArray, queue_size=1)
        self.viz_publisher = rospy.Publisher('/swarm/viz', MarkerArray, queue_size=1)
        rospy.Subscriber("/swarm/cmd_vel", Vector3StampedArray, self.callback_cmd, queue_size=1)

        # =================================================================
        # 🚀 [FMS 模块]: 动态监听与热重置
        # 对比 baseline: 添加了 Timer 定期监听 ROS 参数的变化，使得模拟器可以在不重启的情况下扩展机队。
        # =================================================================
        rospy.Timer(rospy.Duration(0.5), self.check_param_update)
        rospy.Timer(rospy.Duration(1.0/RATE), self.timer_publish)

    def check_param_update(self, event):
        # 🚀 [FMS 模块]: 优先检查是否有强制重置信号
        if rospy.has_param('/swarm_reset') and rospy.get_param('/swarm_reset'):
            if self.num_drones > 0:
                self.respawn_swarm(self.num_drones)
            rospy.set_param('/swarm_reset', False) 
            return 

        # 🚀 [FMS 模块]: 数量变化检查逻辑
        if rospy.has_param('/swarm_num_drones'):
            new_num = rospy.get_param('/swarm_num_drones')
            if new_num != self.num_drones and new_num > 0:
                self.num_drones = new_num
                self.respawn_swarm(new_num)

    def respawn_swarm(self, num):
        # 🚀 [FMS 模块]: 精准数量生成与截断。废弃了 baseline 的写死生成。
        side = int(np.ceil(np.sqrt(num)))
        x = np.linspace(1, 0.5*(side-1) + 1, side)
        y = np.linspace(1, 0.5*(side-1) + 1, side)
        xv, yv = np.meshgrid(x, y)
        
        full_grid = np.zeros((side * side, 3))
        full_grid[:, 0] = xv.flatten()
        full_grid[:, 1] = yv.flatten()
        full_grid[:, 2] = 1.0
        
        self.swarm = full_grid[:num] # 强行剔除矩阵生成多余的无人机，保证 50 就是 50
        rospy.loginfo(f"[Simulator] Respawned exactly {len(self.swarm)} drones.")

    def callback_cmd(self, msg: Vector3StampedArray):
        if len(self.swarm) == 0: return
        vels = np.zeros((len(msg.vector), 3))
        for i, v in enumerate(msg.vector):
            if i < self.num_drones: # 🚀 [FMS 模块]: 防止越界下发指令
                vels[i] = [v.x, v.y, v.z]
        self.swarm += vels * (1.0/RATE)

    def timer_publish(self, event):
        if len(self.swarm) == 0: return
        vector = Vector3StampedArray()
        viz = MarkerArray()
        for i, pose in enumerate(self.swarm):
            p = Vector3(pose[0], pose[1], pose[2])
            vector.vector.append(p)
            
            m = Marker()
            m.header.stamp, m.header.frame_id = rospy.Time.now(), 'map'
            m.type, m.id = 2, i
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pose
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r, m.color.a = 0.9, 1.0
            m.lifetime = rospy.Duration(0.1)
            viz.markers.append(m)
            
        self.pose_publisher.publish(vector)
        self.viz_publisher.publish(viz)

if __name__ == "__main__":
    rospy.init_node('swarm_simulation_node', anonymous=True)
    node = SwarmSimulationNode()
    rospy.spin()