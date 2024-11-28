import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import KDTree

# 定义节点类
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# 定义RRT*算法类
class RRTStar:
    def __init__(self, start, goal, obstacle_list, grid_size, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.node_list = [self.start]

    def get_random_node(self):
        x = random.randint(0, self.grid_size[0] - 1)
        y = random.randint(0, self.grid_size[1] - 1)
        return Node(x, y)

    def get_nearest_node_index(self, node_list, node):
        distances = [(node.x - n.x) ** 2 + (node.y - n.y) ** 2 for n in node_list]
        min_index = distances.index(min(distances))
        return min_index

    def check_collision(self, node):
        if node.x < 0 or node.x >= self.grid_size[0] or node.y < 0 or node.y >= self.grid_size[1]:
            return False
        if (node.x, node.y) in self.obstacle_list:
            return False
        return True

    def get_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = 50.0 * (np.log(nnode) / nnode) ** 0.5
        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_nodes = [self.node_list[dlist.index(i)] for i in dlist if i <= r ** 2]
        return near_nodes

    def planning(self):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node)
            if not self.check_collision(new_node):
                continue

            near_nodes = self.get_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_nodes)

            self.node_list.append(new_node)
            self.rewire(new_node, near_nodes)

            if self.calc_dist_to_goal(new_node.x, new_node.y) <= 1.0:
                final_node = self.steer(new_node, self.goal)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node):
        new_node = Node(to_node.x, to_node.y)
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        if d > 1.0:
            new_node.x = from_node.x + 1.0 * np.cos(theta)
            new_node.y = from_node.y + 1.0 * np.sin(theta)
        new_node.x = int(new_node.x)
        new_node.y = int(new_node.y)
        new_node.cost = from_node.cost + d
        new_node.parent = from_node
        return new_node

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return new_node

        dlist = []
        for node in near_nodes:
            t_node = self.steer(node, new_node)
            if self.check_collision(t_node):
                dlist.append(node.cost + self.calc_distance_and_angle(node, new_node)[0])
            else:
                dlist.append(float("inf"))

        min_cost = min(dlist)
        min_ind = dlist.index(min_cost)

        if min_cost == float("inf"):
            return new_node

        new_node.cost = min_cost
        new_node.parent = near_nodes[min_ind]

        return new_node

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            edge_node = self.steer(new_node, node)
            if not self.check_collision(edge_node):
                continue
            edge_node.cost = new_node.cost + self.calc_distance_and_angle(new_node, node)[0]

            if edge_node.cost < node.cost:
                node.x = edge_node.x
                node.y = edge_node.y
                node.cost = edge_node.cost
                node.parent = new_node

    def calc_dist_to_goal(self, x, y):
        return np.hypot(x - self.goal.x, y - self.goal.y)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

# 创建10x10栅格地图和障碍物
grid_size = (10, 10)
obstacle_list = [(3, 3), (3, 4), (3, 5), (6, 6), (7, 6), (8, 6), (5, 5)]

# 设置起点和终点
start = (1, 0)
goal = (9, 8)

# 实例化RRT*并进行路径规划
rrt_star = RRTStar(start, goal, obstacle_list, grid_size)

import time
init_time = time.time()
path = rrt_star.planning()
print(f'cost time:{time.time() - init_time}')
# 可视化采样点和路径
plt.figure(figsize=(10, 10))
for node in rrt_star.node_list:
    plt.plot(node.x, node.y, "go", markersize=2)
for (ox, oy) in obstacle_list:
    plt.plot(ox, oy, "ks", markersize=20)
if path is not None:
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], "-r")
plt.plot(start[0], start[1], "bs", markersize=10)
plt.plot(goal[0], goal[1], "gs", markersize=10)
plt.grid(True)
plt.axis("equal")
plt.show()
