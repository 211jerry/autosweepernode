#!/usr/bin/env python3
import os
import time
import rclpy
import yaml
import cv2
import numpy as np
import math
import heapq
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf_transformations import quaternion_from_euler
from rclpy.duration import Duration
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


class SweeperNode(BasicNavigator):
    def __init__(self, node_name='sweeper_node'):
        super().__init__(node_name)
        # ===================== 核心参数 =====================
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('robot_width', 0.5)          # 机器人宽度 (m)
        self.declare_parameter('step_size', 0.5)             # 路径点间距 (m)
        self.declare_parameter('overlap', 0.1)               # 路径重叠率 (m)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('inflation_radius', 0.5)      # 地图膨胀半径 (m)
        self.declare_parameter('contour_approx_eps', 0.01)  # 轮廓近似精度 (m)
        self.declare_parameter('wall_offset', 0.1)           # 沿墙偏移距离 (m)
        self.declare_parameter('invert_map', False)           # 地图像素反转开关
        # 可视化参数
        self.declare_parameter('map_output_path', './sweeper_path.png')
        self.declare_parameter('draw_contours', True)
        self.declare_parameter('draw_strips', True)
        self.declare_parameter('draw_waypoints', True)
        self.declare_parameter('draw_path_line', True)
        self.declare_parameter('save_debug_map', True)
        # ===================== 【新增】连通性优化核心参数 =====================
        self.declare_parameter('strip_valid_threshold', 0.95) # 条带有效自由空间占比
        self.declare_parameter('robot_safety_margin', 0.05)   # 机器人安全边距(m)
        self.declare_parameter('max_waypoint_gap', 0.5)       # 最大允许路径点间距(m)，超过则补点
        self.declare_parameter('connected_area_min_pix', 100) # 最小有效连通域像素数，过滤噪点
        # 获取参数
        self._load_parameters()
        # 校验地图文件
        if not self.map_yaml or not os.path.exists(self.map_yaml):
            self.get_logger().fatal(f'地图YAML文件不存在: {self.map_yaml}')
            exit(1)
        self.get_logger().info('参数加载完成，开始生成连通式全覆盖路径...')
        # 生成路径
        self.waypoints = self.generate_coverage_path()
        if not self.waypoints:
            self.get_logger().error('未生成任何有效路径点，节点退出')
            exit(1)
        # 最终路径连通性校验与优化
        self.waypoints = self.optimize_global_path_connectivity(self.waypoints)
        self.get_logger().info(f'路径优化完成，最终生成 {len(self.waypoints)} 个连通路径点')

    def _load_parameters(self):
        """集中加载参数，提升可维护性"""
        self.map_yaml = self.get_parameter('map_yaml').value
        self.robot_width = self.get_parameter('robot_width').value
        self.step_size = self.get_parameter('step_size').value
        self.overlap = self.get_parameter('overlap').value
        self.init_x = self.get_parameter('initial_x').value
        self.init_y = self.get_parameter('initial_y').value
        self.init_yaw = self.get_parameter('initial_yaw').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.contour_approx_eps = self.get_parameter('contour_approx_eps').value
        self.wall_offset = self.get_parameter('wall_offset').value
        self.invert_map = self.get_parameter('invert_map').value
        self.map_output_path = self.get_parameter('map_output_path').value
        self.draw_contours = self.get_parameter('draw_contours').value
        self.draw_strips = self.get_parameter('draw_strips').value
        self.draw_waypoints = self.get_parameter('draw_waypoints').value
        self.draw_path_line = self.get_parameter('draw_path_line').value
        self.save_debug_map = self.get_parameter('save_debug_map').value
        # 新增连通性参数
        self.strip_valid_threshold = self.get_parameter('strip_valid_threshold').value
        self.robot_safety_margin = self.get_parameter('robot_safety_margin').value
        self.max_waypoint_gap = self.get_parameter('max_waypoint_gap').value
        self.connected_area_min_pix = self.get_parameter('connected_area_min_pix').value

    # ===================== 【核心新增】连通性校验基础函数 =====================
    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> list:
        """Bresenham直线算法，生成两点之间的所有像素坐标，用于直线障碍检测"""
        pixels = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            pixels.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pixels

    def is_line_safe(self, binary_map: np.ndarray, p1: tuple, p2: tuple) -> bool:
        """
        核心校验：检查两点之间的直线是否完全无障碍物
        :param binary_map: 二值地图，1=自由空间，0=障碍
        :param p1: 起点像素坐标 (x1,y1)
        :param p2: 终点像素坐标 (x2,y2)
        :return: 直线无遮挡返回True，否则False
        """
        h, w = binary_map.shape
        line_pixels = self.bresenham_line(p1[0], p1[1], p2[0], p2[1])
        # 检查直线上所有像素是否都在自由空间
        for (x, y) in line_pixels:
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
            if binary_map[y, x] != 1:
                return False
        return True

    def split_free_connected_regions(self, binary_map: np.ndarray) -> list:
        """
        分割自由空间连通域，每个房间/独立区域单独规划，从根源避免跨障碍路径
        :param binary_map: 二值地图，1=自由空间，0=障碍
        :return: 连通域列表，每个元素是(区域掩码, 区域边界框)
        """
        # 自由空间掩码（1的区域）
        free_mask = (binary_map == 1).astype(np.uint8)
        # 连通域分析（8邻域，更贴合机器人移动）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_mask, connectivity=8)
        regions = []

        for label in range(1, num_labels):  # 跳过背景0
            area = stats[label, cv2.CC_STAT_AREA]
            # 过滤过小的噪点区域
            if area < self.connected_area_min_pix:
                continue
            # 生成区域掩码
            region_mask = (labels == label).astype(np.uint8)
            # 区域边界框
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, w, h)
            regions.append((region_mask, bbox, area))

        # 按面积从大到小排序，优先清扫大区域
        regions.sort(key=lambda x: x[2], reverse=True)
        self.get_logger().info(f'地图分割为 {len(regions)} 个有效自由连通域')
        return regions

    def a_star_search(self, binary_map: np.ndarray, start: tuple, end: tuple) -> list:
        """
        A*算法，寻找两点之间的最短无障路径，用于连通域之间的路径拼接
        :param binary_map: 二值地图
        :param start: 起点像素坐标 (x,y)
        :param end: 终点像素坐标 (x,y)
        :return: 路径像素坐标列表，无路径返回空
        """
        h, w = binary_map.shape
        # 边界校验
        if (start[0] < 0 or start[0] >= w or start[1] < 0 or start[1] >= h or
            end[0] < 0 or end[0] >= w or end[1] < 0 or end[1] >= h):
            return []
        if binary_map[start[1], start[0]] != 1 or binary_map[end[1], end[0]] != 1:
            return []
        # 8邻域移动
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        # 启发函数：欧氏距离
        def heuristic(p1, p2):
            return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        # 初始化
        open_heap = []
        heapq.heappush(open_heap, (0 + heuristic(start, end), 0, start))
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == end:
                # 回溯路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            closed.add(current)
            # 遍历邻域
            for dx, dy in neighbors:
                next_p = (current[0] + dx, current[1] + dy)
                # 边界和障碍校验
                if next_p[0] < 0 or next_p[0] >= w or next_p[1] < 0 or next_p[1] >= h:
                    continue
                if binary_map[next_p[1], next_p[0]] != 1:
                    continue
                # 计算g值
                new_g = current_g + math.hypot(dx, dy)
                if next_p not in g_score or new_g < g_score[next_p]:
                    g_score[next_p] = new_g
                    f_score = new_g + heuristic(next_p, end)
                    heapq.heappush(open_heap, (f_score, new_g, next_p))
                    came_from[next_p] = current
        # 无路径
        return []

    # ===================== 原有函数优化 =====================
    def parse_map_yaml(self, yaml_path):
        """解析地图YAML文件"""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'解析地图YAML失败: {str(e)}')
            return None, None, None, None, None
        map_dir = os.path.dirname(yaml_path)
        img_path = os.path.join(map_dir, data['image'])
        resolution = data['resolution']
        origin = data['origin']
        occupied_thresh = data.get('occupied_thresh', 0.65)
        free_thresh = data.get('free_thresh', 0.196)
        negate = data.get('negate', 0)
        if negate == 1:
            self.invert_map = True
            self.get_logger().warn('地图YAML检测到negate=1，自动开启反转')
        return img_path, resolution, origin, occupied_thresh, free_thresh

    def inflate_map(self, binary_map, inflation_radius_pix):
        """地图膨胀，扩大障碍区域"""
        if inflation_radius_pix <= 1:
            return binary_map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_radius_pix*2, inflation_radius_pix*2))
        inflated = cv2.erode(binary_map, kernel, iterations=1)
        closed = cv2.morphologyEx(inflated, cv2.MORPH_CLOSE, kernel)
        return closed

    def load_map_as_binary(self, img_path, occupied_thresh, free_thresh):
        """加载并二值化地图"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f'无法加载地图图像: {img_path}')
            return None
        self.original_map = img.copy()
        h, w = img.shape
        self.map_h, self.map_w = h, w
        # 地图反转适配
        if self.invert_map:
            img = 255 - img
            self.get_logger().info('已执行地图像素反转')
        # 二值化
        occ_thresh_val = int(occupied_thresh * 255)
        free_thresh_val = int(free_thresh * 255)
        binary = np.zeros_like(img, dtype=np.uint8)
        binary[img >= free_thresh_val] = 1
        binary[img < free_thresh_val] = 0
        # 保存调试地图
        if self.save_debug_map:
            cv2.imwrite('./debug_binary_map.png', (binary * 255).astype(np.uint8))
            self.get_logger().info('二值化调试地图已保存: ./debug_binary_map.png')
        # 地图膨胀
        inflation_radius_pix = int(self.inflation_radius / self.resolution)
        if inflation_radius_pix > 0:
            binary = self.inflate_map(binary, inflation_radius_pix)
            self.get_logger().info(f'地图膨胀完成，半径: {self.inflation_radius}m ({inflation_radius_pix}像素)')
            if self.save_debug_map:
                cv2.imwrite('./debug_inflated_map.png', (binary * 255).astype(np.uint8))
        return binary

    def detect_obstacle_contours(self, binary_map):
        """检测障碍轮廓"""
        obstacle_map = np.where(binary_map == 0, 255, 0).astype(np.uint8)
        if self.save_debug_map:
            cv2.imwrite('./debug_obstacle_map.png', obstacle_map)
        # 查找外轮廓
        contours, _ = cv2.findContours(obstacle_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        self.get_logger().info(f'原始检测到 {len(contours)} 个障碍轮廓')
        # 过滤无效轮廓
        filtered_contours = []
        approx_eps_pix = self.contour_approx_eps / self.resolution
        min_contour_length_pix = self.robot_width / self.resolution * 2
        max_contour_area = self.map_w * self.map_h * 0.9
        for cnt in contours:
            contour_length = cv2.arcLength(cnt, closed=True)
            if contour_length < min_contour_length_pix:
                continue
            contour_area = cv2.contourArea(cnt)
            if contour_area > max_contour_area:
                self.get_logger().warn(f'过滤地图外框轮廓，面积: {contour_area}像素')
                continue
            approx_cnt = cv2.approxPolyDP(cnt, approx_eps_pix, closed=True)
            filtered_contours.append(approx_cnt)
        self.get_logger().info(f'过滤后剩余 {len(filtered_contours)} 个有效障碍轮廓')
        self.detected_contours = filtered_contours
        return filtered_contours

    # ===================== 【重构】单连通域内弓字条带生成 =====================
    def generate_strips_for_region(self, binary_map: np.ndarray, region_mask: np.ndarray, bbox: tuple) -> list:
        """
        为单个连通域生成弓字形扫描条带，确保条带完全在区域内，不跨障碍
        :param binary_map: 全局二值地图
        :param region_mask: 当前连通域掩码
        :param bbox: 连通域边界框
        :return: 条带列表
        """
        x_min, y_min, w_region, h_region = bbox
        x_max = x_min + w_region
        y_max = y_min + h_region
        strips = []
        # 条带宽度
        strip_width_pix = max(1, int((self.robot_width - self.overlap) / self.resolution))
        self.strip_width_pix = strip_width_pix
        # 遍历生成水平弓字条带
        for strip_idx, strip_y_start in enumerate(range(y_min, y_max, strip_width_pix)):
            strip_y_end = min(y_max, strip_y_start + strip_width_pix)
            strip_center_y = (strip_y_start + strip_y_end) / 2
            # 条带区域：必须同时在自由空间和当前连通域内
            strip_mask = region_mask[strip_y_start:strip_y_end, x_min:x_max] & binary_map[strip_y_start:strip_y_end, x_min:x_max]
            # 计算每一列的有效占比
            col_valid_ratio = np.sum(strip_mask, axis=0) / strip_width_pix
            valid_cols = np.where(col_valid_ratio >= self.strip_valid_threshold)[0]
            if len(valid_cols) == 0:
                continue
            # 拆分连续有效区间
            valid_cols += x_min  # 映射回全局坐标
            continuous_intervals = np.split(valid_cols, np.where(np.diff(valid_cols) != 1)[0] + 1)
            for interval in continuous_intervals:
                if len(interval) < int(self.step_size / self.resolution):
                    continue
                start_x = interval[0]
                end_x = interval[-1]
                # 弓字形换向
                yaw = 0.0 if strip_idx % 2 == 0 else math.pi
                strips.append((yaw, start_x, end_x, strip_center_y))
        self.get_logger().info(f'单连通域生成 {len(strips)} 个有效条带')
        return strips

    # ===================== 【重构】单连通域内路径点生成，确保相邻点连通 =====================
    def generate_waypoints_for_region(self, binary_map: np.ndarray, strips: list) -> list:
        """
        为单个连通域生成路径点，严格保证相邻点之间直线无障
        :param binary_map: 二值地图
        :param strips: 条带列表
        :return: 区域内路径点列表 (x_pix, y_pix, yaw)
        """
        waypoints = []
        step_pix = max(1, int(self.step_size / self.resolution))
        safety_radius_pix = int((self.robot_width/2 + self.robot_safety_margin) / self.resolution)
        h, w = binary_map.shape

        for strip_idx, (yaw, start_x, end_x, center_y) in enumerate(strips):
            is_even_strip = (strip_idx % 2 == 0)
            # 弓字形扫描方向
            if is_even_strip:
                scan_range = range(start_x, end_x + 1, step_pix)
                current_yaw = yaw
            else:
                scan_range = range(end_x, start_x - 1, -step_pix)
                current_yaw = yaw
            # 生成条带内路径点
            strip_points = []
            y_pix = int(round(center_y))
            for x_pix in scan_range:
                # 边界校验
                if x_pix < 0 or x_pix >= w or y_pix < 0 or y_pix >= h:
                    continue
                # 机身邻域安全校验
                x_min = max(0, x_pix - safety_radius_pix)
                x_max = min(w, x_pix + safety_radius_pix)
                y_min = max(0, y_pix - safety_radius_pix)
                y_max = min(h, y_pix + safety_radius_pix)
                if np.sum(binary_map[y_min:y_max, x_min:x_max]) != (x_max-x_min)*(y_max-y_min):
                    continue
                strip_points.append((x_pix, y_pix, current_yaw))
            # 条带内无有效点，跳过
            if not strip_points:
                continue
            # ===================== 核心：条带间连接点连通性校验 =====================
            if waypoints:
                # 上一条带的最后一个点
                last_point = waypoints[-1]
                # 当前条带的第一个点
                first_point = strip_points[0]
                # 检查两点之间是否直线连通
                if not self.is_line_safe(binary_map, (last_point[0], last_point[1]), (first_point[0], first_point[1])):
                    # 不连通，尝试反向连接当前条带的最后一个点
                    reverse_first_point = strip_points[-1]
                    if self.is_line_safe(binary_map, (last_point[0], last_point[1]), (reverse_first_point[0], reverse_first_point[1])):
                        # 反向当前条带，换向
                        strip_points = strip_points[::-1]
                        strip_points = [(p[0], p[1], p[2] + math.pi) for p in strip_points]
                    else:
                        # 仍不连通，用A*生成连接路径
                        self.get_logger().warn(f'条带{strip_idx}与前一条带不连通，生成补全路径')
                        connect_path = self.a_star_search(binary_map, (last_point[0], last_point[1]), (first_point[0], first_point[1]))
                        if connect_path:
                            # 补全路径点，朝向沿路径方向
                            for i in range(1, len(connect_path)):
                                px, py = connect_path[i]
                                dx = px - connect_path[i-1][0]
                                dy = py - connect_path[i-1][1]
                                point_yaw = math.atan2(dy, dx)
                                waypoints.append((px, py, point_yaw))
            # 添加当前条带的路径点
            waypoints.extend(strip_points)
        return waypoints

    # ===================== 【核心】全局路径连通性优化 =====================
    def optimize_global_path_connectivity(self, waypoints: list) -> list:
        """
        全局路径最终优化：
        1. 校验所有相邻点的直线连通性，不连通则补全路径
        2. 合并共线冗余点，减少导航点数量
        3. 过滤过近的重复点
        """
        if len(waypoints) < 2:
            return waypoints
        self.get_logger().info('开始全局路径连通性优化...')
        h, w = self.binary_map.shape
        step_pix = max(1, int(self.step_size / self.resolution))
        optimized = []
        # 第一步：相邻点连通性校验与补全
        for i in range(len(waypoints)):
            current_wp = waypoints[i]
            current_pix = self.world2pix(current_wp[0], current_wp[1])
            current_yaw = current_wp[2]
            if i == 0:
                optimized.append(current_wp)
                continue
            # 上一个点
            last_wp = optimized[-1]
            last_pix = self.world2pix(last_wp[0], last_wp[1])
            # 检查直线连通性
            if self.is_line_safe(self.binary_map, last_pix, current_pix):
                # 连通，直接添加
                optimized.append(current_wp)
            else:
                # 不连通，A*补全路径
                self.get_logger().warn(f'路径点{i}与前一点不连通，补全路径')
                connect_path = self.a_star_search(self.binary_map, last_pix, current_pix)
                if connect_path and len(connect_path) > 2:
                    # 按步长采样补全点
                    for j in range(step_pix, len(connect_path)-1, step_pix):
                        px, py = connect_path[j]
                        # 计算朝向
                        dx = px - connect_path[j-1][0]
                        dy = py - connect_path[j-1][1]
                        point_yaw = math.atan2(dy, dx)
                        # 像素转世界坐标
                        world_x, world_y = self.pix2world(px, py)
                        optimized.append((world_x, world_y, point_yaw))
                # 添加当前点
                optimized.append(current_wp)
        # 第二步：过滤过近的重复点
        filtered = []
        min_distance = self.step_size * 0.5
        for wp in optimized:
            if not filtered:
                filtered.append(wp)
                continue
            last = filtered[-1]
            dx = wp[0] - last[0]
            dy = wp[1] - last[1]
            if math.hypot(dx, dy) >= min_distance:
                filtered.append(wp)
        # 第三步：合并共线点，减少导航点
        if len(filtered) < 3:
            final_waypoints = filtered
        else:
            final_waypoints = [filtered[0]]
            for i in range(1, len(filtered)-1):
                prev = final_waypoints[-1]
                curr = filtered[i]
                next_p = filtered[i+1]
                # 计算三点是否共线
                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                dx2 = next_p[0] - curr[0]
                dy2 = next_p[1] - curr[1]
                # 叉积判断共线
                cross = dx1 * dy2 - dy1 * dx2
                if abs(cross) > 1e-6:
                    final_waypoints.append(curr)
            # 添加最后一个点
            final_waypoints.append(filtered[-1])
        # 修正最后一个点的朝向为初始朝向
        if final_waypoints:
            final_waypoints[-1] = (final_waypoints[-1][0], final_waypoints[-1][1], self.init_yaw)
        self.get_logger().info(f'全局路径优化完成，路径点从 {len(waypoints)} 精简为 {len(final_waypoints)}')
        return final_waypoints

    def world2pix(self, x: float, y: float) -> tuple:
        """世界坐标转像素坐标，严格对齐ROS地图规则"""
        x_pix = (x - self.origin[0]) / self.resolution
        y_pix = self.map_h - 1 - ((y - self.origin[1]) / self.resolution)
        x_pix = np.clip(int(round(x_pix)), 0, self.map_w-1)
        y_pix = np.clip(int(round(y_pix)), 0, self.map_h-1)
        return x_pix, y_pix

    def pix2world(self, x_pix: int, y_pix: int) -> tuple:
        """像素坐标转世界坐标"""
        world_x = self.origin[0] + (x_pix + 0.5) * self.resolution
        world_y = self.origin[1] + (self.map_h - 1 - y_pix + 0.5) * self.resolution
        return world_x, world_y

    def filter_waypoints(self, waypoints, min_distance=0.05):
        """过滤重复点（兼容原有逻辑）"""
        if len(waypoints) < 2:
            return waypoints
        filtered = [waypoints[0]]
        for wp in waypoints[1:]:
            dx = wp[0] - filtered[-1][0]
            dy = wp[1] - filtered[-1][1]
            if math.hypot(dx, dy) >= min_distance:
                filtered.append(wp)
        return filtered

    def draw_path_on_map(self, waypoints):
        """路径可视化，优化连通路径显示"""
        draw_img = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)
        h, w = draw_img.shape[:2]
        # 颜色定义（BGR）
        COLOR_CONTOUR = (0, 0, 255)
        COLOR_STRIP = (0, 255, 0)
        COLOR_PATH_LINE = (255, 200, 200)
        COLOR_WAYPOINT = (255, 0, 0)
        COLOR_START = (0, 255, 0)
        COLOR_END = (0, 0, 255)
        # 绘制障碍轮廓
        if self.draw_contours and hasattr(self, 'detected_contours'):
            cv2.drawContours(draw_img, self.detected_contours, -1, COLOR_CONTOUR, 2)
        # 绘制扫描条带
        if self.draw_strips and hasattr(self, 'generated_strips'):
            for (direction, start, end, center) in self.generated_strips:
                if abs(direction) < math.pi/4 or abs(direction) > 3*math.pi/4:
                    cv2.line(draw_img, (int(start), int(center)), (int(end), int(center)), COLOR_STRIP, 1)
                else:
                    cv2.line(draw_img, (int(center), int(start)), (int(center), int(end)), COLOR_STRIP, 1)
        # 绘制路径
        if waypoints:
            pix_waypoints = [self.world2pix(x, y) for x, y, _ in waypoints]
            # 绘制路径连线（完整连通路径）
            if self.draw_path_line and len(pix_waypoints)>=2:
                for i in range(1, len(pix_waypoints)):
                    cv2.line(draw_img, pix_waypoints[i-1], pix_waypoints[i], COLOR_PATH_LINE, 2)
            # 绘制路径点
            if self.draw_waypoints:
                step = max(1, len(pix_waypoints)//200)
                for idx, (x_pix, y_pix) in enumerate(pix_waypoints):
                    if idx % step == 0 and 0 <= x_pix < w and 0 <= y_pix < h:
                        cv2.circle(draw_img, (x_pix, y_pix), 2, COLOR_WAYPOINT, -1)
            # 绘制起点和终点
            cv2.circle(draw_img, pix_waypoints[0], 6, COLOR_START, -1)
            cv2.circle(draw_img, pix_waypoints[-1], 6, COLOR_END, -1)
        # 保存图片
        try:
            cv2.imwrite(self.map_output_path, draw_img)
            self.get_logger().info(f'路径可视化图已保存: {os.path.abspath(self.map_output_path)}')
        except Exception as e:
            self.get_logger().error(f'保存路径图片失败: {str(e)}')

    # ===================== 【重构】主路径生成函数 =====================
    def generate_coverage_path(self):
        """生成连通式全覆盖路径主函数"""
        # 解析地图
        parse_result = self.parse_map_yaml(self.map_yaml)
        if None in parse_result:
            return []
        img_path, self.resolution, self.origin, occ_thresh, free_thresh = parse_result
        self.get_logger().info(f'地图: {img_path}, 分辨率: {self.resolution:.4f} m/像素')
        # 加载二值地图
        binary = self.load_map_as_binary(img_path, occ_thresh, free_thresh)
        if binary is None:
            return []
        self.binary_map = binary  # 全局保存，用于后续校验
        h, w = binary.shape
        self.get_logger().info(f'地图尺寸: {w}x{h} 像素 (实际: {w*self.resolution:.2f}m x {h*self.resolution:.2f}m)')
        # 检测障碍轮廓
        contours = self.detect_obstacle_contours(binary)
        # 分割自由空间连通域
        regions = self.split_free_connected_regions(binary)
        if not regions:
            self.get_logger().error('未检测到有效自由空间')
            return []
        # 全局路径点
        global_waypoints = []
        all_strips = []
        # 遍历每个连通域生成路径
        for region_idx, (region_mask, bbox, _) in enumerate(regions):
            self.get_logger().info(f'处理第 {region_idx+1}/{len(regions)} 个连通域')
            # 生成条带
            strips = self.generate_strips_for_region(binary, region_mask, bbox)
            all_strips.extend(strips)
            # 生成区域内路径点
            region_waypoints_pix = self.generate_waypoints_for_region(binary, strips)
            if not region_waypoints_pix:
                self.get_logger().warn(f'第 {region_idx+1} 个连通域无有效路径点，跳过')
                continue
            # 像素转世界坐标
            region_waypoints = []
            for (x_pix, y_pix, yaw) in region_waypoints_pix:
                world_x, world_y = self.pix2world(x_pix, y_pix)
                region_waypoints.append((world_x, world_y, yaw))
            # 连通域之间的路径拼接
            if global_waypoints:
                self.get_logger().info(f'拼接第 {region_idx+1} 个连通域路径')
                last_point = global_waypoints[-1]
                first_point = region_waypoints[0]
                last_pix = self.world2pix(last_point[0], last_point[1])
                first_pix = self.world2pix(first_point[0], first_point[1])
                # 检查连通性，不连通则用A*补全
                if not self.is_line_safe(binary, last_pix, first_pix):
                    connect_path = self.a_star_search(binary, last_pix, first_pix)
                    if connect_path:
                        self.get_logger().info(f'生成连通域之间的过渡路径，长度: {len(connect_path)}')
                        for i in range(1, len(connect_path)-1, max(1, int(self.step_size/self.resolution))):
                            px, py = connect_path[i]
                            dx = px - connect_path[i-1][0]
                            dy = py - connect_path[i-1][1]
                            yaw = math.atan2(dy, dx)
                            wx, wy = self.pix2world(px, py)
                            global_waypoints.append((wx, wy, yaw))
            # 添加当前连通域路径
            global_waypoints.extend(region_waypoints)
        # 保存全局条带用于可视化
        self.generated_strips = all_strips
        # 绘制路径
        self.draw_path_on_map(global_waypoints)
        return global_waypoints

    # ===================== 导航相关函数（保留原有逻辑，兼容ROS2）=====================
    def get_pose_by_xyyaw(self, x, y, yaw):
        """构造PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def _is_nav2_active(self):
        """检查Nav2是否激活"""
        try:
            action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
            return action_client.wait_for_server(timeout_sec=0.1)
        except Exception:
            return False

    def init_robot_pose(self):
        """设置初始位姿"""
        init_pose = self.get_pose_by_xyyaw(self.init_x, self.init_y, self.init_yaw)
        self.setInitialPose(init_pose)
        timeout_sec = 30.0
        start_time = time.time()
        self.get_logger().info('等待Nav2激活...')
        while not self._is_nav2_active():
            if time.time() - start_time > timeout_sec:
                self.get_logger().fatal(f'等待Nav2激活超时{timeout_sec}秒，退出')
                exit(1)
            time.sleep(0.5)
        self.get_logger().info('Nav2已激活，初始位姿设置完成')

    def nav_to_pose(self, target_pose, index, total):
        """导航到单个目标点"""
        self.get_logger().info(f'[{index}/{total}] 前往目标点: ({target_pose.pose.position.x:.2f}m, {target_pose.pose.position.y:.2f}m)')
        self.goToPose(target_pose)
        nav_timeout = 60.0
        start_time = time.time()
        last_log_time = time.time()
        while not self.isTaskComplete():
            if time.time() - start_time > nav_timeout:
                self.get_logger().warn(f'导航超时{nav_timeout}秒，终止当前任务')
                self.cancelTask()
                return False
            if time.time() - last_log_time > 1.0:
                feedback = self.getFeedback()
                if feedback:
                    remaining = Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9
                    self.get_logger().info(f'预计剩余时间: {remaining:.1f}s')
                last_log_time = time.time()
            rclpy.spin_once(self, timeout_sec=0.1)
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('到达目标点 ✓')
            return True
        else:
            self.get_logger().warn(f'导航失败，结果: {result} ✗')
            return False

    def run(self):
        """执行清扫任务"""
        try:
            self.init_robot_pose()
            total = len(self.waypoints)
            self.get_logger().info(f'开始全覆盖清扫，共 {total} 个连通路径点')
            success_count = 0
            for i, (x, y, yaw) in enumerate(self.waypoints, 1):
                pose = self.get_pose_by_xyyaw(x, y, yaw)
                success = self.nav_to_pose(pose, i, total)
                if success:
                    success_count += 1
                else:
                    self.get_logger().error(f'跳过目标点 {i}/{total}')
            self.get_logger().info(f'清扫完成！成功到达 {success_count}/{total} 个路径点')
        except Exception as e:
            self.get_logger().error(f'清扫过程出错: {str(e)}', exc_info=True)
            raise

def main():
    rclpy.init()
    try:
        node = SweeperNode()
        node.run()
    except Exception as e:
        print(f'节点启动失败: {str(e)}', exc_info=True)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
