import sys
import copy
from pathlib import Path
import os
import pickle
import numpy as np
import math
from geopy.distance import great_circle
from mpmath.math2 import sqrt2
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
# 添加项目根目录和 MixVPR 目录到路径
project_root = str(Path(__file__).parent)
sys.path.extend([project_root, os.path.join(project_root, 'gim-main')])
from hight_load import ElevationQuery


# 四元数实用函数
def rotation_vector_to_quaternion(rvec):
    """将旋转向量转换为四元数"""
    if np.linalg.norm(rvec) < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    rotation = Rotation.from_rotvec(rvec)
    return rotation.as_quat()


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    q_normalized = q / np.linalg.norm(q)
    rotation = Rotation.from_quat(q_normalized)
    return rotation.as_matrix()


def quaternion_residuals(x, data, K, dist_coeffs, rotation_weight=100.0):
    """使用四元数表示的重投影残差函数"""
    # 提取参数：四元数 + 平移向量
    q = x[:4]
    translation = x[4:7]

    # 归一化四元数
    q_normalized = q / np.linalg.norm(q)

    # 将四元数转换为旋转矩阵
    R = quaternion_to_rotation_matrix(q_normalized)

    # 创建转换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = translation

    # 初始化残差存储
    errors = []

    # 相机参数
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist_coeffs

    map_points = [[item[0]['east'], item[0]['north']] for item in data]
    centroid = np.mean(np.array(map_points), axis=0)
    distance_map = [(np.array(point) - centroid).tolist() for point in map_points]
    max_mag = max(np.linalg.norm(v) for v in distance_map) if distance_map else 1
    normalized_map = [(np.array(v) / max_mag).tolist() if max_mag > 0 else v for v in distance_map]

    for idx, obs in enumerate(data):
        try:
            # 世界坐标系 → 相机坐标系
            world_point = obs[0]
            point_w = np.array([world_point['east'], world_point['north'], world_point['up'], 1.0])
            point_c = transform @ point_w

            # 深度检查
            z = point_c[2]
            if z <= 0.01:
                errors.extend([1e6, 1e6])
                continue

            x_norm, y_norm = point_c[0] / z, point_c[1] / z
            r2 = x_norm ** 2 + y_norm ** 2

            # 计算径向畸变因子（与坐标值相乘）
            radial = 1.0 + k1 * r2 + k2 * r2 ** 2

            # 计算切向畸变（直接添加到扭曲坐标上）
            tang_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
            tang_y = 2.0 * p2 * x_norm * y_norm + p1 * (r2 + 2 * y_norm ** 2)

            # 应用畸变：先乘径向因子，再加切向项
            x_distorted = x_norm * radial + tang_x
            y_distorted = y_norm * radial + tang_y

            # 转换到像素坐标
            u_pred = fx * x_distorted + cx
            v_pred = fy * y_distorted + cy

            # 计算残差
            u_obs, v_obs = obs[1]
            errors.extend([(u_pred - u_obs)*((abs(normalized_map[idx][0])**3)*5), (v_pred - v_obs)*((abs(normalized_map[idx][1])**3)*5)])

        except Exception as e:
            errors.extend([1e6, 1e6])

    # 添加正则化约束
    ortho_residual = rotation_weight * (np.linalg.norm(q) - 1.0) ** 2
    angular_velocity = np.linalg.norm(q - np.array([1, 0, 0, 0])) * rotation_weight * 0.1
    errors.append(ortho_residual)
    errors.append(angular_velocity)

    return np.array(errors)


def calculate_relative_position(lat1, lon1, h1, lat2, lon2, h2):
    """计算点2相对于点1的三维相对位置"""
    # 平面位置计算
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    dist_2d = great_circle(point1, point2).meters

    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    y = math.sin(dlon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    bearing = (bearing_deg + 360) % 360

    north = dist_2d * math.cos(bearing_rad)
    east = dist_2d * math.sin(bearing_rad)

    # 三维计算
    dh = h2 - h1
    dist_3d = math.sqrt(dist_2d ** 2 + dh ** 2)
    elevation = math.degrees(math.atan2(dh, dist_2d))

    return {
        'east': east,
        'north': north,
        'up': dh
    }


def pixel_to_lonlat(x_pixel, y_pixel, geotrans, size):
    """计算像素位置的地理坐标"""
    width, height = size[0], size[1]
    # if not (0 <= x_pixel < width and 0 <= y_pixel < height):
    #     raise ValueError(f"坐标超出影像范围: {width}x{height}")
    longitude = float(geotrans[0]) + x_pixel * float(geotrans[1])
    latitude = float(geotrans[3]) + y_pixel * float(geotrans[5])
    return longitude, latitude


def get_precise_elevation(query_system, lon, lat, attempt_surrounding=True):
    """获取精确高程"""
    try:
        elev = query_system.get_elevation(lon, lat)
        if elev is not None:
            return elev

        # 尝试在周边点查询
        if attempt_surrounding:
            offsets = [0.00001, -0.00001]  # 约1米偏移
            elevations = []
            for dx in offsets:
                for dy in offsets:
                    e = query_system.get_elevation(lon + dx, lat + dy)
                    if e is not None:
                        elevations.append(e)
            if elevations:
                return np.mean(elevations)
    except Exception as e:
        print(f"高程查询错误 ({lon}, {lat}): {str(e)}")

    # 返回默认高程值
    return 0.0

def visualize_results(result, positions, K, dist_coeffs, metadata, image_path=None):
    """Visualize optimization results and error distributions"""
    # 创建一个新的figure对象用于显示结果
    fig1 = plt.figure(figsize=(15, 12))

    # 1. 水平位置误差
    ax1 = fig1.add_subplot(2, 2, 1)
    true_pos = result['true_position']
    pred_pos = result['predicted_position']

    ax1.quiver(0, 0, true_pos[0], true_pos[1], angles='xy', scale_units='xy',
               scale=1, color='g', width=0.005, label='True Position')
    ax1.quiver(0, 0, pred_pos[0], pred_pos[1], angles='xy', scale_units='xy',
               scale=1, color='r', width=0.003, label='Predicted Position')
    error_vec = [pred_pos[0] - true_pos[0], pred_pos[1] - true_pos[1]]
    ax1.quiver(true_pos[0], true_pos[1], error_vec[0], error_vec[1],
               angles='xy', scale_units='xy', scale=1, color='b', width=0.002,
               label=f'Error Vector ({result["distance_error"]:.1f}m)')

    # 标记中心点和目标点
    ax1.scatter(0, 0, s=50, c='purple', marker='o', label='Map Center')
    ax1.scatter(true_pos[0], true_pos[1], s=100, c='g', marker='*', label='True Target')
    ax1.scatter(pred_pos[0], pred_pos[1], s=100, c='r', marker='X', label='Predicted Target')

    ax1.set_title(f"Horizontal Position Error: {result['distance_error']:.2f} meters")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel("East Direction (meters)")
    ax1.set_ylabel("North Direction (meters)")

    # 2. 高度误差
    ax2 = fig1.add_subplot(2, 2, 2)
    z_error = pred_pos[2] - true_pos[2]
    ax2.barh(0, z_error, color='b' if abs(z_error) < 5 else 'r')
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title(f"Height Error: {z_error:.2f} meters")
    ax2.set_xlabel("Error Value (meters)")
    ax2.set_yticks([])
    ax2.grid(axis='x')

    # 3. 重投影误差分布
    ax3 = fig1.add_subplot(2, 2, 3)
    errors = []
    q_opt = result['optimized_params'][:4]
    if np.linalg.norm(q_opt) < 1e-6:
        q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q_opt_normalized = q_opt / np.linalg.norm(q_opt)
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = result['optimized_params'][4:7]

    for obs in positions:
        world_point = obs[0]
        pixel_obs = obs[1]
        point_w = np.array([world_point['east'], world_point['north'], world_point['up'], 1.0])

        # 世界坐标 -> 相机坐标
        point_c = R_opt @ point_w[:3] + t_opt

        # 深度检查
        z = point_c[2]
        if z <= 0.01:
            continue

        # 投影到图像平面
        x_norm, y_norm = point_c[0] / z, point_c[1] / z
        r2 = x_norm ** 2 + y_norm ** 2
        k1, k2, p1, p2 = dist_coeffs
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2
        tang_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
        tang_y = p1 * (r2 + 2 * y_norm ** 2) + 2.0 * p2 * x_norm * y_norm

        u_pred = K[0, 0] * (x_norm * radial + tang_x) + K[0, 2]
        v_pred = K[1, 1] * (y_norm * radial + tang_y) + K[1, 2]

        errors.append(math.sqrt((u_pred - pixel_obs[0]) ** 2 + (v_pred - pixel_obs[1]) ** 2))

    if errors:
        ax3.hist(errors, bins=50, color='skyblue', edgecolor='black')
        ax3.axvline(x=np.median(errors), color='r', linestyle='dashed', linewidth=1,
                    label=f'Median: {np.median(errors):.2f}px')
        ax3.set_title(
            f"Reprojection Error Distribution (Points:{len(positions)})\nMean: {np.mean(errors):.2f}px, Max: {np.max(errors):.2f}px")
    else:
        ax3.text(0.5, 0.5, "No valid reprojection error data", ha='center', va='center')
        ax3.set_title("Reprojection Error Data Missing")

    ax3.set_xlabel("Pixel Error (px)")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 3D位置可视化
    ax4 = fig1.add_subplot(2, 2, 4, projection='3d')

    # 所有地图点
    all_points = []
    for obs in positions:
        if isinstance(obs[0], dict) and 'east' in obs[0] and 'north' in obs[0] and 'up' in obs[0]:
            all_points.append([obs[0]['east'], obs[0]['north'], obs[0]['up']])

    if all_points:
        all_points = np.array(all_points)
        ax4.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                    s=10, c='b', alpha=0.2, label='Map Points')

    # 相机位置
    ax4.scatter(0, 0, 0, s=100, c='purple', marker='o', label='Map Center')
    ax4.scatter(true_pos[0], true_pos[1], true_pos[2], s=150, c='g', marker='*', label='True Position')
    ax4.scatter(pred_pos[0], pred_pos[1], pred_pos[2], s=150, c='r', marker='X', label='Predicted Position')

    # 绘制方向向量
    cam_dir = R_opt @ np.array([0, 0, 1]) * 100
    ax4.quiver(pred_pos[0], pred_pos[1], pred_pos[2],
               cam_dir[0], cam_dir[1], cam_dir[2],
               color='m', label='Camera Orientation')

    # 设置坐标轴标签
    ax4.set_xlabel('East Direction (meters)')
    ax4.set_ylabel('North Direction (meters)')
    ax4.set_zlabel('Height (meters)')
    ax4.set_title(f"3D Position Visualization - {metadata['image_name']}")
    ax4.legend()

    fig1.tight_layout()

    # 保存并显示结果
    if 'image_name' in metadata:
        filename = metadata['image_name'].replace('/', '_') + '_result.png'
        fig1.savefig(filename, dpi=150)
        print(f"Visualization results saved to: {filename}")

    # 显示第一张图但不阻塞
    plt.show(block=False)

    # 5. XY方向重投影误差可视化 - 添加绝对值中位数显示
    reproj_data = result['reprojection_errors']
    u_errors = reproj_data['x_direction']['errors']
    v_errors = reproj_data['y_direction']['errors']

    # 创建第二张图
    fig2 = None

    if u_errors and v_errors:
        # 计算中位数和绝对值中位数
        median_u = np.median(u_errors)
        median_v = np.median(v_errors)
        median_abs_u = np.median(np.abs(u_errors))
        median_abs_v = np.median(np.abs(v_errors))

        # 创建标签
        median_label_u = f"Median: {median_u:.2f}px"
        median_label_v = f"Median: {median_v:.2f}px"
        median_abs_label_u = f"Median Abs: {median_abs_u:.2f}px"
        median_abs_label_v = f"Median Abs: {median_abs_v:.2f}px"

        # 创建第二张图
        fig2 = plt.figure(figsize=(15, 10))

        # X方向误差分布
        ax5 = fig2.add_subplot(2, 2, 1)
        ax5.hist(u_errors, bins=50, color='red', alpha=0.7)
        ax5.axvline(0, color='k', linestyle='--')

        # 添加中位数竖线和标签
        ax5.axvline(median_u, color='darkgreen', linestyle='-', linewidth=2,
                    alpha=0.9, label=median_label_u)
        # 添加绝对值中位数竖线
        ax5.axvline(median_abs_u, color='darkorange', linestyle='-', linewidth=2,
                    alpha=0.9, label=median_abs_label_u)
        ax5.axvline(-median_abs_u, color='darkorange', linestyle='-', linewidth=2,
                    alpha=0.9, label=f"-{median_abs_label_u}")

        # 在中位数周围添加半透明区域
        ax5.axvspan(median_u - 0.05, median_u + 0.05, color='green', alpha=0.1)

        ax5.set_title(
            f"X-direction Reprojection Errors\nMean Abs: {reproj_data['x_direction']['mean_abs']:.2f}px, "
            f"Max Abs: {reproj_data['x_direction']['max_abs']:.2f}px")
        ax5.set_xlabel("Error in X (pixels)")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best')

        # Y方向误差分布
        ax6 = fig2.add_subplot(2, 2, 2)
        ax6.hist(v_errors, bins=50, color='blue', alpha=0.7)
        ax6.axvline(0, color='k', linestyle='--')

        # 添加中位数竖线和标签
        ax6.axvline(median_v, color='darkgreen', linestyle='-', linewidth=2,
                    alpha=0.9, label=median_label_v)
        # 添加绝对值中位数竖线
        ax6.axvline(median_abs_v, color='darkorange', linestyle='-', linewidth=2,
                    alpha=0.9, label=median_abs_label_v)
        ax6.axvline(-median_abs_v, color='darkorange', linestyle='-', linewidth=2,
                    alpha=0.9, label=f"-{median_abs_label_v}")

        # 在中位数周围添加半透明区域
        ax6.axvspan(median_v - 0.05, median_v + 0.05, color='green', alpha=0.1)

        ax6.set_title(
            f"Y-direction Reprojection Errors\nMean Abs: {reproj_data['y_direction']['mean_abs']:.2f}px, "
            f"Max Abs: {reproj_data['y_direction']['max_abs']:.2f}px")
        ax6.set_xlabel("Error in Y (pixels)")
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='best')

        # X vs Y误差散点图
        ax7 = fig2.add_subplot(2, 2, 3)
        ax7.scatter(u_errors, v_errors, alpha=0.6, c=np.arange(len(u_errors)), cmap='viridis')
        ax7.axhline(0, color='k', linestyle='--')
        ax7.axvline(0, color='k', linestyle='--')

        # 添加中位数位置标记
        ax7.scatter(median_u, median_v, s=150, c='lime', marker='X',
                    edgecolors='darkgreen', linewidth=2, label='Median Error')
        ax7.scatter(median_abs_u, median_abs_v, s=150, c='orange', marker='s',
                    edgecolors='darkorange', linewidth=2, label='Median Abs Error')

        # 添加中位数线
        ax7.plot([min(u_errors), max(u_errors)], [median_v, median_v],
                 'g--', alpha=0.4, label='Y Median')
        ax7.plot([median_u, median_u], [min(v_errors), max(v_errors)],
                 'g--', alpha=0.4, label='X Median')

        ax7.set_title("X vs Y Reprojection Errors with Median")
        ax7.set_xlabel("X Error (pixels)")
        ax7.set_ylabel("Y Error (pixels)")
        ax7.grid(True, alpha=0.3)
        ax7.legend()

        # X-Y误差向量场
        ax8 = fig2.add_subplot(2, 2, 4)
        for i, (u_err, v_err) in enumerate(zip(u_errors, v_errors)):
            # 为中位数点使用不同的颜色
            if abs(u_err - median_u) < 1e-5 and abs(v_err - median_v) < 1e-5:
                color = 'lime'
                alpha = 1.0
                width = 0.006
            elif abs(u_err) >= median_abs_u or abs(v_err) >= median_abs_v:
                color = 'orange'
                alpha = 0.8
                width = 0.004
            else:
                color = 'green'
                alpha = 0.4
                width = 0.003

            ax8.quiver(0, 0, u_err, v_err, angles='xy', scale_units='xy', scale=1,
                       width=width, color=color, alpha=alpha)

        # 绘制中位数向量
        ax8.quiver(0, 0, median_u, median_v, angles='xy', scale_units='xy', scale=1,
                   width=0.005, color='darkgreen', label=f'Median Vector ({median_u:.2f},{median_v:.2f})')

        # 绘制绝对值中位数向量
        ax8.quiver(0, 0, median_abs_u, median_abs_v, angles='xy', scale_units='xy', scale=1,
                   width=0.005, color='darkorange', label=f'Median Abs Vector ({median_abs_u:.2f},{median_abs_v:.2f})')

        ax8.axhline(0, color='k')
        ax8.axvline(0, color='k')
        ax8.set_title("Error Vector Field with Median Position")
        ax8.set_xlabel("X Error")
        ax8.set_ylabel("Y Error")
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='best')

        fig2.tight_layout()
        error_filename = f"{metadata['image_name']}_xy_errors.png".replace('/', '_')
        fig2.savefig(error_filename, dpi=150)
        print(f"XY error visualization saved to: {error_filename}")

        # 显示第二张图但不阻塞
        plt.show(block=False)

    # 6. 可选：原始图像与投影点
    fig3 = None
    if image_path and os.path.exists(image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 创建第三张图
                fig3, ax9 = plt.subplots(figsize=(12, 8))
                ax9.imshow(img)
                ax9.set_title(f"Original Image: {metadata['image_name']}")

                # 绘制观测点和重投影点
                pixel_points = [obs[1] for obs in positions]
                obs_points = np.array(pixel_points)
                if len(obs_points) > 0:
                    ax9.scatter(obs_points[:, 0], obs_points[:, 1], s=10, c='blue', marker='o', alpha=0.5,
                                label='Observation Points')

                # 绘制重投影点
                reprojections = []
                for obs in positions:
                    world_point = obs[0]
                    pixel_obs = obs[1]
                    point_w = np.array([world_point['east'], world_point['north'], world_point['up'], 1.0])

                    point_c = R_opt @ point_w[:3] + t_opt
                    z = point_c[2]
                    if z <= 0.01:
                        continue

                    # 投影到图像平面（含畸变）
                    x_norm, y_norm = point_c[0] / z, point_c[1] / z
                    r2 = x_norm ** 2 + y_norm ** 2
                    k1, k2, p1, p2 = dist_coeffs
                    radial = 1.0 + k1 * r2 + k2 * r2 ** 2
                    tang_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
                    tang_y = p1 * (r2 + 2 * y_norm ** 2) + 2.0 * p2 * x_norm * y_norm

                    u_pred = K[0, 0] * (x_norm * radial + tang_x) + K[0, 2]
                    v_pred = K[1, 1] * (y_norm * radial + tang_y) + K[1, 2]

                    reprojections.append([u_pred, v_pred])

                if reprojections:
                    repro_points = np.array(reprojections)
                    ax9.scatter(repro_points[:, 0], repro_points[:, 1], s=20, c='red', marker='x', alpha=0.7,
                                label='Reprojection Points')

                ax9.legend()
                repro_filename = f"{metadata['image_name']}_reprojection.png".replace('/', '_')
                fig3.savefig(repro_filename, dpi=150)
                print(f"Reprojection visualization saved to: {repro_filename}")

                # 显示第三张图但不阻塞
                plt.show(block=False)
        except Exception as e:
            print(f"Image projection visualization failed: {str(e)}")

    # 显示所有图形并等待用户关闭
    plt.show()


def compute_reprojection_errors(result, positions, K, dist_coeffs):
    """
    计算重投影误差的统计值
    :param result: 优化结果字典，包含优化后的参数
    :param positions: 观测位置列表，每个元素为 (world_point_dict, pixel_obs)
    :param K: 相机内参矩阵 (3x3)
    :param dist_coeffs: 畸变系数 [k1, k2, p1, p2]
    :return: 包含平均误差、最大误差、中位误差的元组 (mean_error, max_error, median_error)
            如果没有任何有效点，返回 (None, None, None)
    """
    # 初始化误差列表
    u_errors = []  # X方向误差
    v_errors = []  # Y方向误差
    abs_errors = []  # 总误差（欧氏距离）

    # 获取优化后的旋转和平移参数
    q_opt = result['optimized_params'][:4]

    # 归一化四元数
    q_opt_normalized = (q_opt / np.linalg.norm(q_opt)) if np.linalg.norm(q_opt) > 1e-6 else np.array(
        [1.0, 0.0, 0.0, 0.0])

    # 将四元数转换为旋转矩阵
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = result['optimized_params'][4:7]

    # 遍历所有观测点
    for obs in positions:
        world_point = obs[0]
        pixel_obs = obs[1]

        # 构造世界点向量 [east, north, up]
        point_w = np.array([
            world_point['east'],
            world_point['north'],
            world_point['up']
        ])

        # 世界坐标系 → 相机坐标系
        point_c = R_opt @ point_w + t_opt

        # 深度检查（跳过负深度点）
        z = point_c[2]
        if z <= 0.01:
            continue

        # 归一化图像坐标
        x_norm = point_c[0] / z
        y_norm = point_c[1] / z

        # 计算径向和切向畸变
        r2 = x_norm ** 2 + y_norm ** 2
        k1, k2, p1, p2 = dist_coeffs
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2
        tang_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
        tang_y = p1 * (r2 + 2 * y_norm ** 2) + 2.0 * p2 * x_norm * y_norm

        # 应用畸变校正
        x_dist = x_norm * radial + tang_x
        y_dist = y_norm * radial + tang_y

        # 投影到像素坐标
        u_pred = K[0, 0] * x_dist + K[0, 2]
        v_pred = K[1, 1] * y_dist + K[1, 2]

        # 分别记录XY方向误差
        u_error = u_pred - pixel_obs[0]
        v_error = v_pred - pixel_obs[1]
        abs_error = math.sqrt(u_error ** 2 + v_error ** 2)

        u_errors.append(u_error)
        v_errors.append(v_error)
        abs_errors.append(abs_error)

    # 计算统计值
    if abs_errors:
        # 总误差统计
        mean_error = float(np.mean(abs_errors))
        max_error = float(np.max(abs_errors))
        median_error = float(np.median(abs_errors))

        # XY方向误差统计
        u_mean_abs = float(np.mean(np.abs(u_errors)))  # X方向平均绝对误差
        v_mean_abs = float(np.mean(np.abs(v_errors)))  # Y方向平均绝对误差
        u_max_abs = float(np.max(np.abs(u_errors)))  # X方向最大绝对误差
        v_max_abs = float(np.max(np.abs(v_errors)))  # Y方向最大绝对误差
        return (mean_error, max_error, median_error,
                u_mean_abs, v_mean_abs, u_max_abs, v_max_abs,
                u_errors, v_errors)  # 返回原始误差列表
    else:
        return (None, None, None, None, None, None, None, [], [])


def rotation_to_euler(R):
    """
    将旋转矩阵转换为东北天坐标系下的欧拉角（偏航yaw, 俯仰pitch, 滚转roll）
    坐标系定义：
      - 世界坐标系：东(X), 北(Y), 天(Z)
      - 相机坐标系：右(X), 下(Y), 前(Z)
    参数:
        R: 3x3 旋转矩阵（从世界坐标系到相机坐标系）
    返回:
        yaw, pitch, roll: 以度为单位的角度（0-360°和-90-90°）
    """
    # 计算偏航角 (yaw) - 绕Z轴（天向）的旋转
    # 从北向到东向：atan2(右向在东向分量, 右向在北向分量)
    yaw = np.degrees(np.arctan2(R[0, 0], R[1, 0]))
    yaw = yaw % 360  # 规范化到0-360度范围

    # 计算俯仰角 (pitch) - 绕Y轴（北向）的旋转
    # 前后倾斜：atan2(前向在天向分量, 前向在水平面的模长)
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))

    # 计算滚转角 (roll) - 绕X轴（东向）的旋转
    # 左右倾斜：atan2(下向在天向分量, 前向在天向分量)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    return yaw, pitch, roll

def ransac_outlier_removal(positions, K, dist_coeffs,isinitial=False, initial_rvec=None, initial_tvec=None, threshold=5, max_trials=100000):
    """
    使用 PnP RANSAC 剔除异常匹配点，使用外部提供的初始姿态估计

    参数:
        positions: 匹配点列表，每个元素为 (卫星地理坐标, 航拍图像点)
        K: 相机内参矩阵
        dist_coeffs: 相机畸变系数
        initial_rvec: 初始旋转向量（3x1）
        initial_tvec: 初始平移向量（3x1）
        threshold: RANSAC 重投影误差阈值（像素）
        max_trials: RANSAC 最大迭代次数

    返回:
        过滤后的匹配点列表
    """
    # 提取航拍图特征点像素坐标和卫星图地理坐标
    aerial_points = np.array([position[1] for position in positions])
    satellite_geo_points = np.array([[position[0]['east'], position[0]['north'], position[0]['up']]
                                     for position in positions])

    # 检查是否有足够的点
    if len(aerial_points) < 4:  # PnP 至少需要4个点
        return positions

    if isinitial:
        # 使用 PnP RANSAC 估计相机姿态并剔除异常点，使用外部初始估计
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=satellite_geo_points.astype(np.float32),
            imagePoints=aerial_points.astype(np.float32),
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            useExtrinsicGuess=isinitial,  # 使用外部初始估计
            rvec=initial_rvec,  # 初始旋转向量
            tvec=initial_tvec,  # 初始平移向量
            iterationsCount=max_trials,
            reprojectionError=threshold,
            confidence=0.99
        )
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=satellite_geo_points.astype(np.float32),
            imagePoints=aerial_points.astype(np.float32),
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            useExtrinsicGuess=False,
            iterationsCount=max_trials,
            reprojectionError=threshold,
            confidence=0.99
        )

    if not success or inliers is None:
        return positions

    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵 (3x3)
    camera_position = -R.T @ tvec  # 正确计算相机位置 (3x1)

    # 返回过滤后的匹配点
    return camera_position.flatten()


def filter_matches_grid(positions, image_size=(800, 600), grid_size=(16, 12), num=5):
    # 将输入转换为NumPy数组以便处理
    mkpts0 = np.array([position[1] for position in positions])

    # 初始化全为False的掩码数组
    mask = np.zeros(len(positions), dtype=bool)

    # 计算每个网格的宽度和高度
    grid_width = image_size[0] / grid_size[0]
    grid_height = image_size[1] / grid_size[1]

    # 遍历每个网格
    for i in range(grid_size[0]):  # 列循环
        for j in range(grid_size[1]):  # 行循环
            # 计算当前网格的边界
            x_min = i * grid_width
            x_max = (i + 1) * grid_width
            y_min = j * grid_height
            y_max = (j + 1) * grid_height

            # 找出在当前网格内的点
            in_grid_mask = (
                    (mkpts0[:, 0] >= x_min) &
                    (mkpts0[:, 0] < x_max) &
                    (mkpts0[:, 1] >= y_min) &
                    (mkpts0[:, 1] < y_max)
            )

            # 获取网格内点的索引
            indices_in_grid = np.where(in_grid_mask)[0]

            # 如果网格内点数超过阈值，则随机选择num个点
            if len(indices_in_grid) > num:
                selected_indices = random.sample(list(indices_in_grid), num)
            else:
                selected_indices = indices_in_grid

            # 更新掩码：将选中的点标记为True
            mask[selected_indices] = True

    return [positions[index] for index,item in enumerate(mask) if item ]  # 返回布尔掩码数组


def inject_position_errors(positions, error_ratio):
    """
    在卫星图3D点中注入错误
    参数:
        positions: 原始位置信息列表，每个元素为元组(satellite_geo_dict, aerial_pixel_coords)
        error_ratio: 要注入错误的比例 (0.0-1.0)
    返回:
        包含错误注入的位置信息列表
    """
    # 计算需要注入错误的点数
    total_points = len(positions)
    num_errors = max(1, int(total_points * error_ratio))  # 至少注入1个错误

    # 计算所有3D点的坐标范围，用于生成合理的随机错误点
    satellite_points = np.array([[p[0]['east'], p[0]['north'], p[0]['up']] for p in positions])
    min_vals = np.min(satellite_points, axis=0)
    max_vals = np.max(satellite_points, axis=0)
    ranges = max_vals - min_vals

    # 创建深拷贝以避免修改原始数据
    corrupted_positions = [
        ({'east': p[0]['east'], 'north': p[0]['north'], 'up': p[0]['up']}, p[1])
        for p in positions
    ]

    # 随机选择要注入错误的点
    indices_to_corrupt = random.sample(range(total_points), num_errors)

    # 注入错误：替换卫星图3D点坐标
    for idx in indices_to_corrupt:
        # 在数据范围内生成随机错误坐标
        rand_east = min_vals[0] + random.random() * ranges[0]
        rand_north = min_vals[1] + random.random() * ranges[1]
        rand_up = min_vals[2] + random.random() * ranges[2]

        # 替换原始坐标
        corrupted_positions[idx] = (
            {'east': rand_east, 'north': rand_north, 'up': rand_up},
            corrupted_positions[idx][1]  # 保持航拍图像素坐标不变
        )

    return corrupted_positions

def location_opt(data_all, query_system, error_threshold_stage1=5.0, error_threshold_stage2=2.0,
                 error_threshold_stage3=0.8, visualize=True):
    mkpts0, mkpts1, _, _, metadata = data_all
    print(f"初始匹配点数量: {len(mkpts0)}对")

    # 相机内参
    K = np.array([
        [750.62614972, 0, 402.41007535],
        [0, 750.26301185, 292.98832147],
        [0, 0, 1]
    ])

    # 畸变系数
    dist_coeffs = np.array([
        -0.11592226392258145,
        0.1332261251415265,
        -0.00043977637330175616,
        0.0002380609784102606
    ])

    # 地理变换参数
    im_geotrans = (6.769466400146, 2.6822090148935768e-06, 0.0, 50.670353914958, 0.0, -2.6822716474094548e-06)
    rate1 = metadata['rate1']
    rate2 = metadata['rate2']

    # 获取地图中心点坐标
    lon0, lat0 = pixel_to_lonlat(metadata['pixel_x'], metadata['pixel_y'], im_geotrans, [10937600, 2752200])
    elevation0 = query_system.get_elevation(lon0, lat0)
    print(f"地图中心点: 经度={lon0:.6f}, 纬度={lat0:.6f}, 高程={elevation0:.1f}m")
    lon1, lat1 = pixel_to_lonlat(metadata['pixel_x']+metadata['width'], metadata['height']+metadata['width'], im_geotrans, [10937600, 2752200])
    print(f"地图中心点: 经度={lon1:.6f}, 纬度={lat1:.6f}, 高程={elevation0:.1f}m")
    # 计算相机初始高度（比地图中心高90米）
    initial_height = elevation0 + 90.0 if elevation0 is not None else 100.0

    # 准备位置数据
    positions = []
    for i in range(len(mkpts0)):
        # 计算匹配点在地图上的像素坐标
        pixel_x = metadata['pixel_x'] + mkpts1[i][0] * rate1
        pixel_y = metadata['pixel_y'] + mkpts1[i][1] * rate2

        try:
            # 精确地理坐标
            lon, lat = pixel_to_lonlat(pixel_x, pixel_y, im_geotrans, [10937600, 2752200])
            # 获取高程
            elevation = query_system.get_elevation(lon, lat)
            if elevation is None:
                elevation = 0.0

            # 计算相对位置
            location = calculate_relative_position(lat0, lon0, elevation0, lat, lon, elevation)
            positions.append([location, mkpts0[i]])
        except Exception as e:
            print(f"点{i}位置计算失败: {str(e)}")

    F, mask = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=50,
        confidence=0.99
    )
    if mask is not None:
        mask = mask.ravel().astype(bool)
        positions = [positions[index] for index,item in enumerate(mask) if item ]

    positions = sorted(positions, key=lambda x: x[0]['up'])

    # # 错误注入
    # positions = inject_position_errors(positions, 0.5)

    positions_A = filter_matches_grid(positions, num=int(len(positions) / (16 * 12)))
    positions_B = filter_matches_grid(positions, num=int(len(positions) / (16 * 12)))
    positions_C = filter_matches_grid(positions, num=int(len(positions) / (16 * 12)))

    positions_no_hight = [copy.deepcopy(item) for item in positions]
    for index, item in enumerate(positions_no_hight):
        positions_no_hight[index][0]['up'] = 0


    if len(positions_A) <200 or len(positions_B) <200:
        return None

    pre_position = [0,0,0]

#############################################优化A
    hight_positions_A = sum([item[0]['up'] for item in positions_A]) / len(positions_A)
    hight_positions_min_A = sum([item[0]['up'] for item in positions_A[:int(len(positions_A) * 0.05)]]) / (len(positions_A) * 0.05)
    hight_positions_max_A = sum([item[0]['up'] for item in positions_A[int(len(positions_A) * 0.9):]]) / (len(positions_A) * 0.1)
    print(f'平均高度：{hight_positions_A}')
    print(f'最小高度：{hight_positions_min_A}')
    print(f'最大高度：{hight_positions_max_A}')


    # for index, item in enumerate(positions):
    #     positions[index][0]['east'] +=min(12,(hight_positions - hight_positions_min)) * 0.4
    positions_no_hight_A = [copy.deepcopy(item) for item in positions_A]
    for index, item in enumerate(positions_no_hight_A):
        positions_no_hight_A[index][0]['up']=0

    #pnp解算
    camera_position_pnp = ransac_outlier_removal(positions,K,dist_coeffs).tolist()
    # camera_position_pnp = [0,0,0]

    # 计算统计量
    up_values = [loc[0]['up'] for loc in positions_A]
    max_up = np.max(up_values)
    min_up = np.min(up_values)
    median_up = np.median(up_values)
    mean_up = np.mean(up_values)
    variance_up = np.var(up_values)

    # 2. 初始参数设置 (改进版)
    # - 旋转: 正视前方
    # - 平移: 地图中心点上方 (初始高度)
    rvec_initial = np.array([0, -1.5, 0])
    q0 = rotation_vector_to_quaternion(rvec_initial)
    t0 = np.array([0, 0, initial_height])
    x0 = np.concatenate([q0, t0])
    print(f"优化初始参数: 四元数={q0}, 平移={t0} (高度={initial_height:.1f}m)")

    # 检查第一阶段结果是否在第二阶段边界内
    bounds_lb = np.array([-2, -2, -2, -2, -1000, -1000, 50])
    bounds_ub = np.array([2, 2, 2, 2, 1000, 1000, 1000])

    print("\n=== 第一阶段优化 ===")
    result_stage1 = least_squares(
        fun=quaternion_residuals,
        x0=x0,
        args=(positions_no_hight, K, dist_coeffs, 0.0),  # 加强旋转约束
        method='trf',
        loss='cauchy',
        bounds=(bounds_lb, bounds_ub),
        verbose=1,
        ftol=1e-8,
        max_nfev=1000
    )
    optimized_params_stage1 = result_stage1.x
    # 提取最终优化结果
    q_opt = optimized_params_stage1[:4]
    if np.linalg.norm(q_opt) > 1e-6:
        q_opt_normalized = q_opt / np.linalg.norm(q_opt)
    else:
        q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = optimized_params_stage1[4:7]
    # 计算相机位置
    camera_position = -R_opt.T @ t_opt
    pre_position1 = camera_position.tolist()

    # 6. 第二阶段异常点再剔除（重投影误差 > 0.8）
    print("\n=== 第二阶段优化 ===")
    result_stage2 = least_squares(
        fun=quaternion_residuals,
        x0=x0,
        args=(positions, K, dist_coeffs, 0.0),  # 超强旋转约束
        method='trf',
        loss='cauchy',
        bounds=(bounds_lb, bounds_ub),
        verbose=1,
        ftol=1e-8,
        max_nfev=1000
    )
    optimized_params_stage2 = result_stage2.x
    # 提取最终优化结果
    q_opt = optimized_params_stage2[:4]
    if np.linalg.norm(q_opt) > 1e-6:
        q_opt_normalized = q_opt / np.linalg.norm(q_opt)
    else:
        q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = optimized_params_stage2[4:7]
    # 计算相机位置
    camera_position = -R_opt.T @ t_opt
    pre_position2 = camera_position.tolist()

    optimized_params_stage_A = (optimized_params_stage1 + optimized_params_stage2) / 2

    pre_position_A=[pre_position1[0], pre_position1[1], pre_position2[2]]
    yaw, pitch, roll = rotation_to_euler(R_opt)
    pre_position_A[2] += (hight_positions_A-hight_positions_min_A)

    #############################################优化B
    hight_positions_B = sum([item[0]['up'] for item in positions_B]) / len(positions_B)
    hight_positions_min_B = sum([item[0]['up'] for item in positions_B[:int(len(positions_B) * 0.05)]]) / (
                len(positions_A) * 0.05)
    hight_positions_max_B = sum([item[0]['up'] for item in positions_B[int(len(positions_B) * 0.9):]]) / (
                len(positions_B) * 0.1)
    print(f'平均高度：{hight_positions_B}')
    print(f'最小高度：{hight_positions_min_B}')
    print(f'最大高度：{hight_positions_max_B}')

    # for index, item in enumerate(positions):
    #     positions[index][0]['east'] +=min(12,(hight_positions - hight_positions_min)) * 0.4
    positions_no_hight_B = [copy.deepcopy(item) for item in positions_B]
    for index, item in enumerate(positions_no_hight_B):
        positions_no_hight_B[index][0]['up'] = 0


    # 计算统计量
    up_values = [loc[0]['up'] for loc in positions_B]
    max_up = np.max(up_values)
    min_up = np.min(up_values)
    median_up = np.median(up_values)
    mean_up = np.mean(up_values)
    variance_up = np.var(up_values)

    # 2. 初始参数设置 (改进版)
    # - 旋转: 正视前方
    # - 平移: 地图中心点上方 (初始高度)
    rvec_initial = np.array([0, -1.5, 0])
    q0 = rotation_vector_to_quaternion(rvec_initial)
    t0 = np.array([0, 0, initial_height])
    x0 = np.concatenate([q0, t0])
    print(f"优化初始参数: 四元数={q0}, 平移={t0} (高度={initial_height:.1f}m)")

    # 检查第一阶段结果是否在第二阶段边界内
    bounds_lb = np.array([-2, -2, -2, -2, -1000, -1000, 50])
    bounds_ub = np.array([2, 2, 2, 2, 1000, 1000, 1000])

    print("\n=== 第一阶段优化 ===")
    result_stage1 = least_squares(
        fun=quaternion_residuals,
        x0=optimized_params_stage_A,
        args=(positions_no_hight_B, K, dist_coeffs, 0.0),  # 加强旋转约束
        method='trf',
        loss='cauchy',
        bounds=(bounds_lb, bounds_ub),
        verbose=1,
        ftol=1e-8,
        max_nfev=1000
    )
    optimized_params_stage1 = result_stage1.x
    # 提取最终优化结果
    q_opt = optimized_params_stage1[:4]
    if np.linalg.norm(q_opt) > 1e-6:
        q_opt_normalized = q_opt / np.linalg.norm(q_opt)
    else:
        q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = optimized_params_stage1[4:7]
    # 计算相机位置
    camera_position = -R_opt.T @ t_opt
    pre_position1 = camera_position.tolist()

    # 6. 第二阶段异常点再剔除（重投影误差 > 0.8）
    print("\n=== 第二阶段优化 ===")
    result_stage2 = least_squares(
        fun=quaternion_residuals,
        x0=optimized_params_stage_A,
        args=(positions_B, K, dist_coeffs, 0.0),  # 超强旋转约束
        method='trf',
        loss='cauchy',
        bounds=(bounds_lb, bounds_ub),
        verbose=1,
        ftol=1e-8,
        max_nfev=1000
    )
    optimized_params_stage2 = result_stage2.x
    # 提取最终优化结果
    q_opt = optimized_params_stage2[:4]
    if np.linalg.norm(q_opt) > 1e-6:
        q_opt_normalized = q_opt / np.linalg.norm(q_opt)
    else:
        q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
    R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
    t_opt = optimized_params_stage2[4:7]
    # 计算相机位置
    camera_position = -R_opt.T @ t_opt
    pre_position2 = camera_position.tolist()

    pre_position_B = [pre_position1[0], pre_position1[1], pre_position2[2]]
    yaw, pitch, roll = rotation_to_euler(R_opt)
    pre_position_B[2] += (hight_positions_A - hight_positions_min_A)

    #############################################仲裁段

    print(f'\n')
    print(f'======仲裁后两次优化误差：',np.linalg.norm(np.array(pre_position_A)-np.array(pre_position_B)))

    if np.linalg.norm(np.array(pre_position_A)-np.array(pre_position_B)) > 2:

        hight_positions_C = sum([item[0]['up'] for item in positions_C]) / len(positions_C)
        hight_positions_min_C = sum([item[0]['up'] for item in positions_C[:int(len(positions_C) * 0.05)]]) / (
                    len(positions_C) * 0.05)
        hight_positions_max_C = sum([item[0]['up'] for item in positions_C[int(len(positions_C) * 0.9):]]) / (
                    len(positions_C) * 0.1)
        print(f'平均高度：{hight_positions_C}')
        print(f'最小高度：{hight_positions_min_C}')
        print(f'最大高度：{hight_positions_max_C}')

        # for index, item in enumerate(positions):
        #     positions[index][0]['east'] +=min(12,(hight_positions - hight_positions_min)) * 0.4
        positions_no_hight_C = [copy.deepcopy(item) for item in positions_C]
        for index, item in enumerate(positions_no_hight_C):
            positions_no_hight_C[index][0]['up'] = 0

        # 计算统计量
        up_values = [loc[0]['up'] for loc in positions_C]
        max_up = np.max(up_values)
        min_up = np.min(up_values)
        median_up = np.median(up_values)
        mean_up = np.mean(up_values)
        variance_up = np.var(up_values)

        # 2. 初始参数设置 (改进版)
        # - 旋转: 正视前方
        # - 平移: 地图中心点上方 (初始高度)
        rvec_initial = np.array([0, -1.5, 0])
        q0 = rotation_vector_to_quaternion(rvec_initial)
        t0 = np.array([0, 0, initial_height])
        x0 = np.concatenate([q0, t0])
        print(f"优化初始参数: 四元数={q0}, 平移={t0} (高度={initial_height:.1f}m)")

        # 检查第一阶段结果是否在第二阶段边界内
        bounds_lb = np.array([-2, -2, -2, -2, -1000, -1000, 50])
        bounds_ub = np.array([2, 2, 2, 2, 1000, 1000, 1000])

        print("\n=== 第一阶段优化 ===")
        result_stage1 = least_squares(
            fun=quaternion_residuals,
            x0=x0,
            args=(positions_no_hight_C, K, dist_coeffs, 0.0),  # 加强旋转约束
            method='trf',
            loss='cauchy',
            bounds=(bounds_lb, bounds_ub),
            verbose=1,
            ftol=1e-8,
            max_nfev=1000
        )
        optimized_params_stage1 = result_stage1.x
        # 提取最终优化结果
        q_opt = optimized_params_stage1[:4]
        if np.linalg.norm(q_opt) > 1e-6:
            q_opt_normalized = q_opt / np.linalg.norm(q_opt)
        else:
            q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
        R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
        t_opt = optimized_params_stage1[4:7]
        # 计算相机位置
        camera_position = -R_opt.T @ t_opt
        pre_position1 = camera_position.tolist()

        # 6. 第二阶段异常点再剔除（重投影误差 > 0.8）
        print("\n=== 第二阶段优化 ===")
        result_stage2 = least_squares(
            fun=quaternion_residuals,
            x0=x0,
            args=(positions_C, K, dist_coeffs, 0.0),  # 超强旋转约束
            method='trf',
            loss='cauchy',
            bounds=(bounds_lb, bounds_ub),
            verbose=1,
            ftol=1e-8,
            max_nfev=1000
        )
        optimized_params_stage2 = result_stage2.x
        # 提取最终优化结果
        q_opt = optimized_params_stage2[:4]
        if np.linalg.norm(q_opt) > 1e-6:
            q_opt_normalized = q_opt / np.linalg.norm(q_opt)
        else:
            q_opt_normalized = np.array([1.0, 0.0, 0.0, 0.0])
        R_opt = quaternion_to_rotation_matrix(q_opt_normalized)
        t_opt = optimized_params_stage2[4:7]
        # 计算相机位置
        camera_position = -R_opt.T @ t_opt
        pre_position2 = camera_position.tolist()

        pre_position_C = [pre_position1[0], pre_position1[1], pre_position2[2]]
        yaw, pitch, roll = rotation_to_euler(R_opt)
        pre_position_C[2] += (hight_positions_C - hight_positions_min_C)

        if (np.linalg.norm(np.array(pre_position_A)-np.array(pre_position_C))<np.linalg.norm(np.array(pre_position_B)-np.array(pre_position_C))
                and np.linalg.norm(np.array(pre_position_A)-np.array(pre_position_C))<1.5):
            pre_position = (np.array(pre_position_A)+np.array(pre_position_C))/2
            pre_position = pre_position.tolist()

            print(f'======仲裁后第三次优化误差:', np.linalg.norm(np.array(pre_position_A) - np.array(pre_position_C)))
            print(f'\n')
        elif (np.linalg.norm(np.array(pre_position_A)-np.array(pre_position_C))>=np.linalg.norm(np.array(pre_position_B)-np.array(pre_position_C))
              and np.linalg.norm(np.array(pre_position_B)-np.array(pre_position_C))<1.5):
            pre_position = (np.array(pre_position_B) + np.array(pre_position_C)) / 2
            pre_position = pre_position.tolist()

            print(f'仲裁后第三次优化误差：', np.linalg.norm(np.array(pre_position_B) - np.array(pre_position_C)))
            print(f'\n')

    else:
        pre_position = (np.array(pre_position_A) + np.array(pre_position_B)) / 2
        pre_position = pre_position.tolist()
##########################################################################################仲裁结束

    # 获取真实位置
    filepath_csv = '/home/superz/Desktop/data/vpair/poses.csv'
    if os.path.exists(filepath_csv):
        df_true_csv = pd.read_csv(filepath_csv)
        target_row = df_true_csv[df_true_csv['filename'] == metadata['image_name']]

        if not target_row.empty:
            true_lon = target_row['lon'].item()
            true_lat = target_row['lat'].item()
            true_alt = target_row['altitude'].item()

            # 查询真实位置的高程
            true_elevation = query_system.get_elevation(true_lon, true_lat)
            print(
                f"真实目标点: 经度={true_lon:.6f}, 纬度={true_lat:.6f}, 标称高度={true_alt:.1f}m, 实际高程={true_elevation if true_elevation is not None else '未知'}")

            ture_position = calculate_relative_position(
                lat0, lon0, elevation0,
                true_lat, true_lon, true_alt
            )
            true_pos = [ture_position['east'], ture_position['north'], ture_position['up']]
        else:
            print("警告: 未找到对应的真实位置数据")
            true_pos = [0, 0, 0]
    else:
        print("警告: 真实位置CSV文件不存在")
        true_pos = [0, 0, 0]

    # 计算精度
    position_error = np.array(pre_position) - np.array(true_pos)
    distance_error = np.linalg.norm(position_error)

    if np.linalg.norm(pre_position)==0:
        distance_error = 0

    # 使用最终优化参数计算重投影误差
    (mean, max_err, median,
     u_mean, v_mean, u_max, v_max,
     u_errors, v_errors) = compute_reprojection_errors(
        {'optimized_params': optimized_params_stage2},  # 创建临时结果字典
        positions, K, dist_coeffs
    )

    print(optimized_params_stage2)
    # 组织结果
    result_data = {
        'predicted_position': pre_position,
        'predicted_position1': pre_position1,
        'predicted_position2': pre_position2,
        'pre_position_A':pre_position_A,
        'camera_position_pnp':camera_position_pnp,
        'true_position': true_pos,
        'position_error': position_error.tolist(),
        'distance_error': distance_error,
        'yaw':yaw,
        'pitch':pitch,
        'roll':roll,
        'n_original_points': len(positions),
        'n_cleaned_points': len(positions),
        'n_final_points': len(positions),  # 添加最终点数
        'optimized_params': optimized_params_stage2,
        'metadata': metadata,
        'max_up': float(max_up),
        'min_up': float(min_up),
        'median_up': float(median_up),
        'mean_up': float(mean_up),
        'variance_up':float(variance_up),
        'reprojection_errors': {
            'total': {
                'mean': round(mean, 2) if mean is not None else None,
                'max': round(max_err, 2) if max_err is not None else None,
                'median': round(median, 2) if median is not None else None,
            },
            'x_direction': {
                'mean_abs': round(u_mean, 2) if u_mean is not None else None,
                'max_abs': round(u_max, 2) if u_max is not None else None,
                'errors': u_errors  # 原始误差列表
            },
            'y_direction': {
                'mean_abs': round(v_mean, 2) if v_mean is not None else None,
                'max_abs': round(v_max, 2) if v_max is not None else None,
                'errors': v_errors  # 原始误差列表
            }
        }
    }

    # 打印详细误差统计
    print("\n===== Reprojection Errors =====")
    print(f"Total: mean={mean:.2f}px, max={max_err:.2f}px, median={median:.2f}px")
    print(f"X-direction: mean_abs={u_mean:.2f}px, max_abs={u_max:.2f}px")
    print(f"Y-direction: mean_abs={v_mean:.2f}px, max_abs={v_max:.2f}px")

    # 可视化结果
    if visualize:
        image_path = f"./data/images/{metadata['image_name']}" if 'image_name' in metadata else None
        visualize_results(result_data, optimized_params_stage2, K, dist_coeffs, metadata, image_path)

    return result_data


# 主程序入口
if __name__ == "__main__":
    # 加载匹配点数据
    with open('./data/data_all_10.pkl', 'rb') as f:
        data_all = pickle.load(f)

    # 加载高程数据
    npy_path = '/home/superz/Desktop/data/vpair/hight_elevation.npy'
    geo_txt_path = '/home/superz/Desktop/data/vpair/hight_elevation_geo.txt'
    query_system = ElevationQuery(npy_path, geo_txt_path)

    # 运行优化（严格误差阈值设置）
    result = location_opt(data_all, query_system,
                          error_threshold_stage1=1,
                          error_threshold_stage2=1,
                          error_threshold_stage3=0.5)

    # 打印结果
    if result is not None:
        print("\n===== 最终优化结果 =====")
        print(f"预测位置 (东, 北, 高): {result['predicted_position']}")
        print(f"真实位置 (东, 北, 高): {result['true_position']}")
        print(f"位置误差: {result['position_error']}")
        print(f"欧氏距离误差: {result['distance_error']:.2f}米")

        print("\n===== 匹配点分析 =====")
        print(f"原始匹配点数: {result['n_original_points']}")
        print(
            f"有效匹配点数: {result['n_cleaned_points']} ({result['n_cleaned_points'] / result['n_original_points'] * 100:.1f}%)")

        print("\n===== 分析报告 =====")
        if result['distance_error'] < 10:
            print("✅ 优化成功! 位置误差小于10米")
        elif result['distance_error'] < 30:
            print("⚠️ 中等精度，有改进空间")
            print("- 检查高程数据的准确性")
            print("- 尝试调整初始高度估计")
            print("- 增加更多匹配特征点")
        else:
            print("❌ 优化需要重大改进")
            print("- 确认相机标定参数准确性")
            print("- 验证地理变换参数(geotrans)")
            print("- 考虑使用IMU等传感器辅助")
            print("- 检查原始匹配点质量")

        print("\n👉 详细可视化结果已生成，请查看相关图片文件")
    else:
        print("优化过程失败，无有效结果")