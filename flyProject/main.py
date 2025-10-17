# main.py
import sys
from pathlib import Path
import os
from typing import List, Dict, Tuple, Any, Optional, Union
import json
import pickle
import time
from xml.etree.ElementTree import tostring
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from sympy import false

# 添加项目根目录和 MixVPR 目录到路径
project_root = str(Path(__file__).parent)
sys.path.extend([project_root, os.path.join(project_root, 'MixVPR')])
sys.path.extend([project_root, os.path.join(project_root, 'gim-main')])

# 项目模块导入
from featurePoints import load_model_featurePoints, find_FeaturePoint, image_show,image_show1
from MixVPR.main import VPRModel
from globleFind import load_model, preprocess_image, extract_features, cosine_similarity
from geopy.distance import great_circle
import ast
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, least_squares
from location_optism import location_opt
from hight_load import ElevationQuery
import rasterio
from rasterio.windows import Window
from DataLogger import DataLogger

# ==================== 配置参数 ====================
class Config:
    """项目配置参数"""
    # 路径配置
    MAP_DIR = './mapLocation'
    QUERY_DIR = '/home/superz/Desktop/data/vpair/queries'
    MODEL_MAP_FILE = "/home/superz/Desktop/flyProject/flyProject/MixVPR/last.ckpt"
    MODEL_FEATURE_FILE = "gim-main/weights/gim_loftr_50h.ckpt"
    TIF_PATH = "/home/superz/Desktop/data/vpair/vpair.tif"
    HIGHT_NPY_PATH = '/home/superz/Desktop/data/vpair/hight_elevation.npy'
    HIGHT_GEO_TXT_PATH = '/home/superz/Desktop/data/vpair/hight_elevation_geo.txt'

    width_index = 5

    width_index_add = 3
    # 处理范围配置(850-1860)
    QUERY_START_INDEX = 970
    QUERY_END_INDEX = 1070

    # 全局查询的范围
    # query_which = [42, 407]#850
    # query_which = [48, 406]  # 860
    # query_which = [65, 406]  # 890
    # query_which = [71, 405]  # 900
    # query_which = [89, 405]#930
    query_which = [114, 415] #970
    # query_which = [19, 62]  # 1520
    # query_which = [23, 5]  # 1590

    # 图像处理参数
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    SCALE_FACTORS = [1.0, 0.6, 0.36]  # 0.6^0, 0.6^1, 0.6^2

    # 优化参数
    INITIAL_PARAMS = np.array([0, -1.57, 0, 0, 0, 306.83439])
    ERROR_THRESHOLD_STAGE1 = 1000
    ERROR_THRESHOLD_STAGE2 = 1000
    ERROR_THRESHOLD_STAGE3 = 1000


# ==================== 图像处理函数 ====================
def crop_tif_rasterio(tif_path: str, x_start: int, y_start: int, width: int, height: int) -> Image.Image:
    """
    使用 rasterio 裁剪 TIFF 文件，超出部分填充黑色

    Args:
        tif_path: TIFF文件路径
        x_start: 裁剪起始x坐标
        y_start: 裁剪起始y坐标
        width: 裁剪宽度
        height: 裁剪高度

    Returns:
        PIL Image对象和裁剪区域信息
    """
    with rasterio.open(tif_path) as src:
        # 计算实际有效的裁剪区域
        valid_x_start = max(x_start, 0)
        valid_y_start = max(y_start, 0)
        valid_x_end = min(x_start + width, src.width)
        valid_y_end = min(y_start + height, src.height)

        # 计算有效区域的尺寸
        valid_width = max(0, valid_x_end - valid_x_start)
        valid_height = max(0, valid_y_end - valid_y_start)

        # 创建全黑图像（目标尺寸）
        if src.count < 3:
            raise ValueError("TIFF需要至少3个波段(RGB)")

        # 使用源文件的数据类型创建黑色图像
        dtype = src.dtypes[0]
        black_img = np.zeros((height, width, 3), dtype=dtype)

        # 如果有有效区域，读取并填充到黑色图像中
        if valid_width > 0 and valid_height > 0:
            window = Window(valid_x_start, valid_y_start, valid_width, valid_height)
            data = src.read(window=window)

            # 确保至少有3个波段
            if data.shape[0] < 3:
                raise ValueError("TIFF需要至少3个波段(RGB)")

            # 转换为 (height, width, bands)
            rgb_data = np.transpose(data[:3], (1, 2, 0))

            # 计算在目标图像中的位置
            target_x = valid_x_start - x_start
            target_y = valid_y_start - y_start

            # 将有效数据填充到黑色图像中
            black_img[target_y:target_y + valid_height,
            target_x:target_x + valid_width] = rgb_data

        # 创建PIL图像
        img = Image.fromarray(black_img, 'RGB')
        return img, [x_start, y_start, width, height]


def denoise_and_blur(image: Union[Image.Image, np.ndarray], level: int = 2) -> Image.Image:
    """
    对图像进行去噪和适当高斯模糊处理

    Args:
        image: 输入图像（PIL.Image或numpy数组）
        level: 去噪强度

    Returns:
        处理后的PIL.Image对象
    """
    # 转换为OpenCV格式（BGR）
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    # 1. 去噪处理（使用非局部均值去噪）
    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=level,  # 亮度分量滤波强度
        hColor=5,  # 颜色分量滤波强度
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 转换回PIL格式
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


def zoned_gamma_correction(img: np.ndarray, gamma: float = 1.0, clip_threshold: float = 0.95) -> np.ndarray:
    """
    分区伽马校正

    Args:
        img: 输入图像 (BGR 格式)
        gamma: 伽马值
        clip_threshold: 过曝阈值

    Returns:
        校正后的图像
    """
    # 转换为 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 计算亮度分布
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cum_hist = np.cumsum(hist) / np.sum(hist)

    # 找到过曝阈值
    overexposed_threshold = np.where(cum_hist > clip_threshold)[0]
    overexposed_threshold = overexposed_threshold[0] if len(overexposed_threshold) > 0 else 220

    # 创建亮度分区
    low_mask = v < 100  # 暗区
    mid_mask = (v >= 100) & (v < overexposed_threshold)  # 中间区
    high_mask = v >= overexposed_threshold  # 高亮区

    # 分区应用伽马校正
    v_corrected = np.zeros_like(v, dtype=np.float32)

    # 暗区：提升亮度（gamma < 1）
    v_corrected[low_mask] = np.power(v[low_mask] / 255.0, 0.8) * 255

    # 中间区：保持原样
    v_corrected[mid_mask] = v[mid_mask]

    # 高亮区：降低亮度（gamma > 1）
    v_corrected[high_mask] = np.power(v[high_mask] / 255.0, 1.5) * 255

    v_corrected = np.clip(v_corrected, 0, 255).astype(np.uint8)

    # 合并通道
    hsv_corrected = cv2.merge([h, s, v_corrected])
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)


def reduce_contrast_linear(img: np.ndarray, factor: float = 0.8) -> np.ndarray:
    """
    线性降低图片对比度

    Args:
        img: 输入图像 (BGR 格式)
        factor: 对比度降低因子 (0-1, 0为最低对比度)

    Returns:
        对比度降低后的图像
    """
    # 将图像转换为浮点类型
    img_float = img.astype(np.float32) / 255.0

    # 计算图像平均值
    mean_val = np.mean(img_float)

    # 应用对比度降低公式
    # output = mean + factor * (input - mean)
    reduced = mean_val + factor * (img_float - mean_val)

    # 转换回 uint8 类型
    return (np.clip(reduced, 0, 1) * 255).astype(np.uint8)


def filter_matches(mkpts0, mkpts1, max_iter=10000, reproj_threshold=2):
    """
    使用畸变矫正和RANSAC剔除误匹配特征点，返回原始坐标的内点

    参数：
    mkpts0 - 相机图像的特征点坐标列表（原始坐标），格式: [[x0,y0], [x1,y1], ...]
    mkpts1 - 卫星图像的特征点坐标列表（原始坐标），格式: [[x0,y0], [x1,y1], ...]
    K - 相机内参矩阵，3x3的numpy数组
    dist_coeffs - 相机畸变系数，1维numpy数组
    max_iter - RANSAC最大迭代次数，默认10000
    reproj_threshold - 重投影误差阈值，默认2.0像素

    返回：
    mkpts0_new - 筛选后的相机图像特征点列表（原始坐标）
    mkpts1_new - 筛选后的卫星图像特征点列表（原始坐标）
    """
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

    # 检查输入数据
    if len(mkpts0) < 4 or len(mkpts1) < 4 or len(mkpts0) != len(mkpts1):
        return [], []  # 输入无效返回空列表

    # 转换为numpy数组
    points0_cam = np.array(mkpts0, dtype=np.float32)
    points1_sat = np.array(mkpts1, dtype=np.float32)

    # ==== 畸变矫正 (仅对相机图像点) ====
    points0_cam_reshaped = points0_cam.reshape(-1, 1, 2)
    points0_undistorted = cv2.undistortPoints(
        points0_cam_reshaped,
        K,
        dist_coeffs,
        None,
        K
    ).reshape(-1, 2)

    # ==== RANSAC匹配 ====
    if len(points0_undistorted) < 4:  # 更新为最少4个点
        return mkpts0, mkpts1  # 点数不足时返回所有原始点

    # 计算单应性矩阵并获取内点掩码
    _, mask = cv2.findHomography(
        points1_sat,  # 源点（卫星图像点）
        points0_undistorted,  # 目标点（相机图像矫正后的点）
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=reproj_threshold,
        confidence=0.999,
        maxIters=max_iter
    )

    # 如果没有获取掩码，返回所有原始点
    if mask is None:
        return mkpts0, mkpts1

    # 获取内点掩码并筛选原始坐标点
    inlier_mask = mask.ravel().astype(bool)
    mkpts0_new = points0_cam[inlier_mask].tolist()  # 使用原始相机点
    mkpts1_new = points1_sat[inlier_mask].tolist()  # 使用原始卫星点

    return mkpts0_new, mkpts1_new

def get_img0_corners_in_img1(img0_rgb, img1_rgb, mkpts0, mkpts1):
    """
    计算img0的四个角点在img1坐标系中的对应位置

    参数:
    img0_rgb - 第一张图像的RGB数组
    img1_rgb - 第二张图像的RGB数组
    mkpts0 - 第一张图像中的匹配点
    mkpts1 - 第二张图像中的匹配点

    返回:
    corners_in_img1 - img0的四个角点在img1中的坐标，顺序为[左上, 右上, 右下, 左下]
                     如果计算失败则返回None
    """
    mkpts0, mkpts1 = filter_matches(mkpts0, mkpts1, reproj_threshold=3)

    # 1. 获取图像尺寸
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # 2. 检查匹配点数量
    if len(mkpts0) < 15 or len(mkpts1) < 15:
        print("匹配点不足，无法计算单应性矩阵")
        return None

    # 3. 计算从img0到img1的单应性矩阵
    # 注意：这里需要计算从img0到img1的变换，所以输入点顺序为(mkpts0, mkpts1)
    pts0 = np.array(mkpts0, dtype=np.float32)
    pts1 = np.array(mkpts1, dtype=np.float32)

    # 计算从img0到img1的单应性矩阵
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC)

    if H is None:
        print("无法计算单应性矩阵")
        return None

    # 4. 定义img0的四个角点
    corners_img0 = np.array([
        [0, 0],  # 左上角
        [w0 - 1, 0],  # 右上角
        [w0 - 1, h0 - 1],  # 右下角
        [0, h0 - 1]  # 左下角
    ], dtype=np.float32).reshape(-1, 1, 2)

    # 5. 计算角点在img1中的位置
    corners_in_img1 = cv2.perspectiveTransform(corners_img0, H)

    # 6. 将结果转换为标准格式 (4个点的坐标)
    return corners_in_img1.reshape(-1, 2)


# ==================== 特征处理函数 ====================
def load_map_features(map_dir: str) -> Dict[str, Any]:
    """
    加载分块存储的地图特征和元数据

    Args:
        map_dir: 特征存储目录

    Returns:
        包含索引、元数据和特征的字典
    """
    # 1. 加载索引文件
    with open(os.path.join(map_dir, "index.json"), "r") as f:
        index_info = json.load(f)

    # 2. 加载元数据
    metadata = []
    with open(index_info["metadata_file"], "r") as f:
        for line in f:
            metadata.append(json.loads(line))

    # 3. 按需加载特征块（内存高效方式）
    features = []
    for chunk_file in index_info["chunk_files"]:
        # 完整路径处理
        if not os.path.isabs(chunk_file):
            chunk_file = os.path.join(map_dir, os.path.basename(chunk_file))
        features.append(np.load(chunk_file, mmap_mode='r'))

    return {
        "index": index_info,
        "metadata": metadata,
        "features": features
    }


def metadata_process(metadata_obj):
    if isinstance(metadata_obj['metadata'], str):
        try:
            start = metadata_obj['metadata'].find('{')
            if start != -1:
                metadata_obj['metadata'] = ast.literal_eval(metadata_obj['metadata'][start:])
            else:
                metadata_obj['metadata'] = {}
        except:
            metadata_obj['metadata'] = {}
    return metadata_obj

def get_max_idex(feature_data):
    max_idex_x = 0
    max_idex_y = 0
    for item in feature_data["metadata"]:
        metadata_obj = metadata_process(item)
        idex_x = metadata_obj['metadata']['row_idx']
        idex_y = metadata_obj['metadata']['col_idx']
        if idex_x > max_idex_x:
            max_idex_x = idex_x
        if idex_y > max_idex_y:
            max_idex_y = idex_y
    return max_idex_x, max_idex_y

def query_image(feature_data, query_vector, x_position, y_position, max_x_index, max_y_index, width=20,
                          top_k=5):
    """
    优化版查询函数 - 核心功能版本
    """
    # ===== 阶段1: 筛选区域内的元数据索引 =====
    # 预处理所有元数据
    preprocessed_meta = [metadata_process(item) for item in feature_data["metadata"]]

    # 计算边界
    min_x = max(0, x_position - width // 2)
    max_x = min(max_x_index, x_position + width // 2)
    min_y = max(0, y_position - width // 2)
    max_y = min(max_y_index, y_position + width // 2)

    # 使用集合提高查找效率
    feature_index_set = set()
    for meta in preprocessed_meta:
        row, col = meta['metadata']['row_idx'], meta['metadata']['col_idx']
        if min_x <= row <= max_x and min_y <= col <= max_y:
            feature_index_set.add(meta['idx'])

    # ===== 阶段2: 向量化相似度计算 =====
    # 收集所有需要计算的特征向量
    selected_features = []
    selected_indices = []

    start_idx = 0
    for chunk in feature_data["features"]:
        for i, feature in enumerate(chunk):
            if start_idx + i in feature_index_set:
                selected_features.append(feature)
                selected_indices.append(start_idx + i)
        start_idx += len(chunk)

    # 转换为NumPy数组进行向量化计算
    if selected_features:
        selected_features = np.array(selected_features)
        query_vector_np = np.array(query_vector).reshape(1, -1)

        # 向量化余弦相似度计算
        dot_products = np.dot(selected_features, query_vector_np.T).flatten()
        norm_features = np.linalg.norm(selected_features, axis=1)
        norm_query = np.linalg.norm(query_vector_np)

        # 避免除以零
        epsilon = 1e-10
        similarities = dot_products / (norm_features * norm_query + epsilon)

        # 组合结果
        similar_map = [
            [selected_features[i], similarities[i], selected_indices[i]]
            for i in range(len(similarities))
        ]
    else:
        similar_map = []

    # ===== 阶段3: 排序并截取Top-K结果 =====
    similar_map.sort(key=lambda x: x[1], reverse=True)
    similar_map = similar_map[:top_k]

    # ===== 阶段4: 添加元数据到结果 =====
    for item in similar_map:
        metadata_obj = feature_data["metadata"][item[2]]
        processed_meta = metadata_process(metadata_obj)
        item.append(processed_meta)

    return similar_map


def generate_images(image: Image.Image) -> List[Image.Image]:
    """
    生成不同尺度的图像

    Args:
        image: 原始图像

    Returns:
        不同尺度的图像列表
    """
    width, height = image.size
    images = []

    for scale_factor in Config.SCALE_FACTORS:
        # 计算不同尺度的图像长宽
        if height / width >= Config.TARGET_HEIGHT / Config.TARGET_WIDTH:
            target_width = width * scale_factor
            target_height = target_width * (Config.TARGET_HEIGHT / Config.TARGET_WIDTH)
        else:
            target_height = height * scale_factor
            target_width = target_height * (Config.TARGET_WIDTH / Config.TARGET_HEIGHT)

        # 计算截取的图像区域
        left = int((width - target_width) / 2)
        top = int((height - target_height) / 2)
        right = int((width + target_width) / 2)
        bottom = int((height + target_height) / 2)
        crop_box = (left, top, right, bottom)

        # 截取图像
        image_pre = np.array(image.crop(crop_box).resize((Config.TARGET_WIDTH, Config.TARGET_HEIGHT)))
        image_pre = zoned_gamma_correction(image_pre)
        image_pre = reduce_contrast_linear(image_pre)
        images.append(Image.fromarray(image_pre))
    return images


def calculate_bounding_rect(corners):
    """
    计算恰好包裹四个角点的轴对齐矩形

    参数:
    corners - img0在img1中的四个角点坐标，形状为(4, 2)

    返回:
    bounding_rect - 包裹角点的矩形，格式为(x, y, width, height)
    """
    # 计算当前边界框的中心点
    min_x = np.min([p[0] for p in corners])
    max_x = np.max([p[0] for p in corners])
    min_y = np.min([p[1] for p in corners])
    max_y = np.max([p[1] for p in corners])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 以中心点为基准进行缩放
    scaled_corners = []
    for p in corners:
        # 计算点相对于中心的向量
        vec_x = p[0] - center_x
        vec_y = p[1] - center_y

        # 缩放向量（1.5倍）
        scaled_vec_x = vec_x * 1.5
        scaled_vec_y = vec_y * 1.5

        # 计算缩放后的点坐标
        scaled_x = center_x + scaled_vec_x
        scaled_y = center_y + scaled_vec_y
        scaled_corners.append([scaled_x, scaled_y])

    # 计算缩放后的边界框
    x_coords = [p[0] for p in scaled_corners]
    y_coords = [p[1] for p in scaled_corners]

    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    return (int(min_x), int(min_y),
            int(max_x - min_x),
            int(max_y - min_y))

# ==================== 结果分析函数 ====================
def calculate_accuracy(map_data: Dict[str, Any], max_score_item: List, query_filename: str) -> Tuple[bool, str]:
    """
    计算场景匹配准确率

    Args:
        map_data: 地图数据
        max_score_item: 最高分匹配项
        query_filename: 查询文件名

    Returns:
        (是否匹配成功, 真实文件名)
    """
    # 解析文件名获取坐标
    dis1 = [int(os.path.splitext(os.path.basename(max_score_item[3]['path']))[0].partition('_')[0]),
            int(os.path.splitext(os.path.basename(max_score_item[3]['path']))[0].partition('_')[2])]
    dis2 = int(query_filename.split('/')[-1].split('.')[0])
    print(dis1, dis2)
    metadata_list_OK = []

    for item in map_data['metadata']:
        if isinstance(item['metadata'], str):
            try:
                start = item['metadata'].find('{')
                if start != -1:
                    item['metadata'] = ast.literal_eval(item['metadata'][start:])
                else:
                    item['metadata'] = {}
            except:
                item['metadata'] = {}

        if abs(item['metadata']['row_idx'] - dis1[0]) < 3 and abs(item['metadata']['col_idx'] - dis1[1]) < 3:
            metadata_list_OK.append(item)
            # print(item)

    # 检查是否匹配成功
    match_success = False
    for item in metadata_list_OK:
        if item['metadata'].get('fly_image') is not None:
            for data in item['metadata']['fly_image']:
                if int(data[0].split('.')[0])==dis2:
                    match_success = True
                    break

    return match_success


def save_reprojection_errors(result: Dict[str, Any], filename: str = 'reprojection_errors.txt'):
    """
    保存重投影误差到文件

    Args:
        result: 优化结果
        filename: 输出文件名
    """
    with open(filename, 'a') as file:
        for key, value in result['reprojection_errors'].items():
            if isinstance(value, (list, np.ndarray)):
                file.write(f"{key}:\n")
                for item in value:
                    file.write(f"\t{item}\n")
            else:
                file.write(f"{key}: {value}\n")
        file.write(f"position_error:{result['position_error']}\n\n")


# ==================== 可视化函数 ====================
def setup_error_plot() -> Tuple[plt.Figure, List, List]:
    """
    设置误差跟踪图

    Returns:
        (figure, axes, lines)
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Multi-dimensional Error Tracking')

    # 配置子图
    ax1.set_ylabel('X Error')
    ax2.set_ylabel('Y Error')
    ax3.set_ylabel('Z Error')
    ax4.set_ylabel('loss')
    ax3.set_xlabel('Iteration')

    # 初始化曲线
    line1, = ax1.plot([], [], 'bp-', label='X')
    line2, = ax2.plot([], [], 'rp-', label='Y')
    line3, = ax3.plot([], [], 'gp-', label='Z')
    line4, = ax4.plot([], [], 'p-', label='L')

    # 添加图例和网格
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend()
        ax.grid(True)

    return fig, [ax1, ax2, ax3, ax4], [line1, line2, line3, line4]


def update_error_plot(fig: plt.Figure, axes: List, lines: List, error_history: List[np.ndarray]):
    """
    更新误差跟踪图

    Args:
        fig: 图形对象
        axes: 子图列表
        lines: 曲线列表
        error_history: 误差历史数据
    """
    # 更新曲线数据
    iterations = np.arange(len(error_history))
    lines[0].set_xdata(iterations)
    lines[0].set_ydata([e[0] for e in error_history])
    lines[1].set_xdata(iterations)
    lines[1].set_ydata([e[1] for e in error_history])
    lines[2].set_xdata(iterations)
    lines[2].set_ydata([e[2] for e in error_history])
    lines[3].set_xdata(iterations)
    lines[3].set_ydata([e[3] for e in error_history])

    # 自动调整坐标范围
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # 重绘图形
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    print("加载模型...")
    model_map = load_model(Config.MODEL_MAP_FILE, device)  # 加载场景匹配模型
    model_feature = load_model_featurePoints(Config.MODEL_FEATURE_FILE, device)  # 加载特征点提取模型

    # 加载地图特征数据
    print("加载地图特征数据...")
    map_data = load_map_features(Config.MAP_DIR)
    print(f"加载成功! 共有 {map_data['index']['total_images']} 张图片")
    print(f"特征分块数: {len(map_data['features'])}")

    # 加载高程数据
    print("加载高程数据...")
    query_system = ElevationQuery(Config.HIGHT_NPY_PATH, Config.HIGHT_GEO_TXT_PATH)

    # 获取查询文件列表
    files_query = [Config.QUERY_DIR + '/' + f for f in os.listdir(Config.QUERY_DIR) if f.endswith('.png')]
    files_query = sorted(files_query, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    files_query = files_query[Config.QUERY_START_INDEX:Config.QUERY_END_INDEX]
    print(f"处理 {len(files_query)} 个查询文件")

    # 初始化统计变量
    acc = 0
    count = 0

    # 设置可视化
    fig, axes, lines = setup_error_plot()
    error_history = []

    #全局查询的范围
    query_which = Config.query_which
    width_index = Config.width_index
    max_x_index,max_y_index = get_max_idex(map_data)

    # 初始化数据记录器
    logger = DataLogger(auto_save_interval=1)
    # 记录实验参数
    logger.log_parameters(
        width_index=width_index,
        width_index_add=Config.width_index_add,
        QUERY_START_INDEX=Config.QUERY_START_INDEX,
        QUERY_END_INDEX = Config.QUERY_END_INDEX,
        dataSet_name = "vpair",
        error_threshold_stage1 = Config.ERROR_THRESHOLD_STAGE1,
        error_threshold_stage2 = Config.ERROR_THRESHOLD_STAGE2,
        error_threshold_stage3 = Config.ERROR_THRESHOLD_STAGE3

    )

    # 处理每个查询文件
    for f in files_query:
        print(f"\n处理文件: {f}")
        image = Image.open(f).convert("RGB")
        images = generate_images(image)

        start = time.perf_counter()

        # 处理多尺度图像
        all_results = []
        # plt.figure(figsize=(20, 9))
        for img_idx, img in enumerate(images[:3]):  # 只处理前3个查询图像
            query_img = preprocess_image(img)
            query_features = extract_features(model_map, query_img, device)
            results = query_image(map_data, query_features,query_which[0],query_which[1],max_x_index,max_y_index,width=width_index, top_k=5)
            all_results.extend(results)

        #     # 显示查询图像
        #     plt.subplot(3, 6, img_idx * 6 + 1)
        #     plt.imshow(img)
        #     plt.title(f"Query {img_idx + 1}\n{f.split('/')[-1].split('.')[0]}")
        #     plt.axis('off')
        #
        #     # 显示匹配结果
        #     for i, (_, score, _, meta) in enumerate(results, 1):
        #         plt.subplot(3, 6, img_idx * 6 + i + 1)
        #         plt.imshow(plt.imread(meta['path']))
        #         plt.title(f"{score:.3f},{os.path.splitext(os.path.basename(meta['path']))[0]}", fontsize=10)
        #         plt.axis('off')
        #
        # plt.show(block=False)
        # # plt.show()
        # plt.pause(5)

        # 取出相似度最高的图像
        all_scores = [item[1] for item in all_results]
        max_score = max(all_scores)
        max_score_item = [item for item in all_results if item[1] == max_score][0]

        end = time.perf_counter()
        time_use=end - start

        #获取航拍图像
        image_fly = image
        #获取参与优化的卫星图像
        image_map_pre = Image.open(max_score_item[3]['path']).convert("RGB")
        mkpts0_pre, mkpts1_pre = find_FeaturePoint(np.array(image_fly), np.array(image_map_pre), model_feature, device)
        print(len(mkpts0_pre))
        count += 1

        mkpts0_juge, mkpts1_juge = filter_matches(mkpts0_pre, mkpts1_pre, reproj_threshold=3)

        # image_show1(np.array(image_map_pre), np.array(image_fly), mkpts1_juge, mkpts0_juge)

        if len(mkpts0_juge) <= 15:
            width_index += Config.width_index_add
            print(f"pre特征点提取太少！")

            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 num_mkpts_first=len(mkpts0_pre),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1],
                                 isGet=0,
                                 time_use=time_use)
            continue
        acc += 1
        print(f"acc:{acc}/{count}={(float(acc) / count * 100):.3f}%----max_score:{max_score:.3f}")
        # # 可视化验证
        # if len(mkpts0_pre) > 30:
        #     image_show1(np.array(image_map_pre), np.array(image_fly), mkpts1_pre, mkpts0_pre)
        # else:
        #     print(f"特征点提取太少！")
        #     continue

        width_index = Config.width_index

        corners = get_img0_corners_in_img1(np.array(image_fly), np.array(image_map_pre), mkpts0_pre, mkpts1_pre)
        if corners is None:
            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 num_mkpts_first=len(mkpts0_pre),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1],
                                 isGet=0,
                                 time_use=time_use)
            continue

        cope_map = calculate_bounding_rect(corners)

        # 开始优化
        image_map_data = max_score_item[3]['metadata']
        image_map_cope = [
            image_map_data['pixel_x'] + int(cope_map[0]),
            image_map_data['pixel_y'] + int(cope_map[1]),
            int(cope_map[2]),
            int(cope_map[3])
        ]
        print(image_map_cope)
        #获取用于优化的卫星图像
        image_map,image_map_cope = crop_tif_rasterio(Config.TIF_PATH, image_map_cope[0], image_map_cope[1],
                                      image_map_cope[2], image_map_cope[3])
        if image_map is None:
            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 num_mkpts_first=len(mkpts0_pre),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1],
                                 isGet=0,
                                 time_use=time_use)
            continue

        rate1 = float(image_map.size[0]) / 1200
        rate2 = float(image_map.size[1]) / 1200
        image_map = image_map.resize((1200, 1200))

        mkpts0, mkpts1 = find_FeaturePoint(np.array(image_fly), np.array(image_map), model_feature, device)
        print(f"找到匹配点: {len(mkpts0)}对")

        # count += 1
        # # 计算场景匹配准确率
        # match_success = calculate_accuracy(map_data, max_score_item, f)
        # if match_success:
        #     acc += 1
        # print(f"acc:{acc}/{count}={(float(acc) / count * 100):.3f}%----max_score:{max_score:.3f}")

        image_show1(np.array(image_map), np.array(image_fly), mkpts1, mkpts0)

        # 可视化验证
        if len(mkpts0) > 200:
            # image_show1(np.array(image_map), np.array(image_fly), mkpts1, mkpts0)
            print('OK')
        else:
            print(f"特征点提取太少！")
            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 num_mkpts_first=len(mkpts0_pre),num_mkpts_last=len(mkpts0),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1],
                                 isGet=2,
                                 time_use=time_use)
            continue

        # 更新全局搜索起始点
        query_which[0] = max_score_item[3]['metadata']['row_idx']
        query_which[1] = max_score_item[3]['metadata']['col_idx']

        # 制作tif地图中截取的照片的元数据
        image_cope_date = {
            'pixel_x': image_map_cope[0],
            'pixel_y': image_map_cope[1],
            'width': image_map_cope[2],
            'height': image_map_cope[3],
            'image_name': f.split('/')[-1],
            'rate1': rate1,
            'rate2': rate2,
        }

        data_all = [mkpts0, mkpts1, image_fly, image_map, image_cope_date]

        # 执行位置优化
        result = location_opt(
            data_all,
            query_system,
            error_threshold_stage1=Config.ERROR_THRESHOLD_STAGE1,
            error_threshold_stage2=Config.ERROR_THRESHOLD_STAGE2,
            error_threshold_stage3=Config.ERROR_THRESHOLD_STAGE3,
            visualize=false
        )

        # if result is not None and result['reprojection_errors']['total']['mean']>100:
        #     result = None

        # 处理优化结果
        if result is not None:
            # if result['n_cleaned_points']==result['n_final_points'] and result['n_cleaned_points']<=10 and result['reprojection_errors']['total']['max']>0.4:
            if 0:
                logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                     acc=acc,
                                     count=count,
                                     isGet=2,
                                     time_use=time_use,
                                     num_mkpts_first=len(mkpts0_pre), num_mkpts_last=len(mkpts0),
                                     query_which_X=query_which[0],
                                     query_which_Y=query_which[1],
                                     n_original_points=result['n_original_points'],
                                     n_cleaned_points=result['n_cleaned_points'],
                                     n_final_points=result['n_final_points'],
                                     pre_position_X=result['predicted_position'][0],
                                     pre_position_Y=result['predicted_position'][1],
                                     pre_position_Z=result['predicted_position'][2],
                                     ture_position_X=result['true_position'][0],
                                     ture_position_Y=result['true_position'][1],
                                     ture_position_Z=result['true_position'][2],
                                     position_error_X=result['position_error'][0],
                                     position_error_Y=result['position_error'][1],
                                     position_error_Z=result['position_error'][2],
                                     distance_error=result['distance_error'])
                print("优化不符合条件，跳过...")
                continue
            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 isGet=1,
                                 time_use=time_use,
                                 num_mkpts_first=len(mkpts0_pre), num_mkpts_last=len(mkpts0),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1],
                                 n_original_points=result['n_original_points'], n_cleaned_points=result['n_cleaned_points'],
                                 n_final_points=result['n_final_points'],
                                 reprojection_errors_mean = result['reprojection_errors']['total']['mean'],
                                 reprojection_errors_max=result['reprojection_errors']['total']['max'],
                                 reprojection_errors_median=result['reprojection_errors']['total']['median'],
                                 max_up=result['max_up'],
                                 min_up= result['min_up'],
                                 median_up=result['median_up'],
                                 mean_up= result['mean_up'],
                                 variance_up=result['variance_up'],

                                 ture_position_X=result['true_position'][0],
                                 ture_position_Y=result['true_position'][1],
                                 ture_position_Z=result['true_position'][2],

                                 pre_position1_X=result['predicted_position1'][0],
                                 pre_position1_Y=result['predicted_position1'][1],
                                 pre_position1_Z=result['predicted_position1'][2],
                                 pre_error1_X=result['predicted_position1'][0]-result['true_position'][0],
                                 pre_error1_Y=result['predicted_position1'][1]-result['true_position'][1],
                                 pre_error1_Z=result['predicted_position1'][2]-result['true_position'][2],

                                 pre_position2_X=result['predicted_position2'][0],
                                 pre_position2_Y=result['predicted_position2'][1],
                                 pre_position2_Z=result['predicted_position2'][2],
                                 pre_error2_X=result['predicted_position2'][0] - result['true_position'][0],
                                 pre_error2_Y=result['predicted_position2'][1] - result['true_position'][1],
                                 pre_error2_Z=result['predicted_position2'][2] - result['true_position'][2],

                                 pre_position_X=result['predicted_position'][0],
                                 pre_position_Y=result['predicted_position'][1],
                                 pre_position_Z=result['predicted_position'][2],
                                 position_error_X=result['position_error'][0],
                                 position_error_Y=result['position_error'][1],
                                 position_error_Z=result['position_error'][2],

                                 position_pnp_X=result['camera_position_pnp'][0],
                                 position_pnp_Y=result['camera_position_pnp'][1],
                                 position_pnp_Z=result['camera_position_pnp'][2],
                                 error_pnp_X=result['camera_position_pnp'][0] - result['true_position'][0],
                                 error_pnp_Y=result['camera_position_pnp'][1] - result['true_position'][1],
                                 error_pnp_Z=result['camera_position_pnp'][2] - result['true_position'][2],

                                 position_A_error_X=result['pre_position_A'][0] - result['true_position'][0],
                                 position_A_error_Y=result['pre_position_A'][1] - result['true_position'][1],
                                 position_A_error_Z=result['pre_position_A'][2] - result['true_position'][2],

                                 pre_distance1=np.linalg.norm(
                                     np.array(result['predicted_position1']) - np.array(result['true_position'])),
                                 pre_distance2=np.linalg.norm(
                                     np.array(result['predicted_position2']) - np.array(result['true_position'])),
                                 distance_error=result['distance_error'],
                                 distance_error_A=np.linalg.norm(
                                     np.array(result['pre_position_A']) - np.array(result['true_position'])),
                                 pnp_distance=np.linalg.norm(
                                     np.array(result['camera_position_pnp']) - np.array(result['true_position'])),
                                 )

            print("\n===== 最终优化结果 =====")
            print(f"预测位置 (东, 北, 高): {result['predicted_position']}")
            print(f"真实位置 (东, 北, 高): {result['true_position']}")
            print(np.array(result['camera_position_pnp'])-np.array(result['true_position']))
            print(f"位置误差: {result['position_error']}")
            print(np.linalg.norm(np.array(result['camera_position_pnp']) - np.array(result['true_position'])))
            print(np.linalg.norm(np.array(result['pre_position_A']) - np.array(result['true_position'])))
            print(f"欧氏距离误差: {result['distance_error']:.2f}米")

            print("\n===== 匹配点分析 =====")
            print(f"原始匹配点数: {result['n_original_points']}")
            print(
                f"有效匹配点数: {result['n_cleaned_points']} ({result['n_cleaned_points'] / result['n_original_points'] * 100:.1f}%)")

            # 保存重投影误差
            save_reprojection_errors(result)

            # 更新误差图
            error_history.append(np.append(result['position_error'], result['distance_error']))
            update_error_plot(fig, axes, lines, error_history)

            # text = input("next")
        else:
            logger.log_variables(flyImage_name=f, stlImage_name=max_score_item[3]['path'],
                                 acc=acc,
                                 count=count,
                                 isGet=2,
                                 time_use=time_use,
                                 num_mkpts_first=len(mkpts0_pre), num_mkpts_last=len(mkpts0),
                                 query_which_X=query_which[0],
                                 query_which_Y=query_which[1])
            print("优化失败，跳过...")
            continue

        print("\n===============")


if __name__ == '__main__':
    main()