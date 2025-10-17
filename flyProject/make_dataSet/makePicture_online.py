# 多线程使用30%偏移滑动方法，分割tif，制作数据集（对vpair_dataset），数据集照片命名为（行索引-列索引）
import numpy as np
from PIL import Image
import os
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
import concurrent.futures
import math
import warnings
import cv2

gdal.UseExceptions()
warnings.simplefilter("ignore", category=FutureWarning, append=True)


def pixel_to_lonlat(x_pixel, y_pixel, tif_path):
    """计算像素位置的地理坐标"""
    dataset = gdal.Open(tif_path)
    if not dataset:
        raise ValueError(f"无法打开TIFF文件: {tif_path}")

    geotrans = dataset.GetGeoTransform()
    width, height = dataset.RasterXSize, dataset.RasterYSize

    if not (0 <= x_pixel < width and 0 <= y_pixel < height):
        raise ValueError(f"坐标 ({x_pixel},{y_pixel}) 超出影像范围: {width}x{height}")

    longitude = geotrans[0] + x_pixel * geotrans[1]
    latitude = geotrans[3] + y_pixel * geotrans[5]
    return longitude, latitude


def image_generate(tif_path, crop_area):
    """从TIFF生成裁剪图像"""
    dataset = gdal.Open(tif_path)
    if not dataset:
        return None

    x_start, y_start, x_end, y_end = map(int, crop_area)
    width, height = dataset.RasterXSize, dataset.RasterYSize

    if (x_start < 0 or y_start < 0 or
            x_end > width or y_end > height or
            x_end <= x_start or y_end <= y_start):
        return None

    try:
        cropped_data = dataset.ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
        if cropped_data is None or cropped_data.size == 0:
            return None

        rgb_data = np.dstack((
            cropped_data[0],  # 红波段
            cropped_data[1],  # 绿波段
            cropped_data[2],  # 蓝波段
        ))
        return Image.fromarray(rgb_data, 'RGB')
    except Exception as e:
        print(f"图像生成错误: {e}")
        return None


def generate_crop_areas(total_width, total_height, size, offset):
    """生成所有裁剪区域坐标并记录行列索引"""
    crop_areas = []
    row_idx = 0
    y = 0

    while y + size[1] <= total_height:
        col_idx = 0
        x = 0
        while x + size[0] <= total_width:
            # 存储格式为: (行索引, 列索引, (x_start, y_start, x_end, y_end))
            crop_areas.append((row_idx, col_idx, (x, y, x + size[0], y + size[1])))
            x += offset[0]
            col_idx += 1

        # 移动到下一行
        y += offset[1]
        row_idx += 1
        print(f"已生成第 {row_idx} 行的裁剪区域...")

    print(f"总共生成: {len(crop_areas)} 个裁剪区域")
    return crop_areas


def process_task(args):
    """处理单个裁剪任务"""
    # 参数包含行列索引
    row_idx, col_idx, crop_area, tif_path, df_list, output_dir, size = args

    # 在每个线程中独立打开处理
    image = image_generate(tif_path, crop_area)
    if image is None:
        return

    # 计算中心点的经纬度
    center_x = int(crop_area[0] + (crop_area[2] - crop_area[0]) / 2)
    center_y = int(crop_area[1] + (crop_area[3] - crop_area[1]) / 2)

    try:
        lon, lat = pixel_to_lonlat(center_x, center_y, tif_path)
    except ValueError as e:
        print(f"坐标计算错误: {e}")
        return

    # 准备元数据
    metadata = {
        "pixel_x": crop_area[0],
        "pixel_y": crop_area[1],
        "width": crop_area[2] - crop_area[0],
        "height": crop_area[3] - crop_area[1],
        "lon": lon,
        "lat": lat,
        "row_idx": row_idx,
        "col_idx": col_idx
    }

    # 保存图像 - 使用行列索引命名文件
    try:
        # 调整图像尺寸到640x480
        image = image.resize((640, 480), resample=Image.LANCZOS)

        # 创建EXIF元数据
        exif_data = b"METADATA:" + str(metadata).encode('utf-8')

        # 使用行列索引命名文件
        output_path = os.path.join(output_dir, f"{row_idx}_{col_idx}.png")
        image.save(output_path, exif=exif_data)
    except Exception as e:
        print(f"保存图像时出错: {e}")


def main():
    """主函数"""
    # ======== 用户需要修改这些参数 ========
    output_dir = "/home/superz/Desktop/data/vpair/dataSet"
    path_tif = "/home/superz/Desktop/data/vpair/vpair.tif"
    path_csv = "/home/superz/Desktop/data/vpair/poses.csv"

    crop_size = [640 * 1.2, 480 * 1.2]  # 裁剪区域大小
    offset_rate = 0.3  # 滑动偏移率
    thread_workers = 32  # 线程数量

    # ===================================

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载无人机位置数据
    df = pd.read_csv(path_csv)
    df_array = np.array(df)
    df_list = df_array.tolist()

    # 获取TIFF文件信息
    tif_dataset = gdal.Open(path_tif)
    if not tif_dataset:
        raise ValueError(f"无法打开TIFF文件: {path_tif}")

    width_tif = tif_dataset.RasterXSize
    height_tif = tif_dataset.RasterYSize
    print(f"TIFF图像尺寸: {width_tif}x{height_tif}")

    # 计算实际偏移量
    offset_pixels = [int(crop_size[0] * offset_rate), int(crop_size[1] * offset_rate)]
    print(f"使用滑动偏移量: 水平 {offset_pixels[0]}px, 垂直 {offset_pixels[1]}px")

    # 生成所有裁剪区域（包含行列索引）
    crop_areas = generate_crop_areas(width_tif, height_tif, crop_size, offset_pixels)
    total_tasks = len(crop_areas)
    print(f"总任务数: {total_tasks}")

    # 准备任务参数 - 包含行列索引
    tasks = [
        (row_idx, col_idx, crop_area, path_tif, df_list, output_dir, crop_size)
        for row_idx, col_idx, crop_area in crop_areas
    ]

    # 使用线程池处理任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_workers) as executor:
        results = list(tqdm(
            executor.map(process_task, tasks),
            total=total_tasks,
            desc="处理图像裁剪",
            unit="crop"
        ))

    print("所有任务处理完成")
    print(f"图片已保存到: {output_dir}")
    print(f"文件名格式为: '行索引_列索引.png' (例如: '0_0.png', '0_1.png' 等)")


if __name__ == "__main__":
    main()