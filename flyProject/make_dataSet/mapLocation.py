import os
import pandas as pd
from PIL import Image
import numpy as np
import ast  # 用于安全评估元数据字符串
from tqdm import tqdm
import math
import time


# 获取图片EXIF中的元数据
def get_image_metadata(image_path):
    """从图片的EXIF数据中提取元数据"""
    try:
        img = Image.open(image_path).convert("RGB")
        raw_exif = img.info.get("exif", b"").decode('utf-8', errors='ignore')

        if '{' in raw_exif and '}' in raw_exif:
            metadata_str = raw_exif[raw_exif.find('{'):raw_exif.rfind('}') + 1]
            try:
                metadata = ast.literal_eval(metadata_str)
                return metadata
            except (SyntaxError, ValueError) as e:
                print(f"\n元数据解析错误: {image_path}, {e}")
                return None
        else:
            return None

    except Exception as e:
        print(f"\n元数据提取错误: {image_path}, {e}")
        return None

def update_image_metadata(image_path, metadata):
    """更新图片的EXIF元数据（完全替换为新的METADATA字节串）"""
    try:
        # 打开图片并构建新的EXIF数据
        img = Image.open(image_path)
        exif_data = b"METADATA:" + str(metadata).encode('utf-8')
        # 使用原格式保存（保留原尺寸和格式）
        img.save(
            image_path,
            exif=exif_data,
            format=img.format  # 保持原图片格式
        )
        return True

    except Exception as e:
        print(f"\n元数据更新错误: {image_path}, {e}")
        return False


# 计算地球上两点间的距离（Haversine公式）
def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两个坐标点间的实际距离（单位：米）"""
    # 地球半径（米）
    R = 6371000

    # 将度转换为弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine公式
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# 为数据集图片添加无人机图像信息
def add_drone_image_info_to_dataset(dataset_dir, drone_csv_path):
    """为数据集图片添加匹配的无人机图像信息"""
    # ================== 1. 加载无人机位置数据 ==================
    print("\n" + "=" * 60)
    print("步骤 1/3: 加载无人机位置数据")
    print("=" * 60)

    try:
        drone_df = pd.read_csv(drone_csv_path)
        required_columns = ['filename', 'lat', 'lon']

        if not all(col in drone_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in drone_df.columns]
            print(f"\n错误: CSV缺少必要列: {missing}")
            return

        # 转换为列表，提高效率
        drone_data = drone_df[required_columns].values.tolist()
        print(f"✓ 已加载 {len(drone_data)} 个无人机图像位置")

    except Exception as e:
        print(f"\nCSV文件加载错误: {e}")
        return

    # ================== 2. 获取数据集图片元数据 ==================
    print("\n" + "=" * 60)
    print("步骤 2/3: 加载数据集图片元数据")
    print("=" * 60)

    # 获取所有PNG文件
    png_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]

    # png_files = png_files[:1000]

    print(f"在文件夹中发现 {len(png_files)} 个PNG文件")
    # 解析卫星图片元数据
    dataset_images = []
    progress = tqdm(
        png_files,
        desc="解析卫星图片元数据",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for filename in progress:
        filepath = os.path.join(dataset_dir, filename)
        metadata = get_image_metadata(filepath)

        if metadata and 'lat' in metadata and 'lon' in metadata:
            metadata['filename'] = filename
            metadata['filepath'] = filepath

            # 初始化每个卫星图片的最佳匹配字段
            metadata['best_drone'] = None  # 存储最佳匹配的无人机文件名
            metadata['best_drone_distance'] = float('inf')  # 初始化为最大距离

            dataset_images.append(metadata)
            progress.set_postfix_str(f"有效: {len(dataset_images)}")
        else:
            progress.set_postfix_str(f"跳过: {filename[:10]}...")

    valid_count = len(dataset_images)
    print(f"✓ 成功加载 {valid_count}/{len(png_files)} 个卫星图片的元数据")

    if valid_count < len(png_files):
        print(f"警告: {len(png_files) - valid_count} 个文件元数据无效或缺少位置信息")

    if not dataset_images:
        print("错误: 没有有效的卫星图片可以匹配")
        return

    # ================== 3. 匹配无人机图像 ==================
    print("\n" + "=" * 60)
    print("步骤 3/3: 匹配无人机图像")
    print("=" * 60)
    print(f"为每个卫星图片查找距离≤100米的最接近无人机图像...")

    stats = {
        'drones_processed': 0,  # 处理的无人机图片总数
        'drones_matched': 0,  # 匹配成功的无人机图片数（<=100m）
        'drones_over_100m': 0,  # 距离>100米的无人机图片数
        'drones_no_satellite': 0,  # 没有卫星图片匹配的无人机数
        'satellites_matched': 0,  # 被匹配上的卫星图片数
        'min_distance': float('inf'),  # 所有匹配的最小距离
        'max_distance': 0,  # 所有匹配的最大距离
        'avg_distance': 0  # 所有匹配的平均距离
    }

    # 为每个无人机图片寻找最近的卫星图片
    drone_progress = tqdm(
        drone_data,
        desc="处理无人机图像",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for drone_info in drone_progress:
        drone_filename, drone_lat, drone_lon = drone_info
        stats['drones_processed'] += 1

        min_distance = float('inf')
        closest_satellite = None

        # 遍历所有卫星图片寻找最近的一个
        for sat_meta in dataset_images:
            distance = haversine_distance(
                drone_lat, drone_lon,
                sat_meta['lat'], sat_meta['lon']
            )

            if distance < min_distance:
                min_distance = distance
                closest_satellite = sat_meta

        # 如果找到卫星图片且距离在200米内
        if closest_satellite and min_distance <= 200:
            # 仅当当前无人机距离更小时才更新
            if min_distance < closest_satellite['best_drone_distance']:
                closest_satellite['best_drone'] = os.path.splitext(drone_filename)[0]
                closest_satellite['best_drone_distance'] = min_distance
                stats['drones_matched'] += 1
                drone_progress.set_postfix({
                    '状态': f'匹配! {min_distance:.1f}m ↓',
                    '总匹配': stats['drones_matched']
                })
            else:
                drone_progress.set_postfix({
                    '状态': f'已有更近 {min_distance:.1f}m',
                    '总匹配': stats['drones_matched']
                })
        elif closest_satellite:  # 距离大于200米
            stats['drones_over_100m'] += 1
            drone_progress.set_postfix({
                '状态': f'距离远 {min_distance:.1f}m',
                '总匹配': stats['drones_matched']
            })
        else:  # 没有找到任何卫星图片
            stats['drones_no_satellite'] += 1
            drone_progress.set_postfix({'状态': '无卫星图片', '总匹配': stats['drones_matched']})

    drone_progress.close()

    # 计算匹配统计
    matched_distances = []

    # 最终确定哪些卫星图片有匹配
    for sat_meta in dataset_images:
        if sat_meta['best_drone'] is not None:
            matched_distances.append(sat_meta['best_drone_distance'])
            stats['satellites_matched'] += 1

            # 添加无人机图像信息到元数据
            sat_meta['drone_image'] = sat_meta['best_drone']

        # 清理临时字段
        del sat_meta['best_drone']
        del sat_meta['best_drone_distance']

    # 计算距离统计
    if matched_distances:
        stats['min_distance'] = min(matched_distances)
        stats['max_distance'] = max(matched_distances)
        stats['avg_distance'] = sum(matched_distances) / len(matched_distances)

    # 匹配报告
    print("\n" + "=" * 60)
    print("匹配报告")
    print("=" * 60)
    print(f"无人机图片总数: {stats['drones_processed']}")
    print(f"  ✓ 匹配成功 (<=100米): {stats['drones_matched']}")
    print(f"  ✗ 距离过远 (>100米): {stats['drones_over_100m']}")
    print(f"  ✗ 无卫星图片匹配: {stats['drones_no_satellite']}")

    print(f"\n卫星图片总数: {len(dataset_images)}")
    print(f"  ✓ 匹配最佳无人机图片: {stats['satellites_matched']}")
    print(f"  ✗ 无匹配: {len(dataset_images) - stats['satellites_matched']}")

    if matched_distances:
        print(f"\n匹配距离统计 (米):")
        print(f"  最小值: {stats['min_distance']:.1f}m")
        print(f"  最大值: {stats['max_distance']:.1f}m")
        print(f"  平均值: {stats['avg_distance']:.1f}m")

    # ================== 4. 更新元数据 ==================
    print("\n" + "=" * 60)
    print("更新数据集图片元数据")
    print("=" * 60)

    stats.update({
        'updated': 0,
        'update_failed': 0
    })

    with open('2.txt', 'a') as f:  # 使用'a'模式追加写入，不会覆盖之前的内容
        for sat_meta in dataset_images:
            # 只更新有匹配的卫星图片
            if 'drone_image' in sat_meta:
                line = f"{sat_meta['drone_image']} {sat_meta['filepath']}\n"
                print(line.strip())  # 打印到控制台
                f.write(line)  # 写入文件
                try:
                    if update_image_metadata(sat_meta['filepath'], sat_meta):
                        stats['updated'] += 1
                    else:
                        stats['update_failed'] += 1
                except Exception as e:
                    print(f"\n更新时出错: {sat_meta['filepath']} - {e}")
                    stats['update_failed'] += 1

    # 最终报告
    print("\n" + "=" * 60)
    print("任务总结")
    print("=" * 60)
    print(f"卫星图片总数: {len(dataset_images)}")
    print(f"  ✓ 成功匹配并更新: {stats['updated']}")
    print(f"  ✗ 更新失败: {stats['update_failed']}")

    if stats['update_failed'] == 0 and stats['satellites_matched'] == stats['updated']:
        print("\n✓ 所有匹配的卫星图片已成功更新！")
    else:
        print("\n! 操作完成但有未更新的图片，请检查错误日志")


# 主函数
def main():
    """主函数"""
    # ======== 用户参数 ========
    dataset_dir = "/home/superz/Desktop/data/vpair/dataSet"  # 数据集文件夹路径
    drone_csv_path = "/home/superz/Desktop/data/vpair/poses.csv"  # 无人机图像位置CSV

    # ===== 执行主任务 =====
    print("=" * 60)
    print("无人机图像与卫星数据集匹配程序")
    print("=" * 60)
    print(f"卫星数据集路径: {dataset_dir}")
    print(f"无人机位置文件: {drone_csv_path}")
    print("\n开始处理...")

    start_time = time.time()

    add_drone_image_info_to_dataset(dataset_dir, drone_csv_path)

    elapsed = time.time() - start_time
    print(f"\n总处理时间: {elapsed:.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    main()