#读取高程数据，并制作成npy文件
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from osgeo import gdal
gdal.UseExceptions()
import warnings
warnings.simplefilter("ignore", category=FutureWarning, append=True)



def generate_elevation_files(tif_path):
    """
    从高程TIF文件生成NPY数据和地理信息文本文件
    参数:
        tif_path: 输入的.tif高程文件路径
    """
    # 打开TIF文件
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise ValueError(f"无法打开TIF文件: {tif_path}")

    # 1. 提取高程数据
    band = dataset.GetRasterBand(1)

    # 检查数据类型并相应处理
    if band.DataType in (gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_Int32, gdal.GDT_UInt16, gdal.GDT_UInt32):
        # 整数类型数据
        elevation_data = band.ReadAsArray()
        nodata_value = band.GetNoDataValue()
        if nodata_value is not None:
            # 使用-9999作为无效数据标记
            elevation_data[elevation_data == nodata_value] = -9999
    else:
        # 浮点类型数据
        elevation_data = band.ReadAsArray().astype(np.float32)
        nodata_value = band.GetNoDataValue()
        if nodata_value is not None:
            elevation_data[elevation_data == nodata_value] = np.nan

    # 2. 获取地理信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 3. 生成输出文件名
    base_path = os.path.splitext(tif_path)[0]
    npy_path = f"{base_path}_elevation.npy"
    geo_txt_path = f"{base_path}_elevation_geo.txt"

    # 4. 保存高程数据为NPY文件
    np.save(npy_path, elevation_data)
    print(f"高程数据已保存到: {npy_path}")

    # 5. 保存地理信息到文本文件
    with open(geo_txt_path, 'w') as f:
        f.write(f"Geotransform: {geotransform}\n")
        f.write(f"Projection: {projection}\n")
        f.write(f"NoDataValue: {nodata_value}\n")
        f.write(f"DataType: {band.DataType}\n")
    print(f"地理信息已保存到: {geo_txt_path}")

    # 关闭数据集
    dataset = None

    return npy_path, geo_txt_path


# 使用示例
if __name__ == "__main__":
    tif_path = r"/home/superz/Desktop/data/vpair/hight.tif"
    npy_path, geo_txt_path = generate_elevation_files(tif_path)

    # 验证生成的文件
    print("\n生成文件验证:")
    print(f"NPY文件存在: {os.path.exists(npy_path)}")
    print(f"地理信息文件存在: {os.path.exists(geo_txt_path)}")