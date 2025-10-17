#制作MixVPR模型的训练集，将同一地点的卫星图片和航拍图片归类为一组照片（每一个卫星图片或者航拍图片提取5张图片），根据多个地点得到多组照片，组成一个训练集
#训练效果还行

import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from osgeo import gdal
gdal.UseExceptions()
import warnings
warnings.simplefilter("ignore", category=FutureWarning, append=True)

def lonlat_to_pixel(lon, lat, geotrans):
    x_origin = geotrans[0]  # 左上角经度
    y_origin = geotrans[3]  # 左上角纬度
    pixel_width = geotrans[1]  # 经度方向分辨率
    pixel_height = geotrans[5]  # 纬度方向分辨率
    rotation_x = geotrans[2]  # 通常为0
    rotation_y = geotrans[4]  # 通常为0

    # 计算行列式（判断是否有旋转）
    det = pixel_width * pixel_height - rotation_x * rotation_y
    if det == 0:
        raise ValueError("仿射变换矩阵不可逆（可能包含旋转或分辨率为零）")

    # 计算像素坐标
    x = ((lon - x_origin) * pixel_height - (lat - y_origin) * rotation_x) / det
    y = ((lat - y_origin) * pixel_width - (lon - x_origin) * rotation_y) / det

    # 四舍五入后转整数
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    return x, y

def image_generate(image_tif,image_csv,directory_image,im_geotrans,rate,size=[640,480],offset = 0.3):
    x, y = lonlat_to_pixel(image_csv[4], image_csv[3], im_geotrans)
    square = Image.open(directory_image + "/" + image_csv[1])
    width, height = square.size
    width = int(width * rate)
    height = int(height * rate)
    square = square.resize((width, height))

    x_start, y_start = x - width / 2, y - height / 2  # 左上角起点
    x_end, y_end = x + width / 2, y + height / 2

    if x_start<0 or y_start<0 or x_end>image_tif.RasterXSize or y_end>image_tif.RasterYSize:
        return []

    cropped_data = image_tif.ReadAsArray(
        xoff=x_start,  # 起始列
        yoff=y_start,  # 起始行
        xsize=x_end - x_start,  # 裁剪宽度
        ysize=y_end - y_start,  # 裁剪高度
    )
    rgb_data = np.dstack((
        cropped_data[0],  # 红波段
        cropped_data[1],  # 绿波段
        cropped_data[2],  # 蓝波段
    ))

    rgb_data = Image.fromarray(rgb_data, 'RGB')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(square)
    axes[0].axis('off')  # 关闭坐标轴
    axes[0].set_title('Image 1')
    # 显示第二张图片
    axes[1].imshow(rgb_data)
    axes[1].axis('off')
    axes[1].set_title('Image 2')
    # 调整子图间距
    plt.tight_layout()
    plt.show()  # 非阻塞模式
    plt.pause(5)  # 暂停1秒
    plt.close()  # 关闭窗口
    exit(1)

    crop_areas = []

    crop_areas.append((width/2-size[0]/2, height/2-size[1]/2, width/2+size[0]/2, height/2+size[1]/2))
    crop_areas.append((crop_areas[0][0]+int(size[0]*offset),crop_areas[0][1],crop_areas[0][2]+int(size[0]*offset),crop_areas[0][3]))
    crop_areas.append((crop_areas[0][0]-int(size[0]*offset),crop_areas[0][1],crop_areas[0][2]-int(size[0]*offset),crop_areas[0][3]))
    crop_areas.append((crop_areas[0][0],crop_areas[0][1]+int(size[1]*offset),crop_areas[0][2],crop_areas[0][3]+int(size[1]*offset)))
    crop_areas.append((crop_areas[0][0], crop_areas[0][1]-int(size[1]*offset), crop_areas[0][2], crop_areas[0][3]-int(size[1]*offset)))

    cropped_images = []
    for crop_area in crop_areas:
        if crop_area[0]<0 or crop_area[1]<0:
            print("image too small")
            return []
        if crop_area[2]>=width or crop_area[3]>=height:
            print("image too small")
            return []
        cropped_images.append(square.crop(crop_area))
        cropped_images.append(Image.fromarray(rgb_data, 'RGB').crop(crop_area))



    return cropped_images

if __name__ == "__main__":
    im_geotrans = (119.805926, 2.682263763820894e-06, 0.0, 32.355491, 0.0, -2.6822856672701103e-06)
    #06: im_geotrans = (109.63516, 2.6821331353621427e-06, 0.0, 32.373177, 0.0, -2.6823108384455756e-06)
    directory_image = r"/media/superz/C0963E58963E4EE2/project/dataSet/UAV_VisLoc_dataset/03/drone"
    directory_csv = r"/media/superz/C0963E58963E4EE2/project/dataSet/UAV_VisLoc_dataset/03/03.csv"
    path_tif = r"/media/superz/C0963E58963E4EE2/project/dataSet/UAV_VisLoc_dataset/03/satellite03.tif"

    image_tif = gdal.Open(path_tif)
    print(f"tif image size:{image_tif.RasterXSize},{image_tif.RasterYSize}")

    #读取数据集csv文件，储存到列表df_list，每一条都包含一张无人机图片的信息
    df = pd.read_csv(directory_csv)
    df_array = np.array(df)
    df_list = df_array.tolist()

    for k,df_im in enumerate(df_list):
        cropped_images = image_generate(image_tif,df_im,directory_image,im_geotrans,rate=0.5)
        for i,cropped_image in enumerate(cropped_images):
            cropped_image.save('./dataSet/image/'+str(df_im[0])+'_'+str(i)+'.png')
            data_csv = {
                "place_id":df_im[0],
                "picture_id":i
            }
            df = pd.DataFrame([data_csv])
            df.to_csv('./dataSet/data.csv', mode='a', header=False, index=False)

        print(k)

