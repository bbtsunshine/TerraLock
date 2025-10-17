#制作MixVPR模型的训练集，适用于vpair，目的是提升模型在vpair中的匹配效果，读取目标地区的tif文件，每一个地区制作5中图片，做成数据集
import cv2
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

def wiener_deblur(img, kernel_size=(5, 5), noise_var=1e-2):
    """
    真正的维纳滤波去模糊（支持彩色图像）
    :param img: 输入模糊图像 (BGR格式，3通道)
    :param kernel_size: 模糊核大小 (默认5x5)
    :param noise_var: 噪声方差 (默认0.01)
    :return: 去模糊后的图像
    """
    # 1. 生成高斯模糊核
    kernel = cv2.getGaussianKernel(kernel_size[0], 0)
    kernel = kernel * kernel.T
    kernel /= np.sum(kernel)  # 归一化核

    # 2. 初始化结果图像
    deblurred = np.zeros_like(img, dtype=np.float32)

    # 3. 对每个通道分别处理
    for c in range(3):
        channel = img[:, :, c].astype(np.float32) / 255.0

        # 计算频域分量
        channel_fft = np.fft.fft2(channel)
        kernel_fft = np.fft.fft2(kernel, s=channel.shape)  # 匹配通道尺寸

        # 维纳滤波公式
        wiener_factor = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise_var)
        deblurred_fft = channel_fft * wiener_factor

        # 逆变换回空间域
        deblurred_channel = np.fft.ifft2(deblurred_fft)
        deblurred_channel = np.abs(deblurred_channel)

        # 存储结果（转换回0-255范围）
        deblurred[:, :, c] = np.clip(deblurred_channel * 255, 0, 255)

    return deblurred.astype(np.uint8)

#输入裁减区域，输出照片
def image_generate(image_tif,crop_area):
    x_start, y_start = crop_area[0], crop_area[1] # 左上角起点
    x_end, y_end =crop_area[2], crop_area[3]

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
    return Image.fromarray(rgb_data, 'RGB').resize((640,480))

#获取裁减区域，一共5个区域
def crop_generate(x,y,size,offset,size_tif):
    crop_areas = []
    crop_areas.append((x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2))
    crop_areas.append((crop_areas[0][0] + int(size[0] * offset), crop_areas[0][1], crop_areas[0][2] + int(size[0] * offset),crop_areas[0][3]))
    crop_areas.append((crop_areas[0][0] - int(size[0] * offset), crop_areas[0][1], crop_areas[0][2] - int(size[0] * offset),crop_areas[0][3]))
    crop_areas.append((crop_areas[0][0], crop_areas[0][1] + int(size[1] * offset), crop_areas[0][2],crop_areas[0][3] + int(size[1] * offset)))
    crop_areas.append((crop_areas[0][0], crop_areas[0][1] - int(size[1] * offset), crop_areas[0][2],crop_areas[0][3] - int(size[1] * offset)))

    for crop_area in crop_areas:
        if crop_area[0]<0 or crop_area[1]<0:
            print("image too small")
            return []
        if crop_area[2]>=size_tif[0] or crop_area[3]>=size_tif[1]:
            print("image too small")
            return []
    return crop_areas

if __name__ == "__main__":
    path_tif = r"/home/superz/Desktop/data/vpair/vpair.tif"

    image_tif = gdal.Open(path_tif)
    print(f"tif image size:{image_tif.RasterXSize},{image_tif.RasterYSize}")
    size = [640 * 1.2, 480 * 1.2]

    place_id = 0
    plt.figure(1)
    print(int(image_tif.RasterXSize/(size[0]*3))*int(image_tif.RasterYSize/(size[1]*3)))



    for i in range(int(image_tif.RasterXSize/(size[0]*3))):
        for j in range(int(image_tif.RasterYSize/(size[1]*3))):

            image_id = 0
            crop_areas = crop_generate(i*size[0],j*size[1],size,0.3,[image_tif.RasterXSize,image_tif.RasterYSize])

            for n,crop_area in enumerate(crop_areas):
                image = image_generate(image_tif,crop_area)

                image_np = np.array(image)
                image = wiener_deblur(image_np, kernel_size=(6,6), noise_var=0.1)
                # def sigmoid_contrast(img, alpha=10, beta=0.5):
                #     img_float = img.astype(np.float32) / 255.0
                #     enhanced = 1 / (1 + np.exp(-alpha * (img_float - beta)))
                #     return (enhanced * 255).astype(np.uint8)
                # high_contrast_img = sigmoid_contrast(image_np, alpha=8, beta=0.5)
                # # 转回 PIL.Image
                image = Image.fromarray(image, 'RGB')


                # plt.imshow(image)
                # plt.show(block=False)
                # plt.pause(0.5)

                image.save(f"/home/superz/Desktop/data/vpair/data_train/image/{place_id}_{image_id}.png")
                data_csv = {
                    "place_id": place_id,
                    "picture_id": image_id
                }
                image_id += 1
                df = pd.DataFrame([data_csv])
                df.to_csv('/home/superz/Desktop/data/vpair/data_train/data.csv', mode='a', header=False, index=False)

                if n == 0:
                    image.transpose(Image.FLIP_LEFT_RIGHT).save(f"/home/superz/Desktop/data/vpair/data_train/image/{place_id}_{image_id}.png")
                    data_csv = {
                        "place_id": place_id,
                        "picture_id": image_id
                    }
                    df = pd.DataFrame([data_csv])
                    df.to_csv('/home/superz/Desktop/data/vpair/data_train/data.csv', mode='a', header=False,
                              index=False)
                    image_id += 1

                    image.transpose(Image.ROTATE_180).save(
                        f"/home/superz/Desktop/data/vpair/data_train/image/{place_id}_{image_id}.png")
                    data_csv = {
                        "place_id": place_id,
                        "picture_id": image_id
                    }
                    df = pd.DataFrame([data_csv])
                    df.to_csv('/home/superz/Desktop/data/vpair/data_train/data.csv', mode='a', header=False,
                              index=False)
                    image_id += 1

            place_id +=1
            print(place_id)

