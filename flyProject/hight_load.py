import numpy as np


class ElevationQuery:
    def __init__(self, npy_path, geo_txt_path):
        """
        初始化高程查询系统
        参数:
            npy_path: 高程数据.npy文件路径
            geo_txt_path: 地理信息文本文件路径
        """
        self.elevation_data = np.load(npy_path)
        self.geotransform, self.projection = self.load_geoinfo(geo_txt_path)

    @staticmethod
    def load_geoinfo(geo_txt_path):
        with open(geo_txt_path, 'r') as f:
            lines = f.readlines()
            geotransform = eval(lines[0].split(': ')[1].strip())
            projection = lines[1].split(': ')[1].strip()
        return geotransform, projection

    def lonlat_to_pixel(self, lon, lat):
        x_origin = self.geotransform[0]
        y_origin = self.geotransform[3]
        pixel_width = self.geotransform[1]
        pixel_height = self.geotransform[5]

        px = int((lon - x_origin) / pixel_width)
        py = int((lat - y_origin) / pixel_height)

        return px, py

    def get_elevation(self, lon, lat):
        px, py = self.lonlat_to_pixel(lon, lat)

        if 0 <= px < self.elevation_data.shape[1] and 0 <= py < self.elevation_data.shape[0]:
            return self.elevation_data[py, px]
        else:
            raise ValueError(f"坐标({lon}, {lat})超出数据范围")


# 使用示例
if __name__ == "__main__":
    # 初始化查询系统
    npy_path = '/home/superz/Desktop/data/vpair/hight_elevation.npy'
    geo_txt_path = '/home/superz/Desktop/data/vpair/hight_elevation_geo.txt'
    query_system = ElevationQuery(npy_path, geo_txt_path)

    # 查询海拔高度
    try:
        lon ,lat= 7.02186713708904 , 50.65026073717327
        elevation = query_system.get_elevation(lon, lat)
        print(f"经纬度({lon}, {lat})处的海拔高度为: {elevation}米")
    except ValueError as e:
        print(e)