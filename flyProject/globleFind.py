#MixVPR的接口文件，包含加载MixVPR模型，调用该模型生成图片特征向量，返回在数据库中与该图片最相似的前5个图片
#将数据库照片转换为特征向量储存到npy文件

import sys
from pathlib import Path
import os
# 添加项目根目录和 MixVPR 目录到路径
# 正确设置Python路径
project_root = str(Path(__file__).parent)
sys.path.extend([
    project_root,
    os.path.join(project_root, 'MixVPR')
])

import torch
import numpy as np
from PIL import Image, ExifTags
import torchvision.transforms as transforms
import glob
import json
from tqdm import tqdm  # 用于进度条
from MixVPR.main import VPRModel


def load_model(ckpt_path,device):
    model = VPRModel.load_from_checkpoint(
        ckpt_path,
        backbone_arch='resnet50',
        layers_to_crop=[4],
        agg_arch='MixVPR',
        agg_config={
            'in_channels': 1024,
            'in_h': 20,
            'in_w': 20,
            'out_channels': 1024,
            'mix_depth': 4,
            'mlp_ratio': 1,
            'out_rows': 4
        }
    )

    model.to(device)
    # 可选：将模型设为评估模式
    model.eval()
    model.freeze()

    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

# 图像预处理
def preprocess_image(img, img_size=(320, 320)):
    """预处理图像：调整大小、归一化、转为张量"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img)


# 提取图像特征
def extract_features(model, image, device):
    image = image.to(device)
    """从单张图像提取特征向量"""
    with torch.no_grad():
        features = model(image.unsqueeze(0))  # 添加批次维度
    return features.cpu().numpy().flatten()


# 计算特征相似度
def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def find_similar_vector_GPU(model, query_image, Map_data, device):
    # 处理查询图像
    query_img = preprocess_image(query_image)
    query_features = extract_features(model, query_img, device)  # 假设返回的是torch.Tensor

    # 将数据库特征转换为GPU张量（一次性转换）
    db_features = torch.stack([torch.from_numpy(item["feature"]).to(device) for item in Map_data])

    # 确保查询特征在GPU上（形状：[1, feature_dim]）
    if isinstance(query_features, np.ndarray):
        query_features = torch.from_numpy(query_features).to(device)
    query_features = query_features.unsqueeze(0)  # 添加batch维度 -> [1, feature_dim]

    # 批量计算余弦相似度（GPU加速）
    def batch_cosine_similarity(query, db):
        """PyTorch GPU批量余弦相似度计算"""
        query_norm = torch.norm(query, p=2, dim=1, keepdim=True)  # [1, 1]
        db_norm = torch.norm(db, p=2, dim=1, keepdim=True)  # [N, 1]
        similarity = torch.mm(query, db.T) / (query_norm * db_norm.T)  # [1, N]
        return similarity.squeeze(0)  # [N]

    similarities = batch_cosine_similarity(query_features, db_features)

    # 获取排序结果（直接在GPU上操作减少数据传输）
    _, indices = torch.sort(similarities, descending=True)
    top_indices = indices[:5]

    # 构造返回结果
    similar_map = [
        [Map_data[i.item()]["metadata"], similarities[i].item()]
        for i in top_indices
    ]

    return similar_map

# 主函数
def find_similar_vector(model, query_image, Map_data, device):

    # 处理查询图像
    query_img = preprocess_image(query_image)
    query_features = extract_features(model, query_img,device)

    # 查找相似图像
    similar_map = []
    for item in Map_data:
        db_features = item["feature"]
        # 计算相似度
        similarity = cosine_similarity(query_features, db_features)
        similar_map.append([item["metadata"], similarity])

    # 按相似度排序
    similar_map.sort(key=lambda x: x[1], reverse=True)
    similar_map = similar_map[:5]

    return similar_map

# 主函数
def find_similar_images(model, query_image, database_dir, device):

    # 处理查询图像
    query_img = preprocess_image(query_image)
    query_features = extract_features(model, query_img,device)

    # 处理数据库图像
    database_images = glob.glob(os.path.join(database_dir, "*.jpg")) + \
                      glob.glob(os.path.join(database_dir, "*.png"))

    # 查找相似图像
    similar_images = []
    for img_path in database_images:
        db_img = Image.open(img_path).convert("RGB")
        db_img_trans = preprocess_image(db_img)
        db_features = extract_features(model, db_img_trans,device)

        # 计算相似度
        similarity = cosine_similarity(query_features, db_features)

        similar_images.append([db_img, similarity])

    # 按相似度排序
    similar_images.sort(key=lambda x: x[1], reverse=True)
    similar_images = similar_images[:5]

    return similar_images





def makeMapFeature_multi(database_dir, model, device, chunk_size=500):
    """
    使用分块处理降低内存占用
    chunk_size: 每处理多少张保存一次
    """
    os.makedirs("./mapLocation", exist_ok=True)

    # 获取所有图片路径
    database_images = glob.glob(os.path.join(database_dir, "*.jpg")) + \
                      glob.glob(os.path.join(database_dir, "*.png"))

    # 初始化元数据存储文件
    metadata_file = open("./mapLocation/metadata.jsonl", "w")

    # 处理图片并分块保存特征
    all_chunks = []  # 存储各分块的文件路径
    chunk_features = []
    processed_count = 0

    for i, img_path in enumerate(tqdm(database_images)):
        # 1. 加载图片和元数据
        db_img = Image.open(img_path).convert("RGB")

        raw_exif = db_img.info.get("exif", b"").decode('utf-8', errors='ignore')  # 读取二进制EXIF
        if '{' in raw_exif and '}' in raw_exif:
            metadata_str = raw_exif[raw_exif.find('{'):raw_exif.rfind('}') + 1]
            exif_data = eval(metadata_str)
        else:
            exif_data = {}
            print("No exif")
        # 保存元数据到独立文件
        metadata_record = {
            "idx": i,
            "path": img_path,
            "metadata": raw_exif
        }
        metadata_file.write(json.dumps(metadata_record) + "\n")

        # 3. 提取特征
        db_img_trans = preprocess_image(db_img)
        with torch.no_grad():
            db_features = extract_features(model, db_img_trans, device).astype(np.float32)

        # 4. 添加到当前分块
        chunk_features.append(db_features)

        # 5. 分块保存逻辑
        if len(chunk_features) >= chunk_size or i == len(database_images) - 1:
            # 保存当前分块
            chunk_file = f"./mapLocation/features_chunk_{len(all_chunks)}.npy"
            np.save(chunk_file, np.array(chunk_features))
            all_chunks.append(chunk_file)

            # 重置临时变量
            chunk_features = []

        processed_count += 1

    # 关闭元数据文件
    metadata_file.close()

    # 6. 创建索引文件 (记录分块信息)
    index_info = {
        "total_images": processed_count,
        "chunk_files": all_chunks,
        "metadata_file": "./mapLocation/metadata.jsonl"
    }
    with open("./mapLocation/index.json", "w") as f:
        json.dump(index_info, f)

    print(f"处理完成! 共处理 {processed_count} 张图片")
    print(f"特征分块: {len(all_chunks)} 个文件")
    print(f"元数据存储在: {index_info['metadata_file']}")

def makeMapFeature(database_dir,model,device):
    database_images = glob.glob(os.path.join(database_dir, "*.jpg")) + \
                      glob.glob(os.path.join(database_dir, "*.png"))
    num = 0
    all_features_data = []
    for img_path in database_images:
        db_img = Image.open(img_path).convert("RGB")
        raw_exif = db_img.info.get("exif", b"").decode('utf-8', errors='ignore')  # 读取二进制EXIF
        if '{' in raw_exif and '}' in raw_exif:
            metadata_str = raw_exif[raw_exif.find('{'):raw_exif.rfind('}') + 1]
            exif_data = eval(metadata_str)
        else:
            exif_data = {}
            print("No exif")

        db_img_trans = preprocess_image(db_img)
        db_features = extract_features(model, db_img_trans,device)

        combined_data = {
            "feature": db_features.astype(np.float32),  # 明确数据类型
            "metadata": exif_data
        }
        all_features_data.append(combined_data)
        num+=1
        print(num)
    np.save("./mapLocation/map_feature.npy", np.array(all_features_data, dtype=object))

# 示例使用
if __name__ == "__main__":
    # 配置文件路径
    query_image = "/home/superz/Desktop/flyProject/flyProject/MixVPR/output/query_images_zzy/20_0.png"#换为你的查询图像
    database_path = "/home/superz/Desktop/data/vpair/dataSet"  # 替换为你的数据库目录
    model_file = "/home/superz/Desktop/flyProject/flyProject/MixVPR/last.ckpt"
    # 加载模型
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用 GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # 回退到 CPU
        print("CUDA not available, using CPU instead.")

    # 加载模型
    model = load_model(model_file,device)
    makeMapFeature_multi(database_path,model,device)