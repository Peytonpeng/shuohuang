import os

UPLOAD_FOLDER = "uploads"  # 服务器文件存储路径
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 创建文件夹（如果不存在）