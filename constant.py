# Token
# 配置密钥（用于签名和验证）
SECRET_KEY = 'shuohuang'

# PostgreSQL数据库连接配置
DB_CONFIG = {
    'host': '10.10.1.127',
    'port': '15432',
    'database': 'shuohuang',  # 连接到shuohuang数据库
    'user': 'resafety',
    'password': 'Resafety!@#$%12345postgresre'
}

UPLOAD_FOLDER = "uploads"  # 服务器文件存储路径