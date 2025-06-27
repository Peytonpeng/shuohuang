# Token
# 配置密钥（用于签名和验证）
SECRET_KEY = 'shuohuangapi'

# PostgreSQL数据库连接配置
# DB_CONFIG = {
#     'host': '10.10.1.127',
#     'port': '15432',
#     'database': 'shuohuang',  # 连接到shuohuang数据库
#     'user': 'resafety',
#     'password': 'Resafety!@#$%12345postgresre'
# }

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': '5432',
    'database': 'postgres',  # 连接到shuohuang数据库
    'user': 'postgres',
    'password': '123456huxian'
}
UPLOAD_FOLDER = "uploads"  # 服务器文件存储路径