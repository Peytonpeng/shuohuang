import sh_analysis.main.Config.DBConfig as db
import psycopg2
from psycopg2 import Error

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(
            host=db['host'],
            port=db['port'],
            database=db['database'],
            user=db['user'],
            password=db['password']
        )
        return conn
    except Error as e:
        print(f"数据库连接失败: {e}")
        return None
