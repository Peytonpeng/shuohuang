# app/db.py
from psycopg2 import Error
import psycopg2
import psycopg2.extras

from analysis_service.app.extensions import logger
from analysis_service.config import DB_CONFIG


def get_db_connection():
    """获取数据库连接（从原 app.py 剪切过来）"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        return conn
    except Error as e:
        logger.error(f"数据库连接失败: {e}")
        return None
