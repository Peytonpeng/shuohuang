# app/extensions.py
import logging
from flask_socketio import SocketIO
from flask_compress import Compress

# SocketIO 实例（原来 app.py 顶部的 socketio）
socketio = SocketIO(cors_allowed_origins="*")
compress = Compress()

logger = logging.getLogger(__name__)

def init_extensions(app):
    """
    在 create_app 中调用：初始化所有扩展
    """
    # 绑定到 app
    socketio.init_app(app)
    compress.init_app(app)

    # 日志与 app 同步
    global logger
    logger = app.logger

    return app
