# app/__init__.py
import logging
import os
from flask import Flask, render_template

from analysis_service.config import SECRET_KEY, UPLOAD_FOLDER
from analysis_service.app.extensions import init_extensions, logger as base_logger
from analysis_service.app.websocket import register_websocket_handlers
from analysis_service.app.routes import register_blueprints

def configure_logging(app):
    """配置日志系统（直接搬你原来的）"""
    if not app.debug:
        # 生产环境 - 使用gunicorn日志
        gunicorn_logger = logging.getLogger('gunicorn.error')
        handler = logging.FileHandler('error.log')
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        gunicorn_logger.addHandler(handler)
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    else:
        # 开发环境 - 控制台输出
        app.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)

    # 确保logger在所有地方可用
    global base_logger
    base_logger = app.logger


def create_app():
    app = Flask(__name__)
    app.secret_key = SECRET_KEY

    # 上传目录
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # 初始化扩展（socketio / compress）
    init_extensions(app)

    # 日志
    configure_logging(app)

    # 注册 WebSocket 事件
    register_websocket_handlers()
    app.logger.info("WebSocket 事件处理器已注册")

    # 注册蓝图
    register_blueprints(app)


    app.logger.info("应用初始化完成")
    return app
