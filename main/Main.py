from flask import Flask, jsonify
from flask_socketio import SocketIO
import routes.select as select_routes
import routes.preprocess as pocesses_routes
import routes.feature as feature_routes
import routes.train as train_routes
import routes.apply as apply_routes
import logging
from sh_analysis.main.model.model_function import set_socketio_instance

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins='*')  # 允许跨域
set_socketio_instance(socketio)

# 注册路由模块
app.register_blueprint(select_routes.select)
app.register_blueprint(pocesses_routes.preProcess)
app.register_blueprint(feature_routes.feature)
app.register_blueprint(train_routes.train)
app.register_blueprint(apply_routes.apply)

# 配置日志 (确保在所有 logger 使用之前)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # 确保 logger 在这里被定义


if __name__ == '__main__':
    app.run(debug=True)