# wsgi.py
from analysis_service.app import create_app
from analysis_service.app.extensions import socketio
from analysis_service.app.websocket import (
    emit_process_progress,
    emit_process_completed,
    emit_process_error,
    emit_process_result,
    emit_epoch_result,
    emit_round_result,
)
from model_function import set_socketio_instance, set_emit_functions

# 创建应用实例
app = create_app()

# 设置全局 SocketIO 实例和 emit 函数
set_socketio_instance(socketio)
emit_funcs = {
    'training_progress': emit_process_progress,
    'epoch_result':      emit_epoch_result,
    'round_result':      emit_round_result,
    'process_result':    emit_process_result,
    'training_completed': emit_process_completed,
    'training_error':     emit_process_error,
}
set_emit_functions(emit_funcs)

# ========== 新增：确保 app 和 socketio 正确绑定 ==========
# 如果 create_app() 没有自动初始化 socketio，需要手动 init
# （根据你的项目结构，可能已处理，此处为保险）
socketio.init_app(app)

# ========== 主程序入口 ==========
if __name__ == '__main__':
    print("🚀 开始启动 Flask + Socket.IO 服务...")
    print("📡 监听地址: http://0.0.0.0:5001")
    socketio.run(
        app,
        host='0.0.0.0',
        port=5001,
        debug=True,
        allow_unsafe_werkzeug=True,  # 允许在非开发环境使用 Werkzeug（仅本地）
        use_reloader=False          # 启用代码修改自动重载
    )