from sh_analysis.main.Config.getConnection import get_db_connection
from sh_analysis.main.model.model_function import train_model, set_socketio_instance, set_emit_functions
import datetime
import psycopg2
import psycopg2.extras
import json


def run_training_with_websocket(training_sessions,current_training_instance_id, model_id_from_request, sample_data_json_with_labels,
                                base_model_train_id_for_process, full_param_data, websocket_session_id):
    from sh_analysis.main.Main import socketio
    from sh_analysis.main.Main import logger

    """
    带WebSocket通信的训练执行函数
    """

    model_definition_id = base_model_train_id_for_process
    conn = get_db_connection()
    if conn is None:
        emit_training_error(websocket_session_id, "数据库连接失败")
        return

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    create_user_process = "system"
    actual_model_name_for_training = "Unknown"

    try:
        # 获取模型名称
        cursor.execute("SELECT model_name FROM tb_analysis_model WHERE model_id = %s", (model_id_from_request,))
        model_info_result = cursor.fetchone()
        if model_info_result is None:
            emit_training_error(websocket_session_id, f"模型ID {model_id_from_request} 未找到")
            return
        actual_model_name_for_training = model_info_result["model_name"]

        # 更新训练会话状态
        if websocket_session_id in training_sessions:
            training_sessions[websocket_session_id].update({
                "status": "training",
                "model_name": actual_model_name_for_training
            })

        # 设置 model_function 模块的 socketio 实例和 emit 函数
        set_socketio_instance(socketio)
        print("training_progress:", emit_training_progress)
        print("epoch_result:", emit_epoch_result)
        print("round_result:", emit_round_result)
        print("training_completed:", emit_training_completed)
        print("training_error:", emit_training_error)
        emit_functions_dict = {
            "training_progress": emit_training_progress,
            "epoch_result": emit_epoch_result,
            "round_result": emit_round_result,
            "training_completed": emit_training_completed,
            "training_error": emit_training_error
        }

        set_emit_functions(emit_functions_dict)

        # 发送训练开始消息
        emit_training_progress(websocket_session_id, {
            "status": "started",
            "message": f"开始 {actual_model_name_for_training} 模型训练",
            "model_definition_id": model_definition_id
        })

        logger.info(
            f"开始 {actual_model_name_for_training} 模型训练 (训练实例ID: {current_training_instance_id}, 定义 ID: {model_definition_id})")

        # --- 调用 train_model 函数，修正参数名 ---
        trained_model_instance, result_dict_from_train = train_model(
            json_data=sample_data_json_with_labels,
            label_column="label",
            model_name=actual_model_name_for_training,
            param_data=full_param_data,
            training_id=websocket_session_id  # 修正：使用正确的参数名
        )

        # 检查训练结果
        is_error = 'error' in result_dict_from_train or '失败' in result_dict_from_train.get('message', '').lower()

        if is_error:
            error_msg = result_dict_from_train.get('message', '训练期间发生未知错误')
            logger.error(f"训练失败: {error_msg}")
            emit_training_error(websocket_session_id, error_msg)

            # 更新训练会话状态为错误
            if websocket_session_id in training_sessions:
                training_sessions[websocket_session_id].update({
                    "status": "error",
                    "error": error_msg,
                    "end_time": datetime.datetime.now().isoformat()
                })
            conn.rollback()
        else:
            # 训练成功，保存结果到数据库
            process_record_id_val = current_training_instance_id
            happen_time_process = datetime.datetime.now()
            try:
                process_data_json_to_db = json.dumps(result_dict_from_train)
            except TypeError as e:
                logger.error(f"错误: 训练结果无法序列化: {e}", exc_info=True)
                process_data_json_to_db = json.dumps({"error": f"结果序列化失败: {e}"})

            # 插入或更新训练结果
            query_insert_train_process = """
            INSERT INTO tb_analysis_model_train_process
            (model_train_id, happen_time, process_data, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (model_train_id) DO UPDATE SET
            happen_time = EXCLUDED.happen_time,
            process_data = EXCLUDED.process_data,
            create_user = EXCLUDED.create_user,
            create_time = EXCLUDED.create_time
            """
            cursor.execute(query_insert_train_process, (
                process_record_id_val, happen_time_process, process_data_json_to_db,
                create_user_process, happen_time_process
            ))
            conn.commit()
            logger.info(f"最终训练结果已保存 (ID: {process_record_id_val})。")

            # 发送训练完成消息
            end_time = datetime.datetime.now()
            emit_training_completed(websocket_session_id, {
                "message": f"模型训练过程结束 (ID: {current_training_instance_id})",
                "end_time": end_time.isoformat(),
                "results": result_dict_from_train
            })

            # 更新训练会话状态为完成
            if websocket_session_id in training_sessions:
                training_sessions[websocket_session_id].update({
                    "status": "completed",
                    "end_time": end_time.isoformat(),
                    "progress": 100,
                    "results": result_dict_from_train
                })

    except Exception as e:
        import traceback
        error_msg = f"run_training_with_websocket 发生意外错误: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        emit_training_error(websocket_session_id, error_msg)

        # 更新训练会话状态为错误
        if websocket_session_id in training_sessions:
            training_sessions[websocket_session_id].update({
                "status": "error",
                "error": error_msg,
                "end_time": datetime.datetime.now().isoformat()
            })

        if conn:
            conn.rollback()

    finally:
        # 清理资源
        if cursor and not cursor.closed:
            cursor.close()
        if conn and not conn.closed:
            conn.close()
        logger.info(f"训练流程结束 (ID: {current_training_instance_id}).")


def emit_training_progress(session_id, data):
    from shuohuang_api.main.Main import socketio
    from shuohuang_api.main.Main import logger

    """发送训练进度消息"""
    try:
        payload = {
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('training_progress', payload, namespace='/training', room=session_id)
        logger.debug(f"发送训练进度消息到房间 {session_id}: {data.get('message', '')}")
    except Exception as e:
        logger.error(f"发送训练进度消息失败: {e}", exc_info=True)


def emit_epoch_result(session_id, data):
    from shuohuang_api.main.Main import socketio
    from shuohuang_api.main.Main import logger

    """发送单个epoch结果"""
    try:
        payload = {
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        print(f"Epoch result payload: {payload}")
        socketio.emit('epoch_result', payload, namespace='/training', room=session_id)
        logger.debug(f"发送epoch结果到房间 {session_id}: epoch {data.get('global_epoch', '')}")
    except Exception as e:
        logger.error(f"发送epoch结果消息失败: {e}", exc_info=True)


def emit_round_result(session_id, data):
    from shuohuang_api.main.Main import socketio
    from shuohuang_api.main.Main import logger

    """发送轮次结果"""
    try:
        payload = {
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('round_result', payload, namespace='/training', room=session_id)
        logger.debug(f"发送轮次结果到房间 {session_id}: round {data.get('current_round', '')}")
    except Exception as e:
        logger.error(f"发送轮次结果消息失败: {e}", exc_info=True)


def emit_training_completed(session_id, data):
    from shuohuang_api.main.Main import socketio
    from shuohuang_api.main.Main import logger

    """发送训练完成消息"""
    try:
        payload = {
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('training_completed', payload, namespace='/training', room=session_id)
        logger.info(f"发送训练完成消息到房间 {session_id}")
    except Exception as e:
        logger.error(f"发送训练完成消息失败: {e}", exc_info=True)


def emit_training_error(session_id, error_msg, details=None):
    from shuohuang_api.main.Main import socketio
    from shuohuang_api.main.Main import logger

    """发送训练错误消息"""
    try:
        payload = {
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(error_msg)
        }
        if details:
            payload['details'] = str(details)

        socketio.emit('training_error', payload, namespace='/training', room=session_id)
        logger.error(f"发送训练错误消息到房间 {session_id}: {error_msg}")
    except Exception as e:
        logger.error(f"发送训练错误消息失败: {e}", exc_info=True)