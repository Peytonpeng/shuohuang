# app/routes/train.py
import threading
import json
import datetime
import uuid

from flask import Blueprint, request, jsonify, session

from analysis_service import model_function
from analysis_service.app.utils.data import validate_param_values
from auth import token_required
from analysis_service.app.db import get_db_connection
from analysis_service.app.extensions import logger
from analysis_service.app.websocket import (
    emit_process_progress,
    emit_process_completed,
    emit_process_error,
    emit_epoch_result,
    emit_round_result,
)
import psycopg2
import psycopg2.extras

train_bp = Blueprint("train", __name__)

# 全局变量：存储活跃训练会话及其关联的线程和停止事件
active_training_processes = {}
training_sessions = {}

# --- **修改后的 run_training_with_websocket 函数：接受 stop_event 并进行清理** ---
#  对应原来 app.py 中的：def run_training_with_websocket(...):
def run_training_with_websocket(current_training_instance_id,
                                model_id_from_request,
                                sample_data_json_with_labels,
                                base_model_train_id_for_process,
                                full_param_data,
                                websocket_session_id,
                                stop_event):
    """
    带WebSocket通信的训练执行函数
    """
    # 导入model_function模块（确保已导入）
    from model_function import train_model, set_socketio_instance, set_emit_functions

    model_definition_id = base_model_train_id_for_process
    conn = get_db_connection()
    if conn is None:
        emit_process_error(websocket_session_id,'training', "数据库连接失败")
        # **如果数据库连接失败，也要清理 active_training_processes 中的条目**
        if current_training_instance_id in active_training_processes:
            del active_training_processes[current_training_instance_id]
        return

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    create_user_process = "system"
    actual_model_name_for_training = "Unknown"

    try:
        cursor.execute("SELECT model_name FROM tb_analysis_model WHERE model_id = %s", (model_id_from_request,))
        model_info_result = cursor.fetchone()
        if model_info_result is None:
            emit_process_error(websocket_session_id,'training', f"模型ID {model_id_from_request} 未找到")
            return
        actual_model_name_for_training = model_info_result["model_name"]

        if websocket_session_id in training_sessions:
            training_sessions[websocket_session_id].update({
                "status": "training",
                "model_name": actual_model_name_for_training
            })

        emit_process_progress(websocket_session_id, 'training',{
            "status": "started",
            "message": f"开始 {actual_model_name_for_training} 模型训练",
            "model_definition_id": model_definition_id
        })

        logger.debug(
            f"开始 {actual_model_name_for_training} 模型训练 (训练实例ID: {current_training_instance_id}, 定义 ID: {model_definition_id})")

        # **调用 train_model 函数，传递 stop_event**
        # 新增model_path
        trained_model_instance, result_dict_from_train, model_path = model_function.train_model(
            json_data=sample_data_json_with_labels,
            label_column="label",
            model_name=actual_model_name_for_training,
            param_data=full_param_data,
            training_id=websocket_session_id,
            stop_event=stop_event # **传递 stop_event**
        )

        #将模型路径保存进数据表tb_analysis_model_train，model_artifact_path字段
        try:
            cursor.execute(
                "UPDATE tb_analysis_model_train "
                "SET model_artifact_path = %s "
                "WHERE model_train_id = %s",
                (model_path, current_training_instance_id)
            )
            conn.commit()
            print(f"成功更新模型路径: {model_path}")
        except Exception as e:
            print(f"更新模型路径失败: {e}")
            conn.rollback()


        # 检查训练结果，首先判断是否是用户中止
        is_stopped_by_user = stop_event.is_set()
        is_error = 'error' in result_dict_from_train or '失败' in result_dict_from_train.get('message', '').lower()

        if is_stopped_by_user:
            message = "训练已中止"
            logger.debug(f"训练 {current_training_instance_id} 已由用户中止。")
            emit_process_completed(websocket_session_id, 'training', {
                "message": message,
                "status": "stopped",
                "end_time": datetime.datetime.now().isoformat()
            })
            # 更新训练会话状态为"中止"
            if websocket_session_id in training_sessions:
                training_sessions[websocket_session_id].update({
                    "status": "stopped",
                    "end_time": datetime.datetime.now().isoformat(),
                    "progress": training_sessions[websocket_session_id].get("progress", 0) # 保留最后进度
                })
        elif is_error:
            error_msg = result_dict_from_train.get('message', '训练期间发生未知错误')
            logger.error(f"训练失败: {error_msg}")
            emit_process_error(websocket_session_id,'training', error_msg)

            if websocket_session_id in training_sessions:
                training_sessions[websocket_session_id].update({
                    "status": "error",
                    "error": error_msg,
                    "end_time": datetime.datetime.now().isoformat()
                })
            conn.rollback()
        else:
            process_record_id_val = current_training_instance_id
            happen_time_process = datetime.datetime.now()
            try:
                process_data_json_to_db = json.dumps(result_dict_from_train)
            except TypeError as e:
                logger.error(f"错误: 训练结果无法序列化: {e}", exc_info=True)
                process_data_json_to_db = json.dumps({"error": f"结果序列化失败: {e}"})

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

            end_time = datetime.datetime.now()
            emit_process_completed(websocket_session_id,'training', {
                "message": f"模型训练过程结束 (ID: {current_training_instance_id})",
                "end_time": end_time.isoformat(),
                "results": result_dict_from_train
            })

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
        emit_process_error(websocket_session_id,'training', error_msg)

        if websocket_session_id in training_sessions:
            training_sessions[websocket_session_id].update({
                "status": "error",
                "error": error_msg,
                "end_time": datetime.datetime.now().isoformat()
            })

        if conn:
            conn.rollback()

    finally:
        # **无论训练结果如何 (成功/失败/中止)，都要清理 active_training_processes 中的条目**
        if current_training_instance_id in active_training_processes:
            del active_training_processes[current_training_instance_id]

        if cursor and not cursor.closed:
            cursor.close()
        if conn and not conn.closed:
            conn.close()
        logger.debug(f"训练流程结束 (ID: {current_training_instance_id}).")

#  对应原来 @app.route('/api/analysis/train/train/sample', methods=['GET'])
@train_bp.route('/train/sample', methods=['GET'])
@token_required
def get_training_sample():
    """
    对应原来：
    @app.route('/train/sample', methods=['GET'])
    ...
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  # 使用 DictCursor
    try:
        query = """
            SELECT 
                from_sample_id AS sample_id, 
                feature_extract AS sample_name, 
                from_sample_type AS sample_type 
            FROM tb_analysis_sample_feature
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            return jsonify({"state": 404, "data": []})

        for result in results:
            try:
                result['sample_type'] = int(result['sample_type'])
            except (ValueError, TypeError):
                result['sample_type'] = None

        return jsonify({"state": 200, "data": results})

    except psycopg2.Error as e:
        return jsonify({"state": 500, "message": f"数据库查询失败: {str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()

# 对应原来 @app.route('/api/analysis/train/train/model', methods=['GET'])
@train_bp.route('/train/model', methods=['GET'])
@token_required
def get_model_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500  # 修改状态码

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  # 使用 DictCursor
    query = "SELECT model_id, model_name FROM tb_analysis_model order by model_id"

    try:
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            return jsonify({"state": 404, "data": []})

        return jsonify({"state": 200, "data": results})  # 无需格式化结果

    except psycopg2.Error as e:  # 捕获psycopg2的异常
        print(f"Error while executing query: {e}")
        return jsonify({"state": 500, "message": f"数据库查询失败: {str(e)}"}), 500  # 修改状态码

    finally:
        cursor.close()
        conn.close()

# 对应原来 @app.route('/api/analysis/train/train/start', methods=['POST'])
@train_bp.route('/train/start', methods=['POST'])
@token_required
def start_training():
    data = request.get_json()
    if not data:
        return jsonify({"state": 400, "message": "无效的输入数据"}), 400

    room_id = session.get("room_id")
    if not room_id:
        # 如果 room_id 不在 session 中，则回退，或者您可以要求它在请求中
        room_id = data.get("room_id") # 尝试从请求体中获取
        if not room_id:
            return jsonify({"state": 400, "message": "请求的JSON数据中必须包含 'room_id' 字段作为 WebSocket 房间ID"}), 400
        session['room_id'] = room_id # 将其设置到 session 中以保持一致性


    model_id_req = data.get("model_id", "")
    param_data_req = data.get("param_data", {})
    sample_data_input_list = data.get("sample_data", [])
    create_user_req = "system"
    create_time_req = datetime.datetime.now()

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # model_train_id字段已删除，改为model_id
    try:
        cursor.execute("SELECT model_id, model_name FROM tb_analysis_model WHERE model_id = %s", (model_id_req,))
        model_meta_result = cursor.fetchone()
        if not model_meta_result:
            cursor.close()
            conn.close()
            return jsonify({"state": 404, "message": f"模型ID '{model_id_req}' 未在 tb_analysis_model 中找到"}), 404

        model_definition_id_from_tb_model = model_meta_result["model_id"]
        actual_model_name = model_meta_result["model_name"]

        from_sample_ids_list = [item["from_sample_id"] for item in sample_data_input_list if "from_sample_id" in item]
        if not from_sample_ids_list:
            cursor.close()
            conn.close()
            return jsonify({"state": 400, "message": "输入样本列表中未提供 'from_sample_id'"}), 400

        placeholders = ', '.join(['%s'] * len(from_sample_ids_list))
        query_features = f"SELECT from_sample_id, feature_sample_data, feature_extract FROM tb_analysis_sample_feature WHERE from_sample_id IN ({placeholders})"
        cursor.execute(query_features, tuple(from_sample_ids_list))
        db_feature_samples_rows = cursor.fetchall()

        if not db_feature_samples_rows:
            cursor.close()
            conn.close()
            return jsonify({"state": 404, "message": "未找到任何请求样本的特征数据"}), 404

        feature_data_map = {
            row["from_sample_id"]: {
                "data_str": row["feature_sample_data"],
                "method": row["feature_extract"]
            } for row in db_feature_samples_rows
        }

        processed_feature_list_for_training = []
        unique_from_sample_ids = sorted(list(set(from_sample_ids_list)))
        label_map = {sample_id: i for i, sample_id in enumerate(unique_from_sample_ids)}

        for requested_sample_id in unique_from_sample_ids:
            if requested_sample_id in feature_data_map:
                feature_info = feature_data_map[requested_sample_id]
                feature_json_str = feature_info["data_str"]
                feature_method_from_db = feature_info["method"]
                assigned_label = label_map[requested_sample_id]

                try:
                    raw_sample_data = json.loads(feature_json_str)
                    processed_feature_list_for_training.append({
                        "raw_data": raw_sample_data.get("data", raw_sample_data),
                        "label": assigned_label,
                        "method": feature_method_from_db
                    })
                except json.JSONDecodeError:
                    logger.warning(f"警告: 样本 {requested_sample_id} 的特征数据JSON解码失败，已跳过。")
                except Exception as e:
                    logger.warning(f"警告: 处理样本 {requested_sample_id} 时发生错误: {e}，已跳过。")
            else:
                logger.warning(f"警告: 请求的样本ID {requested_sample_id} 未在数据库查询结果中找到特征数据。")

        if not processed_feature_list_for_training:
            cursor.close()
            conn.close()
            return jsonify({"state": 404, "message": "没有样本被成功处理以用于训练。"}), 404

        num_unique_labels_generated = len(label_map)
        classification_model_names = ['深度神经网络', '逻辑回归', '随机森林', '支持向量机']
        if actual_model_name in classification_model_names:
            if num_unique_labels_generated > 0:
                if param_data_req.get("num_classes") != num_unique_labels_generated:
                    logger.info(
                        f"信息: 模型 '{actual_model_name}' 的 num_classes 从请求的 {param_data_req.get('num_classes')} 更新为自动生成的标签数量 {num_unique_labels_generated}。")
                param_data_req["num_classes"] = num_unique_labels_generated

        final_feature_data_json_for_training = json.dumps(processed_feature_list_for_training)
        # **使用 UUID 为每次训练生成唯一的实例 ID
        current_training_instance_id = str(uuid.uuid4())
        param_data_req_json = json.dumps(param_data_req)

        #修改的存储逻辑
        model_train_save_name = f"XX对象_{actual_model_name}_{num_unique_labels_generated}分类"

        # model_train_data不存进数据库了，没有用，浪费资源
        query_insert_train_instance = """
        INSERT INTO tb_analysis_model_train
        (model_train_id, model_id, model_train_name, param_data, param_auto_perfect, create_user, create_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query_insert_train_instance, (
            current_training_instance_id, model_id_req,  model_train_save_name,
            param_data_req_json, str(data.get("param_auto_perfect", "")),
            create_user_req, create_time_req
        ))

        # query_insert_sample_link = """
        # INSERT INTO tb_analysis_model_train_sample
        # (model_train_id, from_sample_id, from_sample_type, create_user, create_time)
        # VALUES (%s, %s, %s, %s, %s)
        # """
        # for item in sample_data_input_list:
        #     individual_sample_id_in_request = item.get("from_sample_id")
        #     sample_type_in_request = item.get("from_sample_type")
        #     if individual_sample_id_in_request:
        #         cursor.execute(query_insert_sample_link, (
        #             current_training_instance_id, individual_sample_id_in_request,
        #             sample_type_in_request, create_user_req, create_time_req
        #         ))

        query_insert_sample_link = """
        INSERT INTO tb_analysis_model_train_sample
        (model_train_id, from_sample_id, from_sample_type, create_user, create_time)
        VALUES (%s, %s, %s, %s, %s)
        """

        # 准备批量插入的数据列表
        batch_insert_data = []
        for item in sample_data_input_list:
            individual_sample_id_in_request = item.get("from_sample_id")
            sample_type_in_request = item.get("from_sample_type")
            if individual_sample_id_in_request:  # 只添加有效样本ID的数据
                batch_insert_data.append((
                    current_training_instance_id,
                    individual_sample_id_in_request,
                    sample_type_in_request,
                    create_user_req,
                    create_time_req
                ))

        # 执行批量插入（仅当有数据时）
        if batch_insert_data:
            cursor.executemany(query_insert_sample_link, batch_insert_data)

        conn.commit()

        # 生成WebSocket会话ID
        websocket_session_id = room_id

        stop_event = threading.Event()
        # **存储线程对象和停止事件，以便后续通过 ID 访问**
        # 初始时线程对象可以为 None，启动后更新
        active_training_processes[current_training_instance_id] = {
            "thread": None,
            "stop_event": stop_event
        }

        # 存储训练会话信息 (便于前端查询状态)
        training_sessions[websocket_session_id] = {
            "training_instance_id": current_training_instance_id,
            "model_name": actual_model_name,
            "status": "starting",
            "start_time": datetime.datetime.now().isoformat(),
            "progress": 0
        }

        # 在后台线程中启动训练，并传递停止事件
        training_thread = threading.Thread(
            target=run_training_with_websocket,
            args=(
                current_training_instance_id,
                model_id_req,
                final_feature_data_json_for_training,
                model_definition_id_from_tb_model,
                param_data_req,
                websocket_session_id,
                stop_event # **传递停止事件**
            )
        )
        training_thread.daemon = True
        training_thread.start()

        # **将实际的线程对象存储到 active_training_processes 中**
        active_training_processes[current_training_instance_id]["thread"] = training_thread

        return jsonify({
            "state": 200,
            "data": {
                "success": "true",
                "message": "训练已启动",
                "training_id": current_training_instance_id, # **返回这个 ID 给前端**
                "websocket": {
                    "session_id": websocket_session_id, # **这里使用变量，而不是硬编码字符串**
                    "namespace": "/ns_analysis",
                    "events": {
                        "progress": "training_progress",
                        "epoch_result": "epoch_result",
                        "round_result": "round_result",
                        "completed": "training_completed",
                        "error": "training_error"
                    }
                }
            }
        }), 200

    except psycopg2.Error as e:
        if conn: conn.rollback()
        logger.error(f"数据库操作错误: {e}")
        return jsonify({"state": 500, "message": f"数据库操作失败: {str(e)}"}), 500
    except json.JSONDecodeError as e:
        logger.error(f"请求JSON解析错误: {e}")
        return jsonify({"state": 400, "message": f"请求的JSON数据格式无效: {str(e)}"}), 400
    except ValueError as ve:
        logger.error(f"数据处理错误: {ve}")
        return jsonify({"state": 400, "message": f"数据处理错误: {str(ve)}"}), 400
    except Exception as e:
        if conn: conn.rollback()
        import traceback
        logger.error(f"发生意外错误: {e}\n{traceback.format_exc()}")
        return jsonify({"state": 500, "message": f"发生意外错误: {str(e)}"}), 500
    finally:
        if 'cursor' in locals() and cursor and not cursor.closed:
            cursor.close()
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()

# 对应原来 @app.route('/api/analysis/train/train/cancel', methods=['POST'])
@train_bp.route('/train/cancel', methods=['POST'])
@token_required
def cancel_training():
    data = request.get_json()
    if not data:
        return jsonify({"state": 400, "message": "无效的输入数据"}), 400

    training_instance_id = data.get("training_id")
    if not training_instance_id:
        return jsonify({"state": 400, "message": "请求的JSON数据中必须包含 'training_id' 字段"}), 400

    process_info = active_training_processes.get(training_instance_id)

    if not process_info:
        return jsonify({"state": 404, "message": f"未找到 ID 为 '{training_instance_id}' 的活跃训练进程，可能已完成或不存在。"}), 404

    stop_event = process_info["stop_event"]
    training_thread = process_info["thread"]

    if training_thread and training_thread.is_alive():
        logger.debug(f"收到停止训练请求，训练ID: {training_instance_id}")
        stop_event.set()  # 设置停止事件，向线程发出停止信号
        logger.debug(f"停止事件已设置: {stop_event.is_set()}")

        # 更新训练状态
        # 通过遍历找到正确的 session
        session_to_update = None
        for session_data in training_sessions.values():
            if session_data.get("training_instance_id") == training_instance_id:
                session_to_update = session_data
                break

        if session_to_update:
            session_to_update.update({
                "status": "stopping",
                "message": "正在中止训练..."
            })

        return jsonify({
            "state": 200,
            "data": {
                "success": True,
                "message": f"已向训练实例 '{training_instance_id}' 发出中止信号，训练将在当前操作完成后中止。"
            }
        }), 200
    else:
        # 线程可能已经完成或因其他原因停止
        if training_instance_id in active_training_processes:
            del active_training_processes[training_instance_id]
        return jsonify({
            "state": 409,
            "message": f"训练实例 '{training_instance_id}' 未在运行或已完成。"
        }), 409

# 对应原来 @app.route('/api/analysis/train/train/model/save', methods=['POST'])
@train_bp.route('/train/model/save', methods=['POST'])
@token_required
def save_model(): # 函数名保持不变
    """
    根据 model_train_id 更新 tb_analysis_model_train 表中的 model_train_name。
    请求体中需要包含 'model_train_id' 和 'model_train_name'。
    """
    data = request.get_json()
    if not data:
        return jsonify({"state": 400, "message": "无效的输入数据：请求体为空"}), 400

    model_train_id = data.get("model_train_id")
    new_model_train_name = data.get("model_train_name") # 使用请求中的 model_train_name 作为新名称

    if not model_train_id:
        return jsonify({"state": 400, "message": "无效的输入数据：'model_train_id' 是必需的"}), 400
    if not new_model_train_name:
        return jsonify({"state": 400, "message": "无效的输入数据：'model_train_name' (新的模型名称) 是必需的"}), 400

    create_user = "system" # 或者如果适用，从token中动态获取
    create_time = datetime.datetime.now()

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # 1. 检查模型训练记录是否存在
        cursor.execute("SELECT model_train_id FROM tb_analysis_model_train WHERE model_train_id = %s",
                       (model_train_id,))
        existing_record = cursor.fetchone()
        if not existing_record:
            return jsonify({"state": 404, "message": f"未找到 ID 为 '{model_train_id}' 的模型训练记录。"}), 404

        # 2. 更新 tb_analysis_model_train 表中的 model_train_name
        query_update = """
        UPDATE tb_analysis_model_train
        SET model_train_name = %s, create_user = %s, create_time = %s
        WHERE model_train_id = %s
        """
        cursor.execute(query_update,
                       (new_model_train_name, create_user, create_time, model_train_id))

        conn.commit()
        return jsonify({"state": 200, "data": {"success": "true", "message": "模型训练名称更新成功。"}}), 200
        return jsonify({"state": 200, "data": {"success": "true", "message": "Model saved successfully"}}), 200

    except psycopg2.Error as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({"state": 500, "message": "Failed to save model"}), 500

    finally:
        cursor.close()
        conn.close()

# 对应原来 @app.route('/api/analysis/train/train/list', methods=['GET'])
@train_bp.route('/train/list', methods=['GET'])
@token_required
def list_model_trains():
    """
    返回示例：
    {
        "state": 200,
        "data": [
            {"model_train_id": "uuid-1", "model_train_name": "模型A"},
            {"model_train_id": "uuid-2", "model_train_name": "模型B"}
        ]
    }
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute("""
            SELECT model_train_id, model_train_name
            FROM tb_analysis_model_train
            ORDER BY create_time DESC
        """)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]

        return jsonify({"state": 200, "data": result}), 200

    except Exception as e:
        logger.error(f"查询模型ID和名称失败: {e}")
        return jsonify({"state": 500, "message": "查询失败"}), 500
    finally:
        cursor.close()
        conn.close()

# 对应原来 @app.route('/api/analysis/train/train/delete', methods=['POST'])
@train_bp.route('/train/delete', methods=['POST'])
@token_required
def delete_model_train():
    """
    删除指定的模型训练记录。
    请求体：
    {
        "model_train_ids": ["uuid1", "uuid2", ...]
    }
    返回：
    {
        "state": 200,
        "data": {
            "deleted_ids": [...],
            "not_found_ids": [...]
        }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"state": 400, "message": "请求体为空"}), 400

    ids = data.get("model_train_ids")
    if not ids or not isinstance(ids, list):
        return jsonify({"state": 400, "message": "'model_train_ids' 必须是非空数组"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        # 先查出存在的 id
        placeholders = ', '.join(['%s'] * len(ids))
        cursor.execute(f"SELECT model_train_id FROM tb_analysis_model_train WHERE model_train_id IN ({placeholders})", tuple(ids))
        existing_rows = cursor.fetchall()
        existing_ids = [r["model_train_id"] for r in existing_rows]

        if not existing_ids:
            return jsonify({"state": 404, "message": "未找到任何匹配的模型ID"}), 404

        # 删除这些 id
        placeholders = ', '.join(['%s'] * len(existing_ids))
        cursor.execute(f"DELETE FROM tb_analysis_model_train WHERE model_train_id IN ({placeholders})", tuple(existing_ids))
        conn.commit()

        # 计算未找到的 id
        not_found_ids = [i for i in ids if i not in existing_ids]

        return jsonify({
            "state": 200,
            "data": {
                "deleted_ids": existing_ids,
                "not_found_ids": not_found_ids
            }
        }), 200

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"删除模型失败: {e}")
        return jsonify({"state": 500, "message": "删除模型失败"}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 对应原来 @app.route('/api/analysis/train/train/param/get', methods=['GET'])
@train_bp.route('/train/param/get', methods=['GET'])
@token_required
def get_train_param():
    #1.拿到model id  查数据库  校验
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        train_model_id = request.args.get('model_train_id')
        if not train_model_id:
            return jsonify({"state": 500, "message": "传参为空"})

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT param_config FROM tb_analysis_model WHERE model_id = %s",
            (train_model_id,)
        )
        param = cursor.fetchall()
        #要校验不为空不
        return jsonify({"state": 200, "data": param}), 200
    except Exception as e:
        # 捕获任何服务器内部错误
        logger.error(f"获取模型数据失败: {str(e)}")  # 打印错误以便调试
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# 对应原来 @app.route('/api/analysis/train/train/param/save', methods=['POST'])
@train_bp.route('/train/param/save', methods=['POST'])
@token_required
def save_train_model_param():
    conn = None
    cursor = None
    try:
        data = request.get_json()

        model_id = data.get('model_id')
        param = data.get('params')

        # 参数校验
        if not model_id or not param:
            return jsonify({"state": 400, "message": "缺少必要参数: model_id 或 param"}), 400

        # JSON 解析
        try:
            # 如果 param 已经是列表或字典，直接使用；如果是字符串，则解析
            param_data = json.loads(param) if isinstance(param, str) else param
        except json.JSONDecodeError as e:
            return jsonify({"state": 400, "message": f"JSON 解析失败: {str(e)}"}), 400

        # 参数格式校验
        is_valid, errors = validate_param_values(param_data)
        if not is_valid:
            return jsonify({
                "state": 400,
                "message": "参数校验失败",
                "errors": errors
            }), 400

        # 将Python对象转换为JSON字符串，用于数据库存储和返回
        param_string = json.dumps(param_data, ensure_ascii=False)

        # 数据库操作
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # 检查 model_id 是否存在
        cursor.execute("SELECT COUNT(*) FROM tb_analysis_model WHERE model_id = %s", (model_id,))
        exists = cursor.fetchone()[0]

        if exists:
            # 存在，执行更新
            cursor.execute(
                "UPDATE tb_analysis_model SET param_config = %s WHERE model_id = %s",
                (param_string, model_id)
            )
            conn.commit()
        else:
            # 不存在，提示或处理
            return jsonify({
                "state": 403,
                "data": {
                    "success": "false",
                    "message": "参数上传失败，上传的model_id不存在"
                }
            })

        return jsonify({
            "state": 200,
            "data": {
                "success": "true",
                "message": "参数上传成功"
            }
        })

    except Exception as e:
        logger.error(f"保存模型参数失败: {str(e)}")
        # 避免将底层错误信息直接暴露给客户端
        return jsonify({"state": 500, "message": "服务器内部错误"}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
