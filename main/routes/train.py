import json
import datetime
import os
import uuid
import joblib
import torch
from flask import request, jsonify, Blueprint
import sh_analysis.main.Config.getConnection as get_db_connection
import psycopg2
import threading
from sh_analysis.main.utils.websocket_utils import run_training_with_websocket

train = Blueprint('train', __name__)

training_sessions = {}

@train.route('/api/analysis/train/train/sample', methods=['GET'])
def get_feature_sample_data():
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


@train.route('/api/analysis/train/train/model', methods=['GET'])
def get_model_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500  # 修改状态码

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 使用 DictCursor
    query = "SELECT model_id, model_name, model_data FROM tb_analysis_model"

    try:
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            return jsonify({"state": 404, "data": []})

        return jsonify({"state": 200, "data": results})  # 无需格式化结果

    except psycopg2.Error as e: # 捕获psycopg2的异常
        print(f"Error while executing query: {e}")
        return jsonify({"state": 500, "message": f"数据库查询失败: {str(e)}"}), 500 # 修改状态码

    finally:
        cursor.close()
        conn.close()


@train.route('/api/analysis/train/train/start', methods=['POST'])
def start_model_training():  # 函数名已更新
    data = request.get_json()
    if not data:
        return jsonify({"state": 400, "message": "无效的输入数据"}), 400

    model_id_req = data.get("model_id", "")
    param_data_req = data.get("param_data", {})
    sample_data_input_list = data.get("sample_data", [])
    create_user_req = "system"  # Or from authenticated user
    create_time_req = datetime.datetime.now()

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        cursor.execute("SELECT model_train_id, model_name FROM tb_analysis_model WHERE model_id = %s", (model_id_req,))
        model_meta_result = cursor.fetchone()
        if not model_meta_result:
            cursor.close()
            conn.close()
            return jsonify({"state": 404, "message": f"模型ID '{model_id_req}' 未在 tb_analysis_model 中找到"}), 404

        model_definition_id_from_tb_model = model_meta_result["model_train_id"]
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
                    from shuohuang_api.main.Main import logger
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
        current_training_instance_id = str(uuid.uuid4())
        param_data_req_json = json.dumps(param_data_req)

        query_insert_train_instance = """
        INSERT INTO tb_analysis_model_train
        (model_train_id, model_id, model_train_name, param_data, param_auto_perfect, model_train_data, create_user, create_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query_insert_train_instance, (
            current_training_instance_id, model_id_req, actual_model_name,
            param_data_req_json, str(data.get("param_auto_perfect", "")),
            final_feature_data_json_for_training,
            create_user_req, create_time_req
        ))

        query_insert_sample_link = """
        INSERT INTO tb_analysis_model_train_sample
        (model_train_id, from_sample_id, from_sample_type, create_user, create_time)
        VALUES (%s, %s, %s, %s, %s)
        """
        for item in sample_data_input_list:
            individual_sample_id_in_request = item.get("from_sample_id")
            sample_type_in_request = item.get("from_sample_type")
            if individual_sample_id_in_request:
                cursor.execute(query_insert_sample_link, (
                    current_training_instance_id, individual_sample_id_in_request,
                    sample_type_in_request, create_user_req, create_time_req
                ))

        conn.commit()

        # 生成WebSocket会话ID
        websocket_session_id = f"training_{current_training_instance_id}"

        # 存储训练会话信息
        training_sessions[websocket_session_id] = {
            "training_instance_id": current_training_instance_id,
            "model_name": actual_model_name,
            "status": "starting",
            "start_time": datetime.datetime.now().isoformat(),
            "progress": 0
        }

        # 在后台线程中启动训练
        training_thread = threading.Thread(
            target=run_training_with_websocket,
            args=(
                training_sessions,
                current_training_instance_id,  # 传递 training_instance_id
                model_id_req,
                final_feature_data_json_for_training,
                model_definition_id_from_tb_model,
                param_data_req,
                websocket_session_id
            )
        )
        training_thread.daemon = True
        training_thread.start()

        return jsonify({
            "state": 200,
            "data": {
                "success": "true",
                "message": "训练已启动",
                "training_instance_id": current_training_instance_id,
                "websocket": {
                    "session_id": websocket_session_id,
                    "namespace": "/training",
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


# 获取训练状态的REST API
@train.route('/api/analysis/train/status/<training_instance_id>', methods=['GET'])
def get_training_status(training_instance_id):
    """获取训练状态的REST接口"""
    websocket_session_id = f"training_{training_instance_id}"

    if websocket_session_id in training_sessions:
        session_info = training_sessions[websocket_session_id].copy()
        # 转换datetime对象为字符串
        for key, value in session_info.items():
            if isinstance(value, datetime.datetime):
                session_info[key] = value.isoformat()

        return jsonify({
            "state": 200,
            "data": session_info
        }), 200
    else:
        return jsonify({
            "state": 404,
            "message": "训练会话不存在"
        }), 404


@train.route('/api/analysis/train/train/model/save', methods=['POST'])
def save_model():
    data = request.get_json()
    if not data or "model_train_id" not in data:
        return jsonify({"state": 400, "message": "Invalid input data"}), 400

    model_train_id = data["model_train_id"]
    create_user = "system"
    create_time = datetime.datetime.now()

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "Database connection failed"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # **1. 查询模型信息**
        cursor.execute("SELECT model_id, model_train_name FROM tb_analysis_model_train WHERE model_train_id = %s", (model_train_id,))
        model_result = cursor.fetchone()
        if not model_result:
            return jsonify({"state": 404, "message": "Model training record not found"}), 404

        model_id = model_result["model_id"]
        model_name = model_result["model_train_name"]

        # **2. 读取训练后的模型参数**
        model_dir = "./saved_models"
        print(f"Model name: {model_name}") # 打印 model_name
        print(f"Model directory: {model_dir}") # 打印 model_dir
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name)]
        model_files.sort(reverse=True)  # 按时间降序排序
        print(f"Files in model directory: {os.listdir(model_dir)}") # 列出目录内容
        if not model_files:
            return jsonify({"state": 404, "message": "Trained model file not found"}), 404

        model_path = os.path.join(model_dir, model_files[0])
        print(f"Model path: {model_path}") # 打印 model_path
        model_data = None
        if model_path.endswith('.pt'):
            model_data = torch.load(model_path, map_location=torch.device('cpu'))
        elif model_path.endswith('.joblib'):
            with open(model_path, "rb") as f:
                model_data = joblib.load(f)

        if model_data is None:
            return jsonify({"state": 404, "message": "Trained model file not found"}), 404

        model_data_json = json.dumps(model_data, default=str)  # 确保数据可序列化

        # **3. 插入或更新 `tb_analysis_model` 表**
        cursor.execute("SELECT model_id FROM tb_analysis_model WHERE model_id = %s", (model_id,))
        existing_model = cursor.fetchone()

        if existing_model:
            query_update = """
            UPDATE tb_analysis_model
            SET model_train_id = %s, model_name = %s, model_data = %s, create_user = %s, create_time = %s
            WHERE model_id = %s
            """
            cursor.execute(query_update, (model_train_id, model_name, model_data_json, create_user, create_time, model_id))
        else:
            query_insert = """
            INSERT INTO tb_analysis_model (model_id, model_train_id, model_name, model_data, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query_insert, (model_id, model_train_id, model_name, model_data_json, create_user, create_time))

        conn.commit()
        return jsonify({"state": 200, "data": {"success": "true", "message": "Model saved successfully"}}), 200

    except psycopg2.Error as e:
        print(f"Error saving model: {e}")
        return jsonify({"state": 500, "message": "Failed to save model"}), 500

    finally:
        cursor.close()
        conn.close()


@train.route('/api/analysis/train/train/model/download', methods=['GET'])
def download_model():
    model_train_id = request.args.get("model_train_id", "")

    if not model_train_id:
        return jsonify({"state": 400, "message": "Invalid model_train_id"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "Database connection failed"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # 查询模型数据
        query = """
        SELECT model_data FROM tb_analysis_model WHERE model_train_id = %s
        """
        cursor.execute(query, (model_train_id,))
        model_result = cursor.fetchone()

        if not model_result:
            return jsonify({"state": 404, "message": "Model not found"}), 404

        return jsonify({"state": 200, "data": {"model_train_data": model_result["model_data"]}}), 200

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return jsonify({"state": 500, "message": "Failed to fetch model data"}), 500

    finally:
        cursor.close()
        conn.close()


@train.route('/api/analysis/train/train/param/get', methods=['GET'])
def get_param():
    model_id = request.args.get("model_id", "")
    if not model_id:
        return jsonify({"state": 400, "message": "Invalid model_id"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "Database connection failed"}), 500

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute("SELECT param_data FROM tb_analysis_model_train WHERE model_id = %s", (model_id,))
        result = cursor.fetchone()
        if not result:
            return jsonify({"state": 404, "message": "Model record not found"}), 404

        param_data = result.get('param_data', "")
        return jsonify({
            "state": 200,
            "data": {
                "param_data": param_data
            }
        }), 200
    except psycopg2.Error as e:
        print(f"Error getting param data: {e}")
        return jsonify({"state": 500, "message": "Failed to get param data"}), 500
    finally:
        cursor.close()
        conn.close()


@train.route('/api/analysis/train/train/param/save', methods=['POST'])
def save_param():
    try:
        data = request.get_json()
        model_train_id = data.get('model_train_id', "")
        model_id = data.get('model_id', "")
        param_data = data.get('param_data', {})
        # 将字典转换为JSON字符串
        param_data_str = json.dumps(param_data)

        if not model_train_id or not model_id:
            return jsonify({"state": 400, "message": "Invalid model_train_id or model_id"}), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "Database connection failed"}), 500

        cursor = conn.cursor()
        try:
            query = "UPDATE tb_analysis_model_train SET param_data = %s WHERE model_train_id = %s AND model_id = %s"
            cursor.execute(query, (param_data_str, model_train_id, model_id))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"state": 404, "message": "Record not found for update"}), 404
            return jsonify({
                "state": 200,
                "data": {
                    "success": "true",
                    "message": "Data updated successfully"
                }
            }), 200
        except psycopg2.Error as e:
            print(f"Error updating param data: {e}")
            conn.rollback()
            return jsonify({"state": 500, "message": "Failed to update param data"}), 500
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"state": 400, "message": "Invalid request format"}), 400

