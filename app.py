import time
import joblib
import torch
from flask import Flask, request, jsonify, render_template, session
import threading
from flask_socketio import SocketIO, emit, join_room, leave_room

import model_function
# 先导入set_socketio_instance，后面设置完emit函数后再导入train_model
from model_function import set_socketio_instance
import pandas as pd
import base64
import io
import os
import datetime
from psycopg2 import Error
import psycopg2
import psycopg2.extras
import feature_extraction
# 预处理的import
from methods import Preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from visualization import Visualizer
import logging
from flask import Flask, request, jsonify
import json
import numpy as np
import uuid
from flask_compress import Compress  # 第一行
from auth import token_required, jwt_token, check_system_token
from hello_routes import hello_blueprint

# 配置日志 (确保在所有 logger 使用之前)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 确保 logger 在这里被定义

# 全局变量定义
active_training_processes = {}
training_sessions = {}

app = Flask(__name__)
Compress(app)
app.secret_key = 'shuohuangapi'
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域
set_socketio_instance(socketio)


# === 通用WebSocket事件发送函数 ===
def emit_process_progress(room_id, process_type, data):
    """发送通用处理进度消息"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': process_type,  # 'preprocess', 'feature_extraction', 'training'
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_progress', payload, namespace='/ns_analysis', room=room_id)
        logger.info(f"发送{process_type}进度消息到房间 {room_id} (ns /ns_analysis): {data.get('message', '')}")
    except Exception as e:
        logger.error(f"发送{process_type}进度消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_completed(room_id, process_type, data):
    """发送通用处理完成消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_completed', payload, namespace='/ns_analysis', room=room_id)
        logger.info(f"发送{process_type}完成消息到房间 {room_id} (ns /ns_analysis)")
    except Exception as e:
        logger.error(f"发送{process_type}完成消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_error(room_id, process_type, error_msg, details=None):
    """发送通用处理错误消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(error_msg)
        }
        if details:
            payload['details'] = str(details)
        socketio.emit('process_error', payload, namespace='/ns_analysis', room=room_id)
        logger.error(f"发送{process_type}错误消息到房间 {room_id} (ns /ns_analysis): {error_msg}")
    except Exception as e:
        logger.error(f"发送{process_type}错误消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_result(room_id, process_type, data):
    """发送通用处理中间结果消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.debug(f"发送{process_type}中间结果到房间 {room_id} (ns /ns_analysis)")
    except Exception as e:
        logger.error(f"发送{process_type}中间结果消息失败 (房间 {room_id}): {e}", exc_info=True)


# === train-WebSocket事件发送函数 ===
def emit_epoch_result(room_id, data):  # 参数名改为 room_id
    """发送单个epoch结果"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': 'training',
            'sub_type':'epoch_result',
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.info(f"发送epoch结果到房间 {room_id} (ns /ns_analysis): epoch {data.get('global_epoch', '')}")
    except Exception as e:
        logger.error(f"发送epoch结果消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_round_result(room_id, data):  # 参数名改为 room_id
    """发送轮次结果"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': 'training',
            'sub_type': 'round_result',
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.info(f"发送轮次结果到房间 {room_id} (ns /ns_analysis): round {data.get('current_round', '')}")
    except Exception as e:
        logger.error(f"发送轮次结果消息失败 (房间 {room_id}): {e}", exc_info=True)





# === WebSocket处理函数 ===
@socketio.on('connect', namespace='/ns_analysis')
def handle_ns_analysis_connect():
    """处理客户端连接到 /ns_analysis namespace"""
    logger.info(f"客户端 {request.sid} 连接到 /ns_analysis namespace")
    emit('connected_to_namespace', {
        'message': '已连接到 /ns_analysis 命名空间。请发送 join_room 事件加入指定房间。',
        'sid': request.sid,
        'timestamp': datetime.datetime.now().isoformat()
    }, room=request.sid)


@socketio.on('disconnect', namespace='/ns_analysis')
def handle_ns_analysis_disconnect():
    """处理客户端从 /ns_analysis namespace 断开连接"""
    room_to_leave = session.get('room')
    if room_to_leave:
        leave_room(room_to_leave)
        logger.info(f"客户端 {request.sid} 自动离开房间 {room_to_leave} (从 /ns_analysis namespace 断开)")
        session.pop('room', None)
    else:
        logger.info(f"客户端 {request.sid} 从 /ns_analysis namespace 断开 (未在 session 中找到房间信息)")


@socketio.on('join_room', namespace='/ns_analysis')
def on_join_ns_analysis_room(data):
    """处理客户端加入处理房间的请求"""
    room_to_join = data.get('room_id')
    if not isinstance(room_to_join, str) or not room_to_join:
        logger.warning(f"客户端 {request.sid} 尝试加入无效的房间名: {room_to_join}。数据: {data}")
        emit('room_join_error', {
            'message': '无效的房间名 (room_id) 或未提供。',
            'requested_room': room_to_join,
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid)
        return

    join_room(room_to_join)
    session['room_id'] = room_to_join
    logger.info(f"客户端 {request.sid} 加入房间: {room_to_join} (ns /ns_analysis)")

    emit('room_joined', {
        'message': f'成功加入房间: {room_to_join}',
        'room_id': room_to_join,
        'timestamp': datetime.datetime.now().isoformat()
    }, room=request.sid)


@socketio.on('leave_room', namespace='/ns_analysis')
def on_leave_ns_analysis_room(data):
    """处理客户端主动离开处理房间的请求"""
    room_to_leave = data.get('room')
    if not isinstance(room_to_leave, str) or not room_to_leave:
        logger.warning(f"客户端 {request.sid} 尝试离开无效的房间名: {room_to_leave}。数据: {data}")
        emit('room_leave_error', {
            'message': '无效的房间名 (room) 或未提供。',
            'requested_room': room_to_leave,
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid)
        return

    leave_room(room_to_leave)
    logger.info(f"客户端 {request.sid} 主动离开房间: {room_to_leave} (ns /ns_analysis)")

    if session.get('room') == room_to_leave:
        session.pop('room', None)

    emit('room_left', {
        'message': f'已离开房间: {room_to_leave}',
        'room': room_to_leave,
        'timestamp': datetime.datetime.now().isoformat()
    }, room=request.sid)




# 配置emit函数映射，将model_function中的emit事件映射到ns_analysis命名空间
from model_function import set_emit_functions

emit_funcs = {
    'training_progress': emit_process_progress,
    'epoch_result': emit_epoch_result,
    'round_result': emit_round_result,
    'training_completed': emit_process_completed,
    'training_error': emit_process_error
}
set_emit_functions(emit_funcs)
#定义全局变量：存储活跃训练会话及其关联的线程和停止事件
active_training_processes = {}
# 在设置完emit函数后导入train_model
from model_function import train_model

# PostgreSQL数据库连接配置
DB_CONFIG = {
    'host': '10.10.1.127',
    'port': '15432',
    'database': 'shuohuang',  # 连接到shuohuang数据库
    'user': 'resafety',
    'password': 'Resafety!@#$%12345postgresre'
}

UPLOAD_FOLDER = "uploads"  # 服务器文件存储路径
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 创建文件夹（如果不存在）


def get_db_connection():
    """获取数据库连接"""
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
        print(f"数据库连接失败: {e}")
        return None


@app.route('/api/analysis/login/token', methods=['POST'])
def token():
    # 获得系统token
    SystemToken = request.headers.get('SystemToken')
    # print('SystemToken'+SystemToken)
    flag = check_system_token(SystemToken)
    # 验证系统token
    if flag is False:
        return jsonify({'error': 'Invalid system token'}), 401
    else:
        value = jwt_token(SystemToken)
        # 将token保存到session作为默认的WebSocket room id
        session['room_id'] = value['token']
        logger.info(f"用户登录成功，设置room ID为: {value['token']}")
        return jsonify(value)


@app.route('/api/analysis/train/select/import', methods=['POST'])
@token_required
def import_selection_file():
    """
    文件上传接口：
    - 参数：
      - file: 上传的文件
    - 响应：
      - 成功返回文件标识符 (file_id)
      - 失败返回错误信息
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"state": 400, "message": "未提供上传文件"}), 400

        file = request.files['file']
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({"state": 400, "message": "未选择文件"}), 400

        original_filename = file.filename  # 获取原始文件名
        file_id = str(uuid.uuid4())  # 生成唯一的文件标识符
        server_filename = f"{file_id}"  # 服务器上存储的文件名使用 file_id
        file_path = os.path.join(UPLOAD_FOLDER, server_filename)

        # 检查原始文件名是否已存在于数据库中
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        try:
            cursor = conn.cursor()
            check_sql = "SELECT file_id FROM tb_analysis_sample_file WHERE file_name = %s"
            cursor.execute(check_sql, (original_filename,))
            existing_file = cursor.fetchone()
            if existing_file:
                return jsonify({"state": 409, "message": f"文件名 '{original_filename}' 已存在"}), 409
        except Exception as e:
            cursor.close()
            conn.close()
            return jsonify({"state": 500, "message": f"数据库查询失败: {str(e)}"}), 500
        finally:
            if conn:
                cursor.close()

        # 保存文件到服务器，使用 file_id 作为文件名
        try:
            file.save(file_path)
        except Exception as e:
            if conn:
                conn.close()
            return jsonify({"state": 500, "message": f"文件保存失败: {str(e)}"}), 500

        # 读取 CSV
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            if conn:
                conn.close()
            os.remove(file_path)  # 删除已保存但读取失败的文件
            return jsonify({"state": 404, "message": "文件未找到"}), 404
        except pd.errors.ParserError:
            if conn:
                conn.close()
            os.remove(file_path)  # 删除已保存但解析失败的文件
            return jsonify({"state": 400, "message": "CSV 解析失败"}), 400

        # 插入数据库，file_name 存储原始文件名
        try:
            cursor = conn.cursor()
            sql = """
            INSERT INTO tb_analysis_sample_file (file_id, file_name, file_path, demo, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (file_id, original_filename, file_path, "自动上传", "admin", datetime.datetime.now()))
            conn.commit()
            return jsonify({"state": 200, "data": {"file_id": file_id}}), 200
        except Exception as e:
            conn.rollback()
            os.remove(file_path)  # 删除已保存但数据库插入失败的文件
            return jsonify({"state": 500, "message": f"数据库操作失败: {str(e)}"}), 500
        finally:
            if conn:
                cursor.close()
                conn.close()

    except Exception as e:
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500


@app.route('/api/analysis/train/select/sample', methods=['GET'])
@token_required
def get_sample_data():
    file_id = request.args.get('file_id')
    create_user = request.args.get('create_user', 'system')

    if not file_id:
        return jsonify({"state": 400, "message": "未提供文件标识符"}), 400

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        cursor = conn.cursor()

        # 查询是否已存在该 file_id 的原始样本数据
        cursor.execute("SELECT original_sample_id, sample_name FROM tb_analysis_sample_original WHERE file_id = %s",
                       (file_id,))
        existing_samples = cursor.fetchall()

        if existing_samples:
            data_to_return = [{"sample_id": sample_id, "sample_name": sample_name} for sample_id, sample_name in
                              existing_samples]
            return jsonify({
                "state": 200,
                "data": data_to_return,
                "message": f"文件标识符 {file_id} 的样本数据已存在，已返回现有数据。"
            }), 200
        else:
            # 查询文件路径
            cursor.execute("SELECT file_path FROM tb_analysis_sample_file WHERE file_id = %s", (file_id,))
            result = cursor.fetchone()

            if not result:
                return jsonify({"state": 404, "message": "未找到对应的文件"}), 404

            file_path = result[0]
            if not os.path.exists(file_path):
                return jsonify({"state": 404, "message": "文件不存在"}), 404

            # 读取 CSV 数据
            df = pd.read_csv(file_path)

            sample_list = []
            data_to_return = []
            file_uuid = str(uuid.uuid4())  # 为当前文件生成一个 UUID

            for i, col_name in enumerate(df.columns):
                # 生成 original_sample_id
                original_sample_id = f"{file_uuid}_{i + 1}"

                # 生成唯一的 sample_name
                sample_name = f"{col_name}"

                # 直接获取列数据作为列表
                sample_data_list = df[col_name].tolist()
                # 将 Python 列表转换为 JSON 字符串
                sample_data_json = json.dumps(sample_data_list)

                data_to_return.append({
                    "sample_id": original_sample_id,
                    "sample_name": col_name
                })

                sample_list.append((
                    original_sample_id, file_id, sample_name, sample_data_json, '1', create_user
                ))

            # 批量插入数据
            insert_query = """
            INSERT INTO tb_analysis_sample_original (
                original_sample_id, file_id, sample_name, sample_data, sample_state, create_user, create_time
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW());
            """
            cursor.executemany(insert_query, sample_list)
            conn.commit()

            return jsonify({
                "state": 200,
                "data": data_to_return
            }), 200

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        return jsonify({"state": 500, "message": f"PostgreSQL 操作失败: {str(e)}"}), 500
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"state": 500, "message": f"操作失败: {str(e)}"}), 500

    finally:
        if conn:
            if cursor:
                cursor.close()
            conn.close()


@app.route('/api/analysis/train/select/wave', methods=['GET'])
@token_required
def get_wave_data():
    """
    根据 sample_ids 返回样本基础信息（ID、名称、原始数据）
    """
    sample_ids = request.args.get('sample_ids')

    if not sample_ids:
        return jsonify({"state": 400, "message": "未提供样本 ID"}), 400

    # 清洗参数并分割为列表
    sample_ids_list = [sid.strip() for sid in sample_ids.split(',') if sid.strip()]
    if not sample_ids_list:
        return jsonify({"state": 400, "message": "样本 ID 格式无效"}), 400

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        # 使用 DictCursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 构建 IN 子句的占位符字符串
        placeholders = ','.join(['%s'] * len(sample_ids_list))

        # 查询样本基础信息（包含sample_name）
        query = f"""
        SELECT original_sample_id, sample_name, sample_data
        FROM tb_analysis_sample_original
        WHERE original_sample_id IN ({placeholders})
        """

        print(f"执行的 SQL 查询: {query}")  # 打印执行的 SQL 语句
        print(f"传递的参数: {tuple(sample_ids_list)}")  # 打印传递的参数

        cursor.execute(query, tuple(sample_ids_list))
        results = cursor.fetchall()

        print(f"查询结果数量: {len(results)}")  # 打印查询结果的数量

        if not results:
            return jsonify({
                "state": 404,  # 根据需求修改为404表示失败
                "message": "未找到样本数据"
            }), 404

        # 构建响应数据
        response_data = []
        for row in results:
            try:
                # 直接返回原始数据（无需解析时间序列）
                sample_data = json.loads(row["sample_data"])
                response_data.append({
                    "sample_id": row["original_sample_id"],
                    "sample_name": row["sample_name"],
                    "sample_data": sample_data  # 直接返回完整数据
                })
            except json.JSONDecodeError:
                print(f"JSON 解析错误的数据: {row['sample_data']}")  # 打印解析失败的数据
                continue  # 跳过格式错误的数据

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        print(f"发生异常: {e}")  # 打印异常信息
        return jsonify({"state": 500, "message": f"操作失败: {str(e)}"}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/api/analysis/train/select/add/sample', methods=['POST'])
@token_required
def add_sample_data():
    """
    根据请求的 sample_ids，将选中的样本状态更新为 2，表示已加入样本库
    """
    # 获取 URL 参数中的 sample_ids
    sample_ids = request.args.get('sample_ids')

    if not sample_ids:
        return jsonify({
            "state": 400,
            "data": {
                "success": "false",
                "message": "未提供样本 ID"
            }
        }), 400

    # 解析逗号分隔的 sample_ids
    sample_ids_list = sample_ids.split(',')

    # 连接数据库
    conn = None
    cursor = None  # 初始化cursor
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({
                "state": 500,
                "data": {
                    "success": "false",
                    "message": "数据库连接失败"
                }
            }), 500

        cursor = conn.cursor()

        # **检查数据库是否存在这些样本 ID**
        placeholders = ', '.join(['%s'] * len(sample_ids_list))
        check_sql = f"""
            SELECT original_sample_id FROM tb_analysis_sample_original 
            WHERE original_sample_id IN ({placeholders})
        """
        cursor.execute(check_sql, tuple(sample_ids_list))
        existing_samples = {row[0] for row in cursor.fetchall()}  # 获取存在的 sample_id

        if not existing_samples:
            return jsonify({
                "state": 404,
                "data": {
                    "success": "false",
                    "message": "没有匹配的样本 ID，无法更新"
                }
            }), 404

        # **过滤掉无效的 sample_ids**
        valid_sample_ids = list(existing_samples)
        if not valid_sample_ids:
            return jsonify({
                "state": 400,
                "data": {
                    "success": "false",
                    "message": "提供的样本 ID 无效，无法更新"
                }
            }), 400

        # **更新数据库中有效的样本**
        placeholders = ', '.join(['%s'] * len(valid_sample_ids))
        update_sql = f"""
            UPDATE tb_analysis_sample_original 
            SET sample_state = '2'
            WHERE original_sample_id IN ({placeholders})
        """
        cursor.execute(update_sql, tuple(valid_sample_ids))
        conn.commit()

        return jsonify({
            "state": 200,
            "data": {
                "success": "true",
                "message": f"成功更新 {cursor.rowcount} 条样本数据"
            }
        }), 200

    except Exception as e:
        return jsonify({
            "state": 500,
            "data": {
                "success": "false",
                "message": f"服务器内部错误: {str(e)}"
            }
        }), 500

    finally:
        if cursor:  # 增加判断，避免cursor未被赋值，导致的报错
            cursor.close()
        if conn:
            conn.close()


# 5.15新增接口，保证后续所有的步骤仅仅使用某一个文件的样本
@app.route('/api/analysis/train/select/files', methods=['GET'])
@token_required
def get_file_identifiers():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "Failed to connect to the database"})

        cursor = conn.cursor()
        cursor.execute("SELECT file_id, file_name FROM tb_analysis_sample_file")
        files = cursor.fetchall()

        if files:
            file_list = []
            for file in files:
                file_list.append({"file_id": file[0], "file_name": file[1]})
            data = {"files": file_list}
            return jsonify({"state": 200, "data": data})
        else:
            return jsonify({"state": 404, "message": "No files found"})

    except Error as e:
        if conn:
            conn.rollback()
        return jsonify({"state": 500, "message": f"Database error: {e}"})
    except Exception as e:
        return jsonify({"state": 500, "message": str(e)})
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/api/analysis/train/pre/sample', methods=['GET'])
@token_required
def get_samples():
    """
    获取已加入样本库的样本，并根据传入的 file_id 进行过滤。
    需要前端在调用时通过查询参数传递 file_id。
    示例: GET /api/analysis/train/pre/sample?file_id=your_selected_file_id
    """
    conn = None
    cursor = None
    try:
        # 1. 从请求的查询参数中获取 file_id
        file_id = request.args.get('file_id')

        # 检查 file_id 是否已提供
        if not file_id:
            return jsonify({"state": 400, "message": "缺少 file_id 参数"}), 400  # 400 Bad Request

        # 调用外部定义的 get_db_connection 函数
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        # 使用 DictCursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 2. 修改 SQL 查询，增加 file_id 的过滤条件，并使用参数化查询防止SQL注入
        sql_query = """
            SELECT original_sample_id AS sample_id, sample_name, 1 AS sample_type
            FROM tb_analysis_sample_original
            WHERE CAST(sample_state AS INTEGER) = 2 AND file_id = %s
        """
        # 执行查询，将 file_id 作为参数传递
        cursor.execute(sql_query, (file_id,))  # 注意这里 file_id 是一个单元素的元组

        samples = cursor.fetchall()

        # 根据查询结果返回不同的状态
        if not samples:
            # 找到了 file_id，但是该 file_id 下没有符合条件的样本
            return jsonify({"state": 404, "data": [], "message": f"在 file_id {file_id} 下没有找到符合条件的样本"}), 404
        else:
            # 找到了符合条件的样本
            # 将 DictRow 对象转换为字典列表以便 jsonify 序列化
            sample_list = [dict(row) for row in samples]
            return jsonify({"state": 200, "data": sample_list}), 200

    except Exception as e:
        # 记录更详细的错误信息到日志（实际应用中应该这么做）
        # app.logger.error(f"Error fetching samples for file_id {file_id}: {e}")
        return jsonify({"state": 500, "message": f"数据库查询失败: {str(e)}"}), 500
    finally:
        # 确保在任何情况下都关闭游标和连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/api/analysis/train/pre/method', methods=['GET'])
@token_required
def get_methods():
    """获取所有可用的方法"""
    methods_data = [
        {"stage": "1", "stage_name": "异常缺失处理", "data": [{"method_name": "异常缺失处理"}]},
        {"stage": "2", "stage_name": "数据降噪", "data": [{"method_name": "高斯滤波"}, {"method_name": "均值滤波"}]},
        {"stage": "3", "stage_name": "数据规范化",
         "data": [{"method_name": "最大最小规范化"}, {"method_name": "Z-score标准化"}]},
        {"stage": "4", "stage_name": "数据降维", "data": [{"method_name": "主成分分析"}, {"method_name": "线性判别"}]}
    ]
    return jsonify({"state": 200, "data": methods_data})


@app.route('/api/analysis/train/pre/confirm', methods=['POST'])
@token_required
def preprocess_confirm():
    """
    数据预处理确认接口 (支持跳过步骤)。
    对于同一个 original_sample_id，只保留最后一次的预处理结果（先删除旧数据再插入新数据）。
    """
    conn = None
    cursor = None
    try:
        data = request.get_json()
        # 接收以逗号分隔的字符串
        sample_ids_str = data.get('sample_ids', '')
        pre_process_methods_str = data.get('pre_process_method', '')

        # 获取WebSocket room id，优先使用请求中指定的，否则使用session中的
        room_id = session.get('room_id')
        if not room_id:
            logger.warning("预处理操作未找到有效的room ID，将无法通过WebSocket发送进度")

        if not sample_ids_str:
            return jsonify({"state": 400, "message": "缺少必要参数 sample_ids"}), 400

        # 将样本ID字符串分割成列表
        sample_ids_list = [sid.strip() for sid in sample_ids_str.split(',') if sid.strip()]

        if not sample_ids_list:
            return jsonify({"state": 400, "message": "无效的样本ID"}), 400

        # WebSocket 通知开始预处理
        if room_id:
            emit_process_progress(room_id, 'preprocess', {
                'status': 'started',
                'message': f'开始预处理 {len(sample_ids_list)} 个样本',
                'total_samples': len(sample_ids_list),
                'processed_samples': 0
            })

        # 处理预处理方法字符串：按逗号分割，并映射到预处理步骤
        # 预处理的固定顺序和可选方法
        pipeline_steps = [
            "异常缺失处理",  # 步骤1
            "数据降噪",  # 步骤2
            "数据规范化",  # 步骤3
            "数据降维"  # 步骤4
        ]

        pipeline_methods = {
            "异常缺失处理": ["异常缺失处理"],
            "数据降噪": ["高斯滤波", "均值滤波"],
            "数据规范化": ["最大最小规范化", "Z-score标准化"],
            "数据降维": ["主成分分析", "线性判别"]
        }

        # 解析用户提交的预处理方法
        pre_process_methods_from_input = []
        method_parts = pre_process_methods_str.split(',')[:4]  # 只取前4个
        while len(method_parts) < 4:
            method_parts.append('')

        for method in method_parts:
            method = method.strip()
            if method:
                # 验证方法是否有效
                valid = False
                for step_methods in pipeline_methods.values():
                    if method in step_methods:
                        valid = True
                        break
                if valid:
                    pre_process_methods_from_input.append(method)
                else:
                    print(f"警告：忽略无效的处理方法 '{method}'")

        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 查询原始样本数据 - 使用参数化查询
        placeholders = ','.join(['%s'] * len(sample_ids_list))
        query = f"""
        SELECT original_sample_id, sample_name, sample_data
        FROM tb_analysis_sample_original
        WHERE original_sample_id IN ({placeholders})
        """
        cursor.execute(query, tuple(sample_ids_list))
        original_samples = cursor.fetchall()

        if not original_samples:
            if room_id:
                emit_process_error(room_id, 'preprocess', "未找到样本数据")
            return jsonify({"state": 404, "message": "未找到样本数据"}), 404

        response_data = []
        total_samples = len(original_samples)

        # 定义 DELETE 语句 (只根据 original_sample_id 删除)
        delete_sql = """
            DELETE FROM tb_analysis_sample_preprocess
            WHERE original_sample_id = %s
        """

        # 定义 INSERT 语句
        insert_sql = """
            INSERT INTO tb_analysis_sample_preprocess (
                preprocess_sample_id, original_sample_id, preprocess_method,
                preprocess_sample_data, create_user, create_time
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        processed_count = 0
        for sample in original_samples:
            try:
                processed_count += 1
                original_id = sample["original_sample_id"]
                original_data_str = sample["sample_data"].strip()

                # 发送进度信息
                if room_id and processed_count % max(1, total_samples // 10) == 0:  # 每处理10%的样本发送一次
                    progress_percentage = (processed_count / total_samples) * 100
                    emit_process_progress(room_id, 'preprocess', {
                        'status': 'processing',
                        # 前端获取这个应该就行？
                        'message': f'正在处理样本 {processed_count}/{total_samples}',
                        'progress': progress_percentage,
                        'current_sample': original_id,
                        'processed_samples': processed_count,
                        'total_samples': total_samples
                    })

                # 确保数据是有效的 JSON 列表
                if original_data_str.startswith('[') and original_data_str.endswith(']'):
                    current_data = json.loads(original_data_str)
                else:
                    raise json.JSONDecodeError("不是有效的 JSON 列表", original_data_str, 0)

                # 提取原始列名，用于生成 preprocess_sample_id
                original_column_name = sample["sample_name"]

                # 保留 before_process_data 结构
                before_process_data = {
                    "data": list(current_data),
                    # 假设 Visualizer 类和方法可用
                    "waveform": Visualizer.generate_waveform(current_data),
                    "spectrum": Visualizer.generate_spectrum(current_data)
                }

                processed_value = list(current_data)
                applied_methods_list_for_log = []  # 用于记录实际应用的步骤日志

                # 标记每个步骤是否已应用，确保每个步骤只应用一次
                applied_steps = {step: False for step in pipeline_steps}

                # 按照 pipeline_steps 的顺序应用方法
                for step in pipeline_steps:
                    method_for_step = None
                    # 遍历用户提交的方法列表，找到属于当前步骤的第一个方法
                    for method in pre_process_methods_from_input:
                        if method in pipeline_methods.get(step, []):
                            method_for_step = method
                            break  # 找到第一个就用它

                    if method_for_step and not applied_steps[step]:
                        try:
                            processed_value = Preprocessor.apply_method(processed_value, method_for_step)
                            applied_methods_list_for_log.append(method_for_step)  # 记录应用的具体方法
                            applied_steps[step] = True  # 标记该步骤已应用
                        except Exception as method_e:
                            print(f"应用方法 '{method_for_step}' 到样本 {original_id} 时出错: {method_e}")
                            applied_methods_list_for_log.append(f"应用 {method_for_step} 失败")

                # 保留 after_process_data 结构
                after_process_data = {
                    "data": list(processed_value),
                    # 假设 Visualizer 类和方法可用
                    "waveform": Visualizer.generate_waveform(processed_value),
                    "spectrum": Visualizer.generate_spectrum(processed_value)
                }

                # 生成 preprocess_sample_id，保留原始列名后缀
                preprocess_sample_id_base = original_id
                # 修改数据表命名规则
                new_preprocess_sample_id = f"{preprocess_sample_id_base}--{original_column_name}"

                # 将实际成功应用的 methods 列表转换为 JSON 字符串，用于数据库存储
                applied_methods_final_json = json.dumps(applied_methods_list_for_log, ensure_ascii=False)

                # --- 新增的删除逻辑 ---
                # 在插入新数据之前，删除具有相同 original_sample_id 的所有旧数据
                # 这样确保同一个 original_sample_id 在表中最多只有一条记录
                cursor.execute(delete_sql, (original_id,))
                # --- 删除逻辑结束 ---

                # 存储最终的预处理结果到 tb_analysis_sample_preprocess
                # original_sample_id 字段直接使用原始样本的 original_sample_id
                cursor.execute(insert_sql, (
                    new_preprocess_sample_id,
                    original_id,  # 直接使用原始样本的 ID
                    applied_methods_final_json,  # 存储实际应用的 methods JSON 字符串
                    json.dumps(processed_value),
                    "system",  # 假设创建用户是 system
                    datetime.datetime.now()
                ))

                # 为响应数据添加条目
                response_data.append({
                    "sample_id": original_id,  # 响应中仍然使用原始样本ID作为标识
                    "sample_name": sample["sample_name"],
                    "beforeprocess": before_process_data,
                    "afterprocess": after_process_data,
                    "process_log": applied_methods_list_for_log,  # 返回实际应用的步骤日志（列表形式）
                    "new_preprocess_sample_id": new_preprocess_sample_id  # 将新生成的预处理样本ID也返回
                })

            except json.JSONDecodeError as e:
                print(f"样本 {original_id} 的数据解析失败: {e}")
                continue  # 对于单个样本的处理失败，记录错误并跳过该样本
            except ValueError as e:
                print(f"样本 {original_id} 预处理方法错误: {str(e)}")
                continue  # 对于单个样本的处理失败，记录错误并跳过该样本
            except Exception as e:
                print(f"处理样本 {original_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue  # 对于单个样本的处理失败，记录错误并跳过该样本

        # 提交事务：确保所有的删除和插入操作是原子性的对于整个请求而言
        # 如果中间有样本处理失败（跳过），不会影响已成功处理并准备插入/删除的数据的提交
        conn.commit()

        # 发送处理完成的WebSocket消息
        if room_id:
            emit_process_completed(room_id, 'preprocess', {
                'status': 'completed',
                'message': f'预处理完成，共处理 {processed_count} 个样本',
                'processed_samples': processed_count,
                'total_samples': total_samples,
                'successful_samples': len(response_data)
            })

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        # 捕获整个请求处理过程中的异常（如数据库连接失败，JSON解析错误等）
        if conn:
            conn.rollback()  # 如果发生任何未捕获的异常，回滚整个事务
        print(f"服务器内部错误: {str(e)}")
        import traceback
        traceback.print_exc()

        # 发送错误消息
        if room_id:
            emit_process_error(room_id, 'preprocess', f"预处理过程中发生错误: {str(e)}")

        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        # 确保在任何情况下都关闭游标和连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/api/analysis/train/feature/fetch', methods=['GET'])
@token_required
def fetch_feature():
    """
    返回多种特征提取方法的列表。
    """
    response_data = {
        "state": 200,
        "data": [
            {
                "stage": "1",
                "methods": [
                    {"method_name": "峭度指标"},
                    {"method_name": "直方图特征"},
                    {"method_name": "傅里叶变换"},
                    {"method_name": "小波变换"},
                    {"method_name": "经验模态分解"},
                    {"method_name": "Wigner-Ville分布"},
                ]
            },
            {
                "stage": "2",
                "methods": [
                    {"method_name": "Fisher判别法"},
                    {"method_name": "Relief算法"}
                ]
            }
        ]
    }
    return jsonify(response_data)


@app.route('/api/analysis/train/feature/sample', methods=['GET'])
@token_required
def fetch_samples_prioritized():
    """
    获取样本列表，对于属于指定 file_id 且状态为 '2' 的每个原始样本：
    优先返回其关联的预处理样本，如果不存在预处理样本，则返回原始样本本身。

    需要前端在调用时通过查询参数传递 file_id。
    示例: GET /api/analysis/train/feature/sample?file_id=your_selected_file_id

    Returns:
        JSON: 包含符合条件的统一格式的样本列表的JSON响应
    """
    conn = None
    cursor = None
    try:
        # 1. 从请求的查询参数中获取 file_id
        file_id = request.args.get('file_id')

        # 检查 file_id 是否已提供
        if not file_id:
            return jsonify({"state": 400, "message": "缺少 file_id 参数"}), 400  # 400 Bad Request

        # 连接数据库
        conn = get_db_connection()  # 假设你有一个获取数据库连接的函数
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  # 使用DictCursor

        # --- 开始数据获取和处理 ---

        # 步骤 A: 获取所有与指定 file_id 且状态为 '2' 关联的原始样本的 ID 和名称
        query_relevant_original_ids = """
            SELECT
                original_sample_id,
                sample_name,
                create_time -- 获取创建时间用于排序
            FROM
                tb_analysis_sample_original
            WHERE
                sample_state = '2' AND file_id = %s
            ORDER BY
                create_time DESC -- 按照原始样本创建时间排序
        """
        cursor.execute(query_relevant_original_ids, (file_id,))
        relevant_originals = cursor.fetchall()

        if not relevant_originals:
            # 如果该 file_id 下没有符合条件的原始样本，直接返回空列表
            return jsonify({"state": 200, "data": []}), 200

        # 将原始样本数据存入字典，方便按 ID 查找
        original_samples_dict = {
            row['original_sample_id']: {
                "sample_id": row["original_sample_id"],
                "sample_name": row["sample_name"],
                "sample_type": 1,
            }
            for row in relevant_originals
        }

        # 步骤 B: 获取所有与这些原始样本 ID 关联的预处理样本
        # 通过 JOIN 关联原始样本表以确保 file_id 和 sample_state='2' 条件
        # 使用 IN 子句限制只查询与步骤 A 中获取的 original_sample_id 相关的预处理样本，提高效率
        relevant_original_ids_list = [row['original_sample_id'] for row in relevant_originals]
        # 将 ID 列表转换为适合 SQL IN 子句的格式 (e.g., ('id1', 'id2', ...))
        # 如果列表为空，SQL会有问题，但前面已经检查 relevant_originals 不为空
        original_ids_placeholder = ','.join(['%s'] * len(relevant_original_ids_list))

        query_preprocess_for_originals = f"""
            SELECT
                tpp.preprocess_sample_id AS sample_id,
                tpp.original_sample_id,
                tpp.preprocess_method,
                tpo.sample_name AS original_sample_name, -- 获取关联的原始样本名称
                tpp.create_time, -- 获取预处理样本创建时间
                2 AS sample_type
            FROM
                tb_analysis_sample_preprocess tpp
            JOIN
                tb_analysis_sample_original tpo ON tpp.original_sample_id = tpo.original_sample_id
            WHERE
                 tpp.original_sample_id IN ({original_ids_placeholder}) -- 只获取与之前查到的原始样本ID关联的预处理样本
                 AND tpo.sample_state = '2' -- 确保关联的原始样本状态为 '2' (双重检查，IN子句已基本保证)
                 AND tpo.file_id = %s -- 根据原始样本表的 file_id 过滤 (双重检查，IN子句已基本保证)
            ORDER BY
                tpp.create_time DESC -- 按照预处理样本创建时间排序
        """
        # 执行查询，参数包括 original_sample_ids 列表的元素和 file_id
        cursor.execute(query_preprocess_for_originals, relevant_original_ids_list + [file_id])
        preprocess_samples_data = cursor.fetchall()

        # 将预处理样本按 original_sample_id 分组
        preprocess_samples_by_original_id = {}
        for sample in preprocess_samples_data:
            original_id = sample['original_sample_id']
            if original_id not in preprocess_samples_by_original_id:
                preprocess_samples_by_original_id[original_id] = []

            # 构建预处理样本的返回格式
            sample_name = f"{sample['original_sample_name']} ({sample['preprocess_method']})" if sample.get(
                'original_sample_name') else f"{sample['original_sample_id']} ({sample['preprocess_method']})"
            preprocess_samples_by_original_id[original_id].append({
                "sample_id": sample["sample_id"],
                "sample_name": sample_name,
                "sample_type": sample["sample_type"],
            })

        # 步骤 C: 构建最终的样本列表，应用优先级逻辑
        final_samples = []
        # 遍历之前查到的相关原始样本的 ID 列表（保持原始样本的排序）
        for original_id_row in relevant_originals:
            original_id = original_id_row['original_sample_id']

            if original_id in preprocess_samples_by_original_id:
                # 如果存在预处理样本，将所有预处理样本加入最终列表
                # 预处理样本已经在步骤 B 中按 create_time DESC 排序
                final_samples.extend(preprocess_samples_by_original_id[original_id])
            else:
                # 如果不存在预处理样本，加入原始样本
                # 从字典中获取原始样本数据，该数据已在步骤 A 中按 create_time DESC 排序
                final_samples.append(original_samples_dict[original_id])

        # final_samples 目前是按照原始样本的 create_time 排序的，
        # 如果某个原始样本有多个预处理样本，这些预处理样本会跟在其原始位置后，
        # 并在其内部按预处理时间排序。这个排序逻辑通常是可接受的。
        # 如果需要全局按时间排序，需要收集所有样本后统一排序，但会丢失原始样本的顺序分组感。
        # 当前实现是按"原始样本出现的顺序"来决定组的顺序，组内再按各自时间排序。

        # 返回响应
        response_data = {
            "state": 200,
            "data": final_samples
        }
        return jsonify(response_data), 200

    except Exception as e:
        # 记录错误
        # 实际应用中应该使用 proper logging framework like app.logger
        print(f"获取样本列表失败: {str(e)}")
        if conn:
            # 对于GET请求，rollback通常不是必须的
            pass

        # 返回错误响应
        return jsonify({
            "state": 500,  # 内部错误
            "message": f"获取样本列表失败: {str(e)}"
        }), 500
    finally:
        # 确保在任何情况下都关闭游标和连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# 5.12特征选择还未实现

@app.route('/api/analysis/train/feature/confirm', methods=['POST'])
@token_required
def feature_confirm():
    conn = None
    cursor = None
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"state": 400, "message": "请求体为空或非JSON格式"}), 400

        sample_ids_input = request_data.get('sample_ids', [])
        if isinstance(sample_ids_input, str):
            sample_ids = [sid.strip() for sid in sample_ids_input.split(',') if sid.strip()]
        elif isinstance(sample_ids_input, list):
            sample_ids = [str(sid).strip() for sid in sample_ids_input if str(sid).strip()]
        else:
            sample_ids = []

        feature_method = request_data.get('feature_method', '')
        current_user = "system"

        # 获取WebSocket room id，优先使用请求中指定的，否则使用session中的
        room_id = session.get('room_id')
        if not room_id:
            logger.warning("特征提取操作未找到有效的room ID，将无法通过WebSocket发送进度")

        if not sample_ids:
            return jsonify({"state": 400, "message": "样本ID列表为空或无效"}), 400

        # WebSocket 通知开始特征提取
        if room_id:
            emit_process_progress(room_id, 'feature_extraction', {
                'status': 'started',
                'message': f'开始特征提取，使用方法: {feature_method}，共 {len(sample_ids)} 个样本',
                'total_samples': len(sample_ids),
                'processed_samples': 0,
                'feature_method': feature_method
            })

        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败（未能获取连接对象）"}), 500

        feature_params = {}
        if feature_method:
            try:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    SELECT feature_extract_param
                    FROM tb_analysis_sample_feature
                    WHERE feature_extract = %s AND feature_extract_param IS NOT NULL
                    ORDER BY create_time DESC
                    LIMIT 1
                """, (feature_method,))
                db_param = cursor.fetchone()
                if db_param and db_param["feature_extract_param"]:
                    param_data = db_param["feature_extract_param"]
                    try:
                        if isinstance(param_data, str):
                            feature_params = json.loads(param_data)
                        elif isinstance(param_data, dict):
                            feature_params = param_data
                        else:
                            logging.warning(f"数据库中 feature_extract_param 格式非字符串或字典: {type(param_data)}")
                            feature_params = {}
                    except json.JSONDecodeError:
                        logging.error(f"解析数据库中特征参数JSON失败: {db_param['feature_extract_param']}")
                        feature_params = {}
                    except Exception as e_parse:
                        logging.error(
                            f"解析数据库中特征参数时发生其他错误: {e_parse}. 参数: {db_param['feature_extract_param']}")
                        feature_params = {}
            except Exception as e_db_param:
                logging.error(f"查询数据库特征参数失败: {e_db_param}")
                feature_params = {}
            finally:
                if cursor:
                    cursor.close()
                    cursor = None

        default_params = {
            "直方图特征": {"bins": 10}, "小波变换": {"wavelet_name": "db4"},
            "经验模态分解": {"max_imfs": None, "sift_thresh": 1e-8}, "Wigner-Ville分布": {}
        }
        method_defaults = default_params.get(feature_method, {})
        for key, val in method_defaults.items():
            feature_params.setdefault(key, feature_params.get(key, val))

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        all_before_process_sample_details = []
        all_processed_sample_details = []

        delete_feature_sql = """
            DELETE FROM tb_analysis_sample_feature
            WHERE from_sample_id = %s
        """
        insert_sql = """
            INSERT INTO tb_analysis_sample_feature (
                feature_sample_id, from_sample_id, from_sample_type,
                feature_extract, feature_extract_param, feature_select,
                feature_sample_data, create_user, create_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        processed_count = 0
        total_samples = len(sample_ids)

        for  sample_id in sample_ids:
            processed_count += 1
            if not sample_id:
                logging.warning("在样本ID列表中发现一个空ID，已跳过。")
                continue
            logging.debug(f"开始处理样本ID: {sample_id}")

            original_sample_data_str = None
            sample_type = None
            determined_legend_text = str(sample_id).split('--')[-1]
            # 修改feature_extract字段赋值：sample_name+feature_method
            feature_extract_field_value = f"{determined_legend_text}_{feature_method}" if feature_method else determined_legend_text

            cursor.execute(
                "SELECT original_sample_id, sample_data, sample_name FROM tb_analysis_sample_original WHERE original_sample_id = %s",
                (sample_id,))
            original_sample_row = cursor.fetchone()

            if original_sample_row:
                if original_sample_row.get("sample_name") == 'label':
                    logging.info(f"样本ID {sample_id} (原始表) 指向 'label' 定义，跳过作为数据源。")
                    original_sample_data_str = None
                else:
                    original_sample_data_str = original_sample_row["sample_data"]
                    sample_type = "1"
                    # 定义
                    determined_legend_text = original_sample_row.get("sample_name")
                    if not determined_legend_text:
                        determined_legend_text = str(sample_id)
                        logging.warning(f"样本ID {sample_id} (原始表): sample_name为空，使用ID '{sample_id}' 作为图例。")
                    else:
                        logging.debug(
                            f"样本ID {sample_id} (原始表): 使用 sample_name '{determined_legend_text}' 作为图例。")

            if original_sample_data_str is None:
                cursor.execute(
                    "SELECT preprocess_sample_data, preprocess_method, original_sample_id FROM tb_analysis_sample_preprocess WHERE preprocess_sample_id = %s",
                    (sample_id,))
                preprocess_sample_data_row = cursor.fetchone()
                if preprocess_sample_data_row:
                    original_sample_data_str = preprocess_sample_data_row["preprocess_sample_data"]
                    sample_type = "2"
                    method_name_json = preprocess_sample_data_row.get("preprocess_method")
                    method_name_str = ""
                    if method_name_json:
                        try:
                            method_name_list = json.loads(method_name_json)
                            if isinstance(method_name_list, list) and method_name_list:
                                method_name_str = "_".join(method_name_list)
                            elif isinstance(method_name_json, str):
                                method_name_str = method_name_json
                        except json.JSONDecodeError:
                            method_name_str = method_name_json
                        except Exception as e_method_name:
                            logging.warning(f"样本ID {sample_id} (预处理表): 解析方法名字段失败: {e_method_name}")
                    parts = sample_id.rsplit('_', 1)
                    if len(parts) > 1 and parts[-1]:
                        determined_legend_text = parts[-1]
                        logging.debug(
                            f"样本ID {sample_id} (预处理表): 从ID提取部分 '{determined_legend_text}' 作为图例。")
                    elif method_name_str:
                        determined_legend_text = method_name_str
                        logging.debug(
                            f"样本ID {sample_id} (预处理表): ID提取部分无效，使用方法名 '{determined_legend_text}' 作为图例。")
                    else:
                        determined_legend_text = str(sample_id)
                        logging.warning(
                            f"样本ID {sample_id} (预处理表): ID提取部分和方法名均无效，使用ID '{determined_legend_text}' 作为图例。")

            if original_sample_data_str is None:
                logging.warning(f"样本ID {sample_id} 未找到有效数据源。")
                continue

            sample_data_json = None
            try:
                if isinstance(original_sample_data_str, str):
                    sample_data_json = json.loads(original_sample_data_str)
                elif isinstance(original_sample_data_str, (list, dict)):
                    sample_data_json = original_sample_data_str
                elif original_sample_data_str is not None:
                    sample_data_json = json.loads(str(original_sample_data_str))
                else:
                    logging.warning(f"样本ID {sample_id} 原始数据为None或无效类型。")
                    continue
            except Exception as e:
                logging.error(f"样本ID {sample_id} JSON解析失败: {e}. 数据: '{str(original_sample_data_str)[:200]}'")
                continue

            raw_data_np = None

            # 发送进度信息
            if room_id and processed_count % max(1, total_samples // 10) == 0:  # 每处理10%的样本发送一次
                progress_percentage = (processed_count / total_samples) * 100
                emit_process_progress(room_id, 'feature_extraction', {
                    'status': 'processing',
                    'message': f'正在处理样本 {processed_count}/{total_samples}',
                    'progress': progress_percentage,
                    'current_sample': sample_id,
                    'processed_samples': processed_count,
                    'total_samples': total_samples,
                    'feature_method': feature_method
                })

            if isinstance(sample_data_json, list):
                try:
                    raw_data_np = np.array(sample_data_json)
                except Exception as e:
                    logging.error(f"样本ID {sample_id} 列表转NumPy失败: {e}");
                    continue
            elif isinstance(sample_data_json, dict):
                if 'features' in sample_data_json:
                    try:
                        raw_data_np = np.array(sample_data_json['features'])
                    except Exception as e:
                        logging.error(f"样本ID {sample_id} dict 'features'转NumPy失败: {e}");
                        continue
                else:
                    try:
                        values_list = list(sample_data_json.values())
                        if len(values_list) == 1 and isinstance(values_list[0], list):
                            raw_data_np = np.array(values_list[0])
                        else:
                            logging.warning(f"样本ID {sample_id} JSON字典结构非预期的包含'features'或单列表值。")
                            continue
                    except Exception as e_dict_values:
                        logging.error(f"样本ID {sample_id} JSON字典值转NumPy失败: {e_dict_values}");
                        continue
            else:
                logging.warning(f"样本ID {sample_id} 解析后JSON非列表或字典 ({type(sample_data_json)})")
                continue

            if raw_data_np is None or raw_data_np.size == 0:
                logging.warning(f"样本ID {sample_id} 原始数据 (raw_data_np) 为空")
                continue
            logging.debug(
                f"样本ID {sample_id}: 最终图例名 '{determined_legend_text}', 原始数据形状 {raw_data_np.shape}")

            data_for_before = raw_data_np.copy()
            if data_for_before.ndim == 0:
                data_for_before = data_for_before.reshape(1, 1)
            elif data_for_before.ndim == 1:
                data_for_before = data_for_before.reshape(-1, 1)

            if data_for_before.ndim <= 2 and data_for_before.size > 0:
                all_before_process_sample_details.append({
                    'legend_text': determined_legend_text,
                    'data_array': data_for_before
                })
            else:
                logging.warning(
                    f"样本ID {sample_id} (图例: {determined_legend_text}) 处理前数据维度 ({data_for_before.shape}) 不合适或为空，不添加到 'beforeprocess' 列表。")

            feature_data = raw_data_np
            if feature_method:
                logging.info(f"样本ID {sample_id}: 应用特征提取 '{feature_method}'")
                try:
                    if feature_method == "峭度指标":
                        if raw_data_np.ndim > 1:
                            feature_data = [feature_extraction.kurtosis_index(raw_data_np[:, i]) for i in
                                            range(raw_data_np.shape[1])]
                            feature_data = np.concatenate(feature_data, axis=1)  # shape: (N, num_channels)
                        else:
                            feature_data = feature_extraction.kurtosis_index(raw_data_np)  # shape: (N, 1)

                    elif feature_method == "直方图特征":
                        if raw_data_np.ndim > 1:
                            feature_list = [feature_extraction.histogram_feature(raw_data_np[:, i],
                                                                                 bins=feature_params.get("bins", 10))
                                            for i in range(raw_data_np.shape[1])]
                            if feature_list and all(isinstance(f, np.ndarray) and f.ndim == 1 for f in feature_list):
                                feature_data = np.stack(feature_list,
                                                        axis=1)  # Stack as columns -> (feature_len, num_cols)
                            else:  # Fallback or error
                                logging.warning(f"样本ID {sample_id}: 直方图特征返回非预期维度或空列表，尝试展平连接。")
                                try:
                                    feature_data = np.concatenate([np.array(f).flatten() for f in feature_list],
                                                                  axis=0) if feature_list else np.array([])
                                except:
                                    feature_data = np.array([])
                        else:
                            feature_data = feature_extraction.histogram_feature(raw_data_np,
                                                                                bins=feature_params.get("bins", 10))
                    elif feature_method == "傅里叶变换":
                        if raw_data_np.ndim > 1:
                            feature_list = [np.abs(feature_extraction.fourier_transform(raw_data_np[:, i])).flatten()
                                            for i in range(raw_data_np.shape[1])]
                            if feature_list and all(isinstance(f, np.ndarray) and f.ndim == 1 for f in feature_list):
                                feature_data = np.stack(feature_list, axis=1)  # Stack as columns
                            else:
                                feature_data = np.array([])
                        else:
                            feature_data = np.abs(feature_extraction.fourier_transform(raw_data_np)).flatten()
                    elif feature_method == "小波变换":
                        wavelet_name = feature_params.get("wavelet_name", "db4")
                        if raw_data_np.ndim > 1:
                            feature_list = [np.array(
                                feature_extraction.wavelet_transform(raw_data_np[:, i], wavelet_name)).flatten() for i
                                            in range(raw_data_np.shape[1])]
                            if feature_list and all(isinstance(f, np.ndarray) and f.ndim == 1 for f in feature_list):
                                feature_data = np.stack(feature_list, axis=1)  # Stack as columns
                            else:
                                feature_data = np.array([])
                        else:
                            feature_data = np.array(
                                feature_extraction.wavelet_transform(raw_data_np, wavelet_name))
                    elif feature_method == "经验模态分解":
                        max_imfs_str = feature_params.get("max_imfs")
                        max_imfs = None
                        if isinstance(max_imfs_str, str):
                            if max_imfs_str.lower() == 'none':
                                max_imfs = None
                            else:
                                try:
                                    max_imfs = int(max_imfs_str)
                                except ValueError:
                                    logging.warning(f"max_imfs '{max_imfs_str}' 转整数失败.")
                        elif isinstance(max_imfs_str, (int, float)):
                            max_imfs = int(max_imfs_str)
                        sift_thresh = feature_params.get("sift_thresh", 1e-8)
                        if raw_data_np.ndim > 1:
                            # Each f_i is expected to be 2D (IMFs, samples)
                            feature_list = [np.array(
                                feature_extraction.empirical_mode_decomposition(raw_data_np[:, i], max_imfs,
                                                                                sift_thresh)) for i in
                                range(raw_data_np.shape[1])]
                            if feature_list and all(isinstance(f, np.ndarray) and f.ndim == 2 for f in feature_list):
                                # Stack along a new axis (axis=-1 or axis=2) to get (IMFs, samples, num_cols)
                                try:
                                    feature_data = np.stack(feature_list, axis=-1)
                                except ValueError as sve:  # Handle cases where IMFs might have different N_samples for different cols if not padded
                                    logging.error(
                                        f"样本ID {sample_id}: EMD stacking failed due to varying shapes: {sve}. Storing as list of arrays.")
                                    feature_data = feature_list  # Store as list if stacking fails
                            else:
                                feature_data = np.array([])  # Or handle error appropriately
                        else:
                            # Returns 2D (IMFs, samples)
                            feature_data = np.array(
                                feature_extraction.empirical_mode_decomposition(raw_data_np, max_imfs, sift_thresh))
                    elif feature_method == "Wigner-Ville分布":
                        if raw_data_np.ndim > 1:
                            # Each f_i is expected to be 2D (Time, Frequency)
                            feature_list = [np.array(feature_extraction.wigner_ville_distribution(raw_data_np[:, i]))
                                            for i in range(raw_data_np.shape[1])]
                            if feature_list and all(isinstance(f, np.ndarray) and f.ndim == 2 for f in feature_list):
                                # Stack along a new axis to get (Time, Frequency, num_cols)
                                try:
                                    feature_data = np.stack(feature_list, axis=-1)
                                except ValueError as sve:
                                    logging.error(
                                        f"样本ID {sample_id}: WVD stacking failed due to varying shapes: {sve}. Storing as list of arrays.")
                                    feature_data = feature_list  # Store as list if stacking fails
                            else:
                                feature_data = np.array([])
                        else:
                            # Returns 2D (Time, Frequency)
                            feature_data = np.array(feature_extraction.wigner_ville_distribution(raw_data_np))
                    else:
                        logging.warning(f"未知或不支持的特征提取方法 '{feature_method}'")
                        feature_data = raw_data_np
                except Exception as e_feature:
                    logging.error(f"样本ID {sample_id}: 应用特征提取 '{feature_method}' 失败: {e_feature}")
                    import traceback;
                    traceback.print_exc()
                    feature_data = raw_data_np

            if not isinstance(feature_data, (np.ndarray, list)):  # Allow list for failed stacking
                feature_data = np.array(feature_data)

            # Check size for ndarray, or if it's a list, check if it's non-empty
            is_empty = False
            if isinstance(feature_data, np.ndarray):
                if feature_data.size == 0: is_empty = True
            elif isinstance(feature_data, list):
                if not feature_data: is_empty = True
            elif feature_data is None:  # Should have been caught by conversion to np.array if it was initially None
                is_empty = True

            if is_empty:
                logging.warning(f"样本ID {sample_id} 特征提取 ({feature_method}) 后数据为空")
                continue

            # For logging, handle if feature_data became a list
            log_shape = feature_data.shape if isinstance(feature_data, np.ndarray) else [
                f.shape if isinstance(f, np.ndarray) else 'non-array' for f in feature_data] if isinstance(feature_data,
                                                                                                           list) else "N/A"
            logging.debug(f"样本ID {sample_id}: 特征提取后数据形状 {log_shape}")

            final_data_for_output_and_db = feature_data  # This can be N-Dim NP array or list of arrays

            # --- Store "after process" data (Uses determined_legend_text) ---
            # This part prepares data for the API response.
            # If final_data_for_output_and_db is 3D, data_for_after will also be 3D.
            # The response generation logic might need adjustment for 3D.
            data_for_after = None
            if isinstance(final_data_for_output_and_db, np.ndarray):
                data_for_after = final_data_for_output_and_db.copy()
                if data_for_after.ndim == 0:
                    data_for_after = data_for_after.reshape(1, 1)
                elif data_for_after.ndim == 1:
                    data_for_after = data_for_after.reshape(-1, 1)
            elif isinstance(final_data_for_output_and_db, list):  # Handle list of arrays case
                # For simplicity in response, try to make it a single 2D array if possible or handle it
                # This part might need more sophisticated logic based on how you want to represent list of arrays in response
                logging.warning(
                    f"样本ID {sample_id}: final_data_for_output_and_db is a list. Response generation might be simplified.")
                # Attempt to use the first element if it's representative for shape, or flatten, or specific logic.
                # For now, let's assume if it's a list, it's harder to fit into the current response structure directly.
                # We might just pass it and let the response logic try to handle it, or skip adding to 'afterprocess'.
                # Let's try to make it a 2D array if elements are 1D and of same length for response:
                try:
                    if all(isinstance(item, np.ndarray) and item.ndim == 1 for item in
                           final_data_for_output_and_db) and len(
                        set(item.shape[0] for item in final_data_for_output_and_db)) == 1:
                        data_for_after = np.stack(final_data_for_output_and_db, axis=1)
                    elif all(isinstance(item, np.ndarray) and item.ndim == 2 for item in
                             final_data_for_output_and_db) and len(
                        set(item.shape for item in final_data_for_output_and_db)) == 1:
                        # This would result in 3D, which current response does not handle well by default.
                        # For now, let's just take the first item for response simplicity if it's a list of 2D arrays.
                        data_for_after = final_data_for_output_and_db[0].copy()  # Simplified for now
                        if data_for_after.ndim == 1: data_for_after = data_for_after.reshape(-1, 1)

                    else:  # fallback for list of arrays
                        data_for_after = np.array([])  # Cannot easily represent for current response
                except:
                    data_for_after = np.array([])

            if data_for_after is not None and data_for_after.size > 0 and data_for_after.ndim <= 2:  # Response part best handles <=2D
                all_processed_sample_details.append({
                    'legend_text': determined_legend_text,
                    'data_array': data_for_after  # This is what API response will use
                })
            else:
                logging.warning(
                    f"样本ID {sample_id} (图例: {determined_legend_text}) 最终处理数据维度不合适或为空 (Shape: {data_for_after.shape if data_for_after is not None else 'None'}), 不添加到 'afterprocess' 列表。")

            try:
                cursor.execute(delete_feature_sql, (sample_id,))
                feature_sample_id = str(uuid.uuid4())

                # Prepare for DB: if it's a list of arrays (from failed stack), convert each to list
                if isinstance(final_data_for_output_and_db, list):
                    feature_sample_data_for_db = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in
                                                  final_data_for_output_and_db]
                elif isinstance(final_data_for_output_and_db, np.ndarray):
                    feature_sample_data_for_db = final_data_for_output_and_db.tolist()
                else:  # Should not happen
                    feature_sample_data_for_db = final_data_for_output_and_db

                params_json_for_db = None
                if feature_method:
                    try:
                        serializable_params = {k: str(v) if isinstance(v, (type(None), np.generic, np.ndarray)) else v
                                               for k, v in feature_params.items()}
                        params_json_for_db = json.dumps(serializable_params, ensure_ascii=False)
                    except Exception as e_serial:
                        logging.error(f"样本ID {sample_id}: 特征参数JSON序列化失败: {e_serial}. 参数: {feature_params}")
                        params_json_for_db = json.dumps({"error": "Serialization failed"})

                        # Wrap the data in a dictionary with a "data" key
                data_to_store_in_db = {"data": feature_sample_data_for_db}
                data_json_db = json.dumps(data_to_store_in_db,
                                          ensure_ascii=False)  # Store potentially multi-dimensional data

                # 执行插入操作
                cursor.execute(insert_sql, (
                    feature_sample_id, sample_id, sample_type,
                    feature_extract_field_value, params_json_for_db, '无',
                    data_json_db, current_user, datetime.datetime.now(),
                ))
                logging.info(f"样本ID {sample_id} 的特征数据已成功准备好插入 (from_sample_id = {sample_id})。")
            except Exception as e_db_prep:
                logging.error(f"样本ID {sample_id} 数据库操作准备或执行失败: {e_db_prep}")
                import traceback;
                traceback.print_exc()
                if conn: conn.rollback()
                raise

        conn.commit()
        logging.info("所有成功处理的样本特征数据已提交到数据库。")

        before_process_response_list = []
        if all_before_process_sample_details:
            for detail in all_before_process_sample_details:
                arr = detail['data_array']
                current_legend = detail['legend_text']
                if arr.size == 0: continue
                n_rows, num_cols = arr.shape[0], arr.shape[1]
                for col_idx in range(num_cols):
                    y_data_list = arr[:, col_idx].tolist()
                    before_process_response_list.append({
                        "label": "处理前数据",
                        "legendData": [current_legend + (f"_{col_idx + 1}" if num_cols > 1 else "")],
                        "x": list(range(n_rows)), "y": y_data_list
                    })

        after_process_response_list = []
        if all_processed_sample_details:  # This uses the potentially simplified/2D data_for_after
            max_components = 0
            valid_arrays_for_cols = [d['data_array'] for d in all_processed_sample_details if
                                     d['data_array'].size > 0 and d['data_array'].ndim >= 1]
            if valid_arrays_for_cols:
                max_components = max((arr.shape[1] if arr.ndim == 2 else 1) for arr in valid_arrays_for_cols)
            max_components = max(1, max_components)

            for j in range(max_components):
                comp_label = f"分量{j + 1}"
                for detail in all_processed_sample_details:
                    arr = detail['data_array']
                    if arr.size > 0 and arr.ndim >= 1:
                        current_cols = arr.shape[1] if arr.ndim == 2 else 1
                        if j < current_cols:
                            y_data_list = arr[:,
                                          j].tolist() if arr.ndim == 2 else arr.flatten().tolist()  # if 1D (N,1) arr[:,0] else flatten
                            n_rows = arr.shape[0]
                            after_process_response_list.append({
                                "label": comp_label,
                                "legendData": [detail['legend_text'] + (f"_{j + 1}" if max_components > 1 else "")],
                                "x": list(range(n_rows)), "y": y_data_list
                            })
        msg_out = "处理成功"
        if not sample_ids:
            msg_out = "请求样本列表为空。"
        elif not all_processed_sample_details and sample_ids:
            msg_out = "未成功处理任何样本以生成用于响应的特征数据。"

        # 发送处理完成的WebSocket消息
        if room_id:
            emit_process_completed(room_id, 'feature_extraction', {
                'status': 'completed',
                'message': f'特征提取完成，共处理 {processed_count} 个样本',
                'processed_samples': processed_count,
                'total_samples': total_samples,
                'successful_samples': len(all_processed_sample_details),
                'feature_method': feature_method
            })

        return jsonify({
            "state": 200, "message": msg_out,
            "data": {"beforeprocess": before_process_response_list, "afterprocess": after_process_response_list}
        }), 200
    except psycopg2.Error as db_err:
        if conn: conn.rollback()
        logging.error(f"数据库操作失败: {db_err}", exc_info=True)

        # 发送错误消息
        if room_id:
            emit_process_error(room_id, 'feature_extraction', f"数据库操作失败: {str(db_err)}")

        return jsonify({"state": 500, "message": f"数据库错误: {str(db_err)}"}), 500
    except json.JSONDecodeError as json_err:
        logging.error(f"请求体JSON解析失败: {json_err}", exc_info=True)

        # 发送错误消息
        if room_id:
            emit_process_error(room_id, 'feature_extraction', f"请求数据JSON格式错误: {str(json_err)}")

        return jsonify({"state": 400, "message": f"请求数据JSON格式错误: {str(json_err)}"}), 400
    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"特征确认过程发生未知错误: {e}", exc_info=True)

        # 发送错误消息
        if room_id:
            emit_process_error(room_id, 'feature_extraction', f"处理失败: {str(e)}")

        return jsonify({"state": 500, "message": f"处理失败: {str(e)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route('/api/analysis/train/feature/param/get', methods=['GET'])
@token_required
def get_feature_extraction_params():
    """
    获取所有可用的特征提取方法及其可配置参数

    Returns:
        JSON 格式的特征提取方法及其参数列表
    """
    # 定义所有可用的特征提取方法及其参数
    feature_extraction_methods = {
        "峭度指标": {
            "name": "峭度指标",
            "description": "计算输入数据每个特征的峰度（Kurtosis Index）",
            "parameters": {}  # 没有可配置参数
        },
        "直方图特征": {
            "name": "直方图特征",
            "description": "计算输入数据每个特征的直方图特征",
            "parameters": {
                "bins": {
                    "type": "int",
                    "default": 10,
                    "min": 2,
                    "max": 100,
                    "description": "直方图的区间数"
                }
            }
        },
        "傅里叶变换": {
            "name": "傅里叶变换",
            "description": "计算输入数据每个特征的傅里叶变换",
            "parameters": {}  # 没有可配置参数
        },
        "小波变换": {
            "name": "小波变换",
            "description": "对输入信号进行小波变换",
            "parameters": {
                "wavelet_name": {
                    "type": "select",
                    "default": "db4",
                    "options": ["haar", "db1", "db2", "db3", "db4", "db5", "sym2", "sym3", "sym4", "coif1", "coif2",
                                "bior1.1", "bior1.3", "bior2.2", "bior2.4"],
                    "description": "小波基的名称"
                },
                "level": {
                    "type": "int",
                    "default": None,
                    "min": 1,
                    "max": 10,
                    "nullable": True,
                    "description": "分解的层数，None表示自动确定最大层数"
                }
            }
        },
        "经验模态分解": {
            "name": "经验模态分解",
            "description": "对输入信号进行经验模态分解",
            "parameters": {
                "max_imfs": {
                    "type": "int",
                    "default": None,
                    "min": 1,
                    "max": 20,
                    "nullable": True,
                    "description": "要提取的最大本征模态函数（IMF）数量，None表示提取尽可能多的IMF"
                },
                "sift_thresh": {
                    "type": "float",
                    "default": 1e-8,
                    "min": 1e-12,
                    "max": 1e-4,
                    "description": "筛选过程的终止阈值"
                },
                "max_iters": {
                    "type": "int",
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "description": "筛选过程的最大迭代次数"
                }
            }
        },
        "Wigner-Ville分布": {
            "name": "Wigner-Ville分布",
            "description": "计算输入信号的Wigner-Ville分布",
            "parameters": {}  # 没有可配置参数
        }
    }

    # 特征选择方法
    feature_selection_methods = {
        "Fisher判别法": {
            "name": "Fisher判别法",
            "description": "使用Fisher判别法进行特征选择，需要标签数据",
            "parameters": {}  # 没有额外参数
        },
        "Relief算法": {
            "name": "Relief算法",
            "description": "使用Relief算法进行特征选择，需要标签数据",
            "parameters": {
                "num_neighbors": {
                    "type": "int",
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "description": "邻居数量"
                }
            }
        }
    }

    return jsonify({
        "state": 200,
        "data": {
            "success": True,
            "message": "获取特征提取参数成功",
            "feature_extraction_methods": feature_extraction_methods,
            "feature_selection_methods": feature_selection_methods
        }
    })


@app.route('/api/analysis/train/feature/param/save', methods=['POST'])
@token_required
def save_feature_extraction_params():
    """
    保存特征提取参数设置到数据库

    Request Body:
        feature_sample_id: 样本标识，可以为空
        feature_extract: 特征提取方法名称
        feature_extract_param: 特征提取参数JSON

    Returns:
        保存结果的状态
    """
    try:
        data = request.json
        feature_sample_id = data.get('feature_sample_id', '')
        feature_extract = data.get('feature_extract', '')
        feature_extract_param = data.get('feature_extract_param', {})

        # 验证特征提取方法是否存在
        feature_methods = ["峭度指标", "直方图特征", "傅里叶变换", "小波变换", "经验模态分解", "Wigner-Ville分布"]
        if feature_extract not in feature_methods:
            return jsonify({
                "state": 404,
                "data": {
                    "success": False,
                    "message": f"无效的特征提取方法: {feature_extract}"
                }
            })

        # 验证参数合法性
        valid_params = True
        error_message = ""

        # 根据不同特征提取方法验证参数
        if feature_extract == "直方图特征":
            bins = feature_extract_param.get("bins")
            if bins is not None and (not isinstance(bins, int) or bins < 2 or bins > 100):
                valid_params = False
                error_message = "bins参数必须是2-100之间的整数"

        elif feature_extract == "小波变换":
            wavelet_name = feature_extract_param.get("wavelet_name")
            level = feature_extract_param.get("level")
            valid_wavelets = ["haar", "db1", "db2", "db3", "db4", "db5", "sym2", "sym3", "sym4", "coif1", "coif2",
                              "bior1.1", "bior1.3", "bior2.2", "bior2.4"]

            if wavelet_name is not None and wavelet_name not in valid_wavelets:
                valid_params = False
                error_message = f"无效的小波基名称: {wavelet_name}"

            if level is not None and level is not None and (not isinstance(level, int) or level < 1 or level > 10):
                valid_params = False
                error_message = "level参数必须是1-10之间的整数或None"

        elif feature_extract == "经验模态分解":
            max_imfs = feature_extract_param.get("max_imfs")
            sift_thresh = feature_extract_param.get("sift_thresh")
            max_iters = feature_extract_param.get("max_iters")

            if max_imfs is not None and max_imfs is not None and (
                    not isinstance(max_imfs, int) or max_imfs < 1 or max_imfs > 20):
                valid_params = False
                error_message = "max_imfs参数必须是1-20之间的整数或None"

            if sift_thresh is not None and (
                    not isinstance(sift_thresh, (int, float)) or sift_thresh < 1e-12 or sift_thresh > 1e-4):
                valid_params = False
                error_message = "sift_thresh参数必须是1e-12到1e-4之间的浮点数"

            if max_iters is not None and (not isinstance(max_iters, int) or max_iters < 100 or max_iters > 10000):
                valid_params = False
                error_message = "max_iters参数必须是100-10000之间的整数"

        if not valid_params:
            return jsonify({
                "state": 404,
                "data": {
                    "success": False,
                    "message": f"参数验证失败: {error_message}"
                }
            })

        # 获取当前用户（这里使用系统默认值，实际应根据你的认证机制获取）
        current_user = "system"

        # 将参数保存到数据库
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 检查feature_sample_id是否存在
            if feature_sample_id:
                # 查询该ID是否存在于数据库中
                cursor.execute("""
                    SELECT feature_sample_id FROM tb_analysis_sample_feature 
                    WHERE feature_sample_id = %s
                """, (feature_sample_id,))

                existing_record = cursor.fetchone()

                if existing_record:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE tb_analysis_sample_feature
                        SET feature_extract = %s,
                            feature_extract_param = %s
                        WHERE feature_sample_id = %s
                    """, (
                        feature_extract,
                        json.dumps(feature_extract_param),
                        feature_sample_id
                    ))
                else:
                    # feature_sample_id存在但记录不存在，返回错误
                    conn.close()
                    return jsonify({
                        "state": 404,
                        "data": {
                            "success": False,
                            "message": f"找不到ID为{feature_sample_id}的特征样本记录"
                        }
                    })
            else:
                # 生成新的feature_sample_id
                feature_sample_id = str(uuid.uuid4())

                # 创建新记录（部分字段设为空或默认值，因为这只是参数保存阶段）
                cursor.execute("""
                    INSERT INTO tb_analysis_sample_feature (
                        feature_sample_id, feature_extract, feature_extract_param, 
                        create_user, create_time
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    feature_sample_id,
                    feature_extract,
                    json.dumps(feature_extract_param),
                    current_user,
                    datetime.datetime.now()
                ))

            # 提交事务
            conn.commit()

            return jsonify({
                "state": 200,
                "data": {
                    "success": True,
                    "message": "参数保存成功",
                    "feature_sample_id": feature_sample_id,
                    "feature_extract": feature_extract,
                    "feature_extract_param": feature_extract_param
                }
            })

        except Exception as e:
            # 回滚事务
            conn.rollback()
            raise e
        finally:
            # 关闭数据库连接
            conn.close()

    except Exception as e:
        return jsonify({
            "state": 404,
            "data": {
                "success": False,
                "message": f"参数保存失败: {str(e)}"
            }
        })


@app.route('/api/analysis/train/train/sample', methods=['GET'])
@token_required
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


@app.route('/api/analysis/train/train/model', methods=['GET'])
@token_required
def get_model_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"state": 500, "message": "数据库连接失败"}), 500  # 修改状态码

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  # 使用 DictCursor
    query = "SELECT model_id, model_name, model_data FROM tb_analysis_model"

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


# 6.7
@app.route('/api/analysis/train/train/start', methods=['POST'])
@token_required
def start_model_training():
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

# --- **修改后的 run_training_with_websocket 函数：接受 stop_event 并进行清理** ---
def run_training_with_websocket(current_training_instance_id, model_id_from_request, sample_data_json_with_labels,
                                base_model_train_id_for_process, full_param_data, websocket_session_id, stop_event):
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

        logger.info(
            f"开始 {actual_model_name_for_training} 模型训练 (训练实例ID: {current_training_instance_id}, 定义 ID: {model_definition_id})")

        # **调用 train_model 函数，传递 stop_event**
        trained_model_instance, result_dict_from_train = model_function.train_model(
            json_data=sample_data_json_with_labels,
            label_column="label",
            model_name=actual_model_name_for_training,
            param_data=full_param_data,
            training_id=websocket_session_id,
            stop_event=stop_event # **传递 stop_event**
        )

        # 检查训练结果，首先判断是否是用户中止
        is_stopped_by_user = stop_event.is_set()
        is_error = 'error' in result_dict_from_train or '失败' in result_dict_from_train.get('message', '').lower()

        if is_stopped_by_user:
            message = "训练已中止"
            logger.info(f"训练 {current_training_instance_id} 已由用户中止。")
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
        logger.info(f"训练流程结束 (ID: {current_training_instance_id}).")

# --- 新增：中止训练接口 ---
@app.route('/api/analysis/train/train/cancel', methods=['POST'])
@token_required
def stop_model_training():
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
        logger.info(f"收到停止训练请求，训练ID: {training_instance_id}")
        stop_event.set()  # 设置停止事件，向线程发出停止信号
        logger.info(f"停止事件已设置: {stop_event.is_set()}")

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


@app.route('/api/analysis/train/train/model/save', methods=['POST'])
@token_required
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
        cursor.execute("SELECT model_id, model_train_name FROM tb_analysis_model_train WHERE model_train_id = %s",
                       (model_train_id,))
        model_result = cursor.fetchone()
        if not model_result:
            return jsonify({"state": 404, "message": "Model training record not found"}), 404

        model_id = model_result["model_id"]
        model_name = model_result["model_train_name"]

        # **2. 读取训练后的模型参数**
        model_dir = "./saved_models"
        print(f"Model name: {model_name}")  # 打印 model_name
        print(f"Model directory: {model_dir}")  # 打印 model_dir
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name)]
        model_files.sort(reverse=True)  # 按时间降序排序
        print(f"Files in model directory: {os.listdir(model_dir)}")  # 列出目录内容
        if not model_files:
            return jsonify({"state": 404, "message": "Trained model file not found"}), 404

        model_path = os.path.join(model_dir, model_files[0])
        print(f"Model path: {model_path}")  # 打印 model_path
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
            cursor.execute(query_update,
                           (model_train_id, model_name, model_data_json, create_user, create_time, model_id))
        else:
            query_insert = """
            INSERT INTO tb_analysis_model (model_id, model_train_id, model_name, model_data, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query_insert,
                           (model_id, model_train_id, model_name, model_data_json, create_user, create_time))

        conn.commit()
        return jsonify({"state": 200, "data": {"success": "true", "message": "Model saved successfully"}}), 200

    except psycopg2.Error as e:
        print(f"Error saving model: {e}")
        return jsonify({"state": 500, "message": "Failed to save model"}), 500

    finally:
        cursor.close()
        conn.close()


@app.route('/api/analysis/train/train/model/download', methods=['GET'])
@token_required
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


@app.route('/api/analysis/train/train/param/get', methods=['GET'])
@token_required
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


@app.route('/api/analysis/train/train/param/save', methods=['POST'])
@token_required
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


# 5.12新增AI模型应用接口：
@app.route('/api/analysis/apply/gather/import', methods=['POST'])
@token_required
def import_apply_file():
    """
    数据文件导入接口 (用于分析应用模块)
    - 参数：
      - file: 上传的文件 (支持 xls, xlsx, mat, txt 类型)
    - 响应：
      - 成功返回文件标识符 (file_id)
      - 失败返回错误信息
    """
    conn = None  # 初始化 conn 为 None
    cursor = None  # 初始化 cursor 为 None
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"state": 400, "message": "未提供上传文件"}), 400

        file = request.files['file']
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({"state": 400, "message": "未选择文件"}), 400

        # 生成文件信息
        file_id = str(uuid.uuid4())
        file_name = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        # 保存文件
        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"state": 500, "message": f"文件保存失败: {str(e)}"}), 500

        # 插入数据库
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        try:
            cursor = conn.cursor()
            sql = """
            INSERT INTO tb_analysis_apply_file (file_id, file_name, file_path, demo, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            # 这里的 "自动上传" 和 "admin" 可以根据实际业务需求动态获取
            cursor.execute(sql, (file_id, file_name, file_path, "自动上传", "admin", datetime.datetime.now()))
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()  # 数据库操作失败时回滚
            return jsonify({"state": 500, "message": f"数据库操作失败: {str(e)}"}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        return jsonify({"state": 200, "data": {"file_id": file_id}}), 200

    except Exception as e:
        # 捕获所有其他未知异常
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500


# 5.12新增AI模型应用接口：
# 这个接口不懂什么意思？
@app.route('/api/analysis/apply/gather/check', methods=['GET'])
@token_required
def check_apply_file():
    """
    在线检查样本数据接口。
    根据 file_id 检查文件是否存在于数据库中以及文件是否在服务器上。

    参数:
      - file_id (str): 文件标识符

    返回:
      JSON: 包含检查结果的JSON响应
    """
    conn = None
    cursor = None
    try:
        file_id = request.args.get('file_id')

        if not file_id:
            return jsonify({
                "state": 400,
                "data": {
                    "success": "false",
                    "message": "未提供文件标识符"
                }
            }), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({
                "state": 500,
                "data": {
                    "success": "false",
                    "message": "数据库连接失败"
                }
            }), 500

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 1. 检查文件ID是否存在于数据库中
        sql_query = "SELECT file_path FROM tb_analysis_apply_file WHERE file_id = %s"
        cursor.execute(sql_query, (file_id,))
        result = cursor.fetchone()

        if not result:
            return jsonify({
                "state": 404,
                "data": {
                    "success": "false",
                    "message": f"文件标识符 '{file_id}' 不存在于数据库中。"
                }
            }), 404  # 根据图片要求，返回 404 表示失败

        file_path = result['file_path']

        # 2. 检查文件是否存在于服务器文件系统中
        if not os.path.exists(file_path):
            return jsonify({
                "state": 404,
                "data": {
                    "success": "false",
                    "message": f"文件 '{file_path}' 在服务器上不存在或已被移动。"
                }
            }), 404  # 根据图片要求，返回 404 表示失败

        # 3. (可选) 检查文件是否为空或损坏，这里只进行简单的大小检查
        if os.path.getsize(file_path) == 0:
            return jsonify({
                "state": 400,  # 文件内容问题，可以返回 400 Bad Request
                "data": {
                    "success": "false",
                    "message": f"文件 '{file_path}' 为空，无有效数据。"
                }
            }), 400

        # 所有检查通过
        return jsonify({
            "state": 200,
            "data": {
                "success": "true",
                "message": "文件数据检查通过，文件有效且可访问。"
            }
        }), 200

    except Exception as e:
        # 捕获其他任何服务器内部错误
        return jsonify({
            "state": 500,
            "data": {
                "success": "false",
                "message": f"服务器内部错误: {str(e)}"
            }
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# 5.12新增AI模型应用接口：
@app.route('/api/analysis/apply/gather/sample', methods=['GET'])
@token_required
def get_apply_sample_data():
    """
    根据文件标识获取样本数据列表 (用于分析应用模块)
    - 参数：
      - file_id (str): 文件标识符
    - 响应：
      - 成功返回样本标识符和名称列表
      - 失败返回错误信息
    """
    file_id = request.args.get('file_id')
    create_user = request.args.get('create_user', 'admin')  # 可以从请求头或session获取实际用户

    if not file_id:
        return jsonify({"state": 400, "message": "未提供文件标识符"}), 400

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 1. 查询是否已存在该 file_id 的样本数据 (tb_analysis_apply_sample)
        cursor.execute("SELECT apply_sample_id, sample_name FROM tb_analysis_apply_sample WHERE file_id = %s",
                       (file_id,))
        existing_samples = cursor.fetchall()

        if existing_samples:
            # 如果已存在，直接返回现有数据
            data_to_return = [{"sample_id": sample['apply_sample_id'], "sample_name": sample['sample_name']}
                              for sample in existing_samples]
            return jsonify({
                "state": 200,
                "data": data_to_return,
                "message": f"文件标识符 '{file_id}' 的样本数据已存在，已返回现有数据。"
            }), 200
        else:
            # 2. 查询文件路径 (tb_analysis_apply_file)
            cursor.execute("SELECT file_path FROM tb_analysis_apply_file WHERE file_id = %s", (file_id,))
            result = cursor.fetchone()

            if not result:
                return jsonify({"state": 404, "message": "未找到对应的文件标识符"}), 404

            file_path = result['file_path']
            if not os.path.exists(file_path):
                return jsonify({"state": 404, "message": "文件在服务器上不存在"}), 404

            # 3. 读取文件
            df = None
            try:
                # 根据文件扩展名选择合适的读取函数
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.csv', '.txt']:
                    df = pd.read_csv(file_path)
                elif file_ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(file_path)
                # TODO: 对于 .mat 文件，需要使用 scipy.io.loadmat
                # TODO: 对于纯 .txt 文件（非CSV格式），可能需要更复杂的解析逻辑
                else:
                    return jsonify({"state": 400, "message": f"不支持的文件类型: {file_ext}"}), 400

            except FileNotFoundError:
                return jsonify({"state": 404, "message": "文件未找到"}), 404
            except (pd.errors.ParserError, pd.errors.EmptyDataError, Exception) as e:
                # 捕获Pandas解析错误，例如CSV格式不正确或文件为空
                return jsonify({"state": 400, "message": f"文件解析失败，请检查文件格式: {str(e)}"}), 400

            if df.empty:
                return jsonify({"state": 400, "message": "文件内容为空，没有可用的样本数据"}), 400

            # 4. 遍历列，生成样本数据并准备插入
            sample_list_to_insert = []
            data_to_return = []
            file_uuid_prefix = str(uuid.uuid4())  # 为当前文件解析批次生成一个 UUID 前缀

            for i, col_name in enumerate(df.columns):
                # 生成唯一的 apply_sample_id
                apply_sample_id = f"{file_uuid_prefix}_{i + 1}"

                # 获取列数据并转换为 JSON 字符串
                sample_data_list = df[col_name].tolist()
                sample_data_json = json.dumps(sample_data_list)

                data_to_return.append({
                    "sample_id": apply_sample_id,
                    "sample_name": col_name
                })

                sample_list_to_insert.append((
                    apply_sample_id, file_id, col_name, sample_data_json, '1', create_user, datetime.datetime.now()
                ))

            # 5. 批量插入数据到 tb_analysis_apply_sample
            insert_query = """
            INSERT INTO tb_analysis_apply_sample (
                apply_sample_id, file_id, sample_name, sample_data, sample_state, create_user, create_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cursor.executemany(insert_query, sample_list_to_insert)
            conn.commit()

            return jsonify({
                "state": 200,
                "data": data_to_return
            }), 200

    except psycopg2.Error as e:
        if conn:
            conn.rollback()  # 数据库操作失败时回滚
        return jsonify({"state": 500, "message": f"PostgreSQL 操作失败: {str(e)}"}), 500
    except Exception as e:
        if conn:
            conn.rollback()  # 确保在其他异常时也回滚
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# 5.12新增AI模型应用接口：
@app.route('/api/analysis/apply/gather/wave', methods=['GET'])
@token_required
def get_apply_wave_data():
    """
    根据样本标识获取样本数据列表 (用于分析应用模块的波形图展示)
    - 参数：
      - sample_ids (str): 样本标识符，逗号分隔 (例如: "sample_id_1,sample_id_2")
    - 响应：
      - 成功返回样本标识、名称、样本数据列表
      - 失败返回错误信息
    """
    sample_ids_param = request.args.get('sample_ids')

    if not sample_ids_param:
        return jsonify({"state": 400, "message": "未提供样本 ID"}), 400

    # 清洗参数并分割为列表
    sample_ids_list = [sid.strip() for sid in sample_ids_param.split(',') if sid.strip()]
    if not sample_ids_list:
        return jsonify({"state": 400, "message": "样本 ID 格式无效"}), 400

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        # 使用 DictCursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 构建 IN 子句的占位符字符串
        placeholders = ','.join(['%s'] * len(sample_ids_list))

        # 查询 tb_analysis_apply_sample 表
        query = f"""
        SELECT apply_sample_id, sample_name, sample_data
        FROM tb_analysis_apply_sample
        WHERE apply_sample_id IN ({placeholders})
        """

        cursor.execute(query, tuple(sample_ids_list))
        results = cursor.fetchall()

        if not results:
            return jsonify({
                "state": 404,  # 根据图片要求，404表示失败（未找到）
                "message": "未找到匹配的样本数据"
            }), 404

        # 构建响应数据
        response_data = []
        for row in results:
            try:
                # 尝试解析 sample_data 字段的 JSON 字符串为 Python 列表
                sample_data_parsed = json.loads(row["sample_data"])
                response_data.append({
                    "sample_id": row["apply_sample_id"],
                    "sample_name": row["sample_name"],
                    "sample_data": sample_data_parsed
                })
            except json.JSONDecodeError as e:
                # 如果 sample_data 不是有效的 JSON 字符串，则跳过或记录错误
                print(f"样本 {row['apply_sample_id']} 的数据解析失败: {e}")
                continue  # 或者可以返回一个错误信息，取决于业务需求

        if not response_data:
            # 如果所有数据都解析失败，则返回未找到
            return jsonify({
                "state": 404,
                "message": "未能成功解析任何样本数据"
            }), 404

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        # 捕获其他任何服务器内部错误
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# 5.12新增AI模型应用接口：
@app.route('/api/analysis/apply/check/model', methods=['GET'])
@token_required
def get_apply_models():
    """
    获取模型数据列表 (用于分析应用模块的模型选择)
    - 参数：无
    - 响应：
      - 成功返回模型标识、名称、模型数据列表
      - 失败返回错误信息 (例如 404 表示未找到任何模型)
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        # 使用 DictCursor 使得查询结果可以通过字段名访问
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 查询 tb_analysis_model 表，获取 model_id, model_name, model_data
        cursor.execute("SELECT model_id, model_name, model_data FROM tb_analysis_model")
        models = cursor.fetchall()  # 获取所有查询结果

        if not models:
            # 根据图片要求，如果未找到任何模型，返回 state 404
            return jsonify({"state": 404, "data": [], "message": "未找到任何模型数据"}), 404

        # Flask 的 jsonify 通常可以直接处理 psycopg2.extras.DictRow 对象列表
        # 如果遇到问题，可以手动转换为列表字典: [dict(model) for model in models]
        return jsonify({"state": 200, "data": models}), 200

    except Exception as e:
        # 捕获任何服务器内部错误
        print(f"获取模型数据失败: {str(e)}")  # 打印错误以便调试
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# /api/analysis/apply/check/model：这个逻辑据复杂，后面再说


# 5.12新增AI模型应用接口：
@app.route('/api/analysis/apply/check/add/sample', methods=['POST'])
@token_required
def add_apply_sample_to_library():
    """
    将样本数据提交至样本库接口 (用于分析应用模块)
    根据请求的 JSON body 中的 sample_ids，将选中的样本状态更新为 '2'，表示已加入样本库。

    参数 (JSON Body):
      - sample_ids (str): 样本标识符，逗号分隔 (例如: "sample_id_1,sample_id_2")

    返回:
      JSON: 包含操作结果的JSON响应
    """
    conn = None
    cursor = None
    try:
        # 获取 JSON 请求体数据
        request_data = request.get_json()
        sample_ids_param = request_data.get('sample_ids')

        if not sample_ids_param:
            return jsonify({
                "state": 400,
                "data": {
                    "success": "false",
                    "message": "未提供样本 ID"
                }
            }), 400

        # 解析逗号分隔的 sample_ids 字符串为列表
        sample_ids_list = [sid.strip() for sid in sample_ids_param.split(',') if sid.strip()]

        if not sample_ids_list:
            return jsonify({
                "state": 400,
                "data": {
                    "success": "false",
                    "message": "样本 ID 格式无效"
                }
            }), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({
                "state": 500,
                "data": {
                    "success": "false",
                    "message": "数据库连接失败"
                }
            }), 500

        cursor = conn.cursor()

        # **1. 检查数据库中是否存在这些样本 ID**
        placeholders = ', '.join(['%s'] * len(sample_ids_list))
        check_sql = f"""
            SELECT apply_sample_id FROM tb_analysis_apply_sample
            WHERE apply_sample_id IN ({placeholders})
        """
        cursor.execute(check_sql, tuple(sample_ids_list))
        existing_samples = {row[0] for row in cursor.fetchall()}  # 获取存在的 apply_sample_id

        if not existing_samples:
            return jsonify({
                "state": 404,  # 根据图片要求，404表示失败
                "data": {
                    "success": "false",
                    "message": "没有匹配的样本 ID，无法更新"
                }
            }), 404

        # **2. 过滤掉无效的 sample_ids，只更新存在的有效样本**
        valid_sample_ids_to_update = list(existing_samples)

        if not valid_sample_ids_to_update:
            return jsonify({
                "state": 400,
                "data": {
                    "success": "false",
                    "message": "提供的样本 ID 无效或无需更新，请检查"
                }
            }), 400

        # **3. 更新数据库中有效的样本的 sample_state 为 '2'**
        update_placeholders = ', '.join(['%s'] * len(valid_sample_ids_to_update))
        update_sql = f"""
            UPDATE tb_analysis_apply_sample
            SET sample_state = '2'
            WHERE apply_sample_id IN ({update_placeholders}) AND sample_state != '2'
        """
        cursor.execute(update_sql, tuple(valid_sample_ids_to_update))
        conn.commit()

        updated_row_count = cursor.rowcount
        if updated_row_count > 0:
            message = f"成功更新 {updated_row_count} 条样本数据至样本库。"
            success_status = "true"
            http_status = 200
        else:
            # 如果没有行被更新，可能是因为所有样本都已经处于状态 '2'
            message = "样本已在样本库中，无需重复添加或未找到待更新样本。"
            success_status = "false"
            http_status = 200  # 视为成功响应，但数据层面无更新

        return jsonify({
            "state": http_status,
            "data": {
                "success": success_status,
                "message": message
            }
        }), http_status

    except Exception as e:
        if conn:
            conn.rollback()  # 发生异常时回滚数据库操作
        print(f"添加样本至样本库失败: {str(e)}")  # 打印错误以便调试
        return jsonify({
            "state": 500,
            "data": {
                "success": "false",
                "message": f"服务器内部错误: {str(e)}"
            }
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)

