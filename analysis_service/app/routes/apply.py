# app/routes/apply.py
import math
from collections import Counter
import numpy as np
from flask import Blueprint, request, jsonify, session
import os
import datetime
import json
import uuid
import pandas as pd
from psycopg2 import Error
import psycopg2
import psycopg2.extras

from analysis_service import feature_extraction
from analysis_service.methods import Preprocessor
from analysis_service.model_function import create_windows_1d, create_windows_wavelet, GenericDNN
from auth import token_required
from config import UPLOAD_FOLDER
from analysis_service.app.db import get_db_connection
from analysis_service.app.extensions import logger
from analysis_service.app.websocket import (
    emit_process_progress,
    emit_process_completed,
    emit_process_error,
    emit_process_result,
)
import joblib
import torch

apply_bp = Blueprint("apply", __name__)

# 对应原来 @app.route('/api/analysis/apply/gather/import', methods=['POST'])
@apply_bp.route('/gather/import', methods=['POST'])
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

# 对应原来 @app.route('/api/analysis/apply/gather/sample', methods=['GET'])
@apply_bp.route('/gather/sample', methods=['GET'])
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
            print(f"DEBUG: File ID '{file_id}' sample data already exists, returning existing data.")
            return jsonify({
                "state": 200,
                "data": data_to_return,
                "message": f"文件标识符 '{file_id}' 的样本数据已存在，已返回现有数据。"
            }), 200
        else:
            print(f"DEBUG: No existing sample data for file ID '{file_id}', proceeding to process file.")
            # 2. 查询文件路径 (tb_analysis_apply_file)
            cursor.execute("SELECT file_path FROM tb_analysis_apply_file WHERE file_id = %s", (file_id,))
            result = cursor.fetchone()

            if not result:
                print(f"ERROR: File ID '{file_id}' not found in tb_analysis_apply_file.")
                return jsonify({"state": 404, "message": "未找到对应的文件标识符"}), 404

            file_path = result['file_path']
            if not os.path.exists(file_path):
                print(f"ERROR: File does not exist on server: {file_path}")
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
                    print(f"ERROR: Unsupported file type: {file_ext} for file {file_path}")
                    return jsonify({"state": 400, "message": f"不支持的文件类型: {file_ext}"}), 400

                # --- DEBUG POINT 1: 刚读取文件后检查 DataFrame 的状态 ---
                print("\n--- DEBUG POINT 1: After pd.read_csv/excel ---")
                print(f"File Path: {file_path}")
                print("DataFrame Head:")
                print(df.head().to_string()) # 使用 .to_string() 避免截断
                print("\nDataFrame dtypes (IMPORTANT!):")
                print(df.dtypes.to_string())
                print("\nDataFrame info (check non-null counts):")
                df.info(verbose=True, show_counts=True) # show_counts=True 显示非空值数量
                print("\nNaN count per column (VERY IMPORTANT!):")
                print(df.isnull().sum().to_string())
                print("--- END DEBUG POINT 1 ---")

            except FileNotFoundError:
                print(f"ERROR: File not found during read attempt: {file_path}")
                return jsonify({"state": 404, "message": "文件未找到"}), 404
            except (pd.errors.ParserError, pd.errors.EmptyDataError, Exception) as e:
                print(f"ERROR: File parsing failed for {file_path}: {str(e)}")
                # 捕获Pandas解析错误，例如CSV格式不正确或文件为空
                return jsonify({"state": 400, "message": f"文件解析失败，请检查文件格式: {str(e)}"}), 400

            if df.empty:
                print(f"WARNING: File content is empty for {file_path}.")
                return jsonify({"state": 400, "message": "文件内容为空，没有可用的样本数据"}), 400

            # 4. 遍历列，生成样本数据并准备插入
            sample_list_to_insert = []
            data_to_return = []
            file_uuid_prefix = str(uuid.uuid4())  # 为当前文件解析批次生成一个 UUID 前缀

            for i, col_name in enumerate(df.columns):
                # 生成唯一的 apply_sample_id
                apply_sample_id = f"{file_uuid_prefix}_{i + 1}"

                # --- DEBUG POINT 2: 获取 tolist() 之前的 Series 状态 ---
                current_series = df[col_name]
                print(f"\n--- DEBUG POINT 2: Column '{col_name}' (ID: {apply_sample_id}) before .tolist() ---")
                print(f"Series dtype: {current_series.dtype}")
                if current_series.isnull().any():
                    print(f"WARNING: Series contains NaN values (count: {current_series.isnull().sum()})")
                    print(f"Indices of NaNs: {current_series[current_series.isnull()].index.tolist()}")
                    # 打印 NaN 周围的值，以便您能检查原始文件对应位置
                    nan_indices = current_series[current_series.isnull()].index
                    for idx in nan_indices:
                        start_idx = max(0, idx - 5)
                        end_idx = min(len(current_series), idx + 6)
                        print(f"  Values around NaN at index {idx}: {current_series.iloc[start_idx:end_idx].tolist()}")
                else:
                    print("Series contains NO NaN values at this stage.")
                print("--- END DEBUG POINT 2 ---")


                sample_data_list = current_series.tolist() # <-- 关键转换点

                # --- DEBUG POINT 3: tolist() 之后，json.dumps() 之前 ---
                print(f"\n--- DEBUG POINT 3: Column '{col_name}' (ID: {apply_sample_id}) after .tolist() ---")
                print(f"List length: {len(sample_data_list)}")
                has_none = any(x is None for x in sample_data_list)
                has_float_nan = any(isinstance(x, float) and math.isnan(x) for x in sample_data_list)
                print(f"List contains Python 'None': {has_none}")
                print(f"List contains Python 'float('nan')': {has_float_nan}")
                if has_none or has_float_nan:
                    print("WARNING: The list now contains None or float('nan')!")
                    # 打印 problematic values
                    problem_elements = [(j, val) for j, val in enumerate(sample_data_list) if val is None or (isinstance(val, float) and math.isnan(val))]
                    for j, val in problem_elements[:10]: # 只打印前10个问题元素
                        print(f"  Problematic value at index {j}: {val}. Type: {type(val)}")
                        print(f"  Values around index {j}: {sample_data_list[max(0, j-5):min(len(sample_data_list), j+6)]}")
                else:
                    print("List contains NO None or float('nan') values at this stage.")
                print("--- END DEBUG POINT 3 ---")

                sample_data_json = json.dumps(sample_data_list) # <-- 另一个关键转换点

                # --- DEBUG POINT 4: json.dumps() 之后，入库之前 ---
                print(f"\n--- DEBUG POINT 4: Column '{col_name}' (ID: {apply_sample_id}) after json.dumps() ---")
                has_json_null = "null" in sample_data_json
                print(f"JSON string contains 'null' literal: {has_json_null}")
                if has_json_null:
                    print("WARNING: The JSON string now contains 'null'!")
                    # 打印 JSON 字符串的片段
                    print(f"Sample JSON snippet (first 500 chars): {sample_data_json[:500]}...")
                else:
                    print("JSON string contains NO 'null' literal at this stage.")
                print("--------------------------------------------------")
                # --- END DEBUG POINT 4 ---


                data_to_return.append({
                    "sample_id": apply_sample_id,
                    "sample_name": col_name
                })

                sample_list_to_insert.append((
                    apply_sample_id, file_id, col_name, sample_data_json, '1', create_user, datetime.datetime.now()
                ))

            # 5. 批量插入数据到 tb_analysis_apply_sample (注意这里已经更正为 tb_analysis_apply_sample)
            insert_query = """
            INSERT INTO tb_analysis_apply_sample (
                apply_sample_id, file_id, sample_name, sample_data, sample_state, create_user, create_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cursor.executemany(insert_query, sample_list_to_insert)
            conn.commit()
            print(f"DEBUG: Successfully inserted {len(sample_list_to_insert)} samples into tb_analysis_apply_sample for file ID '{file_id}'.")

            return jsonify({
                "state": 200,
                "data": data_to_return
            }), 200

    except psycopg2.Error as e:
        if conn:
            conn.rollback()  # 数据库操作失败时回滚
        print(f"ERROR: PostgreSQL operation failed: {str(e)}")
        return jsonify({"state": 500, "message": f"PostgreSQL 操作失败: {str(e)}"}), 500
    except Exception as e:
        if conn:
            conn.rollback()  # 确保在其他异常时也回滚
        print(f"FATAL ERROR: Server internal error: {str(e)}")
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("DEBUG: Database connection closed.")

# 对应原来  @app.route('/api/analysis/apply/gather/wave', methods=['GET'])
@apply_bp.route('/gather/wave', methods=['GET'])
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

# 对应原来 @app.route('/api/analysis/apply/gather/files', methods=['GET'])
@apply_bp.route('/gather/files', methods=['GET'])
@token_required
def get_apply_file_identifiers():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"state": 500, "message": "Failed to connect to the database"})

        cursor = conn.cursor()
        cursor.execute("SELECT file_id, file_name FROM tb_analysis_apply_file")
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

# 对应原来 @app.route('/api/analysis/apply/check/start', methods=['POST'])
@apply_bp.route('/check/start', methods=['POST'])
# @token_required
def start_analysis():
    """
    接收模型和样本数据，执行完整的分析和预测流程。
    （已优化逻辑，可容忍未找到的预处理和特征提取方法）
    """
    # --- 常量定义 ---
    METHOD_EMD = "经验模态分解"
    METHOD_WAVELET = "小波变换"
    METHOD_FFT = "傅里叶变换"
    METHOD_KURTOSIS = "峭度指标"
    METHOD_HISTOGRAM = "直方图特征"
    METHOD_WVD = "Wigner-Ville分布"

    conn = None
    cursor = None
    try:
        # --- 1. 获取并验证请求数据 ---
        data = request.get_json()
        if not data:
            return jsonify({"state": 400, "message": "错误：请求体中没有JSON数据"}), 400

        model_train_id = data.get('model_train_id')
        model_artifact_path_prefix = data.get('model_artifact_path')
        model_train_name = data.get('model_train_name')
        apply_sample_ids = data.get('apply_sample_ids')
        room_id = session.get('room_id')
        if not room_id:
            logger.warning("预处理操作未找到有效的room ID，将无法通过WebSocket发送进度")

        if not all([model_train_id, model_artifact_path_prefix, model_train_name, apply_sample_ids is not None]):
            return jsonify({"state": 400,
                            "message": "错误：请求体中缺少必需字段 (model_train_id, model_artifact_path, model_train_name, apply_sample_ids)"}), 400

        if not isinstance(apply_sample_ids, list) or not apply_sample_ids:
            return jsonify({"state": 400, "message": "错误：'apply_sample_ids'必须是一个非空列表"}), 400

        # --- 2. 建立数据库连接 ---
        conn = get_db_connection()
        if conn is None:
            # 模拟数据，当数据库连接失败时使用
            print("警告: 数据库连接失败，将使用模拟数据进行演示。")
            input_data_map = {sid: list(np.random.randn(2048)) for sid in apply_sample_ids}
            preprocess_methods = [{'method_name': '高斯滤波'}, {'method_name': '一个不存在的方法'}]
            feature_extract_method = "傅里叶变换"  # 或设置为 None 来测试跳过逻辑
            feature_extract_params = {}
        else:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            # --- 3. 从数据库获取输入数据 ---
            input_data_map = {}
            query = "SELECT apply_sample_id, sample_data FROM tb_analysis_apply_sample WHERE apply_sample_id = ANY(%s)"
            cursor.execute(query, (apply_sample_ids,))
            rows = cursor.fetchall()
            if not rows:
                return jsonify({"state": 404, "message": f"未找到ID为 {apply_sample_ids} 的样本数据"}), 404

            total_samples = len(rows)
            loaded_count = 0
            for row in rows:
                sample_id = row['apply_sample_id']
                raw_sample_data = row['sample_data']
                if raw_sample_data is None:
                    print(f"警告: 样本ID {sample_id} 的原始数据为 NULL，跳过。")
                    continue
                try:
                    parsed_list = json.loads(raw_sample_data) if isinstance(raw_sample_data, str) else raw_sample_data
                    if isinstance(parsed_list, list) and len(parsed_list) > 0:
                        input_data_map[sample_id] = parsed_list
                        loaded_count += 1
                        # 新增读取原始数据数据进度条
                        emit_process_progress(room_id, 'apply', {
                            'message': f'正在加载样本 {sample_id}',
                            'current': loaded_count,
                            'total': total_samples
                        })
                    else:
                        print(f"警告: 样本ID {sample_id} 的数据解析后不是非空列表。跳过。")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"错误: 样本ID {sample_id} 的数据解析失败: {e}. 跳过。")
                    continue

            if not input_data_map:
                return jsonify({"state": 404, "message": "所有样本数据缺失或解析失败，无数据可分析。"}), 404

            # --- 4. 从数据库获取分析流程配置 ---
            cursor.execute(
                "SELECT tats.from_sample_id FROM tb_analysis_model_train_sample AS tats WHERE tats.model_train_id = %s LIMIT 1",
                (model_train_id,))
            train_sample_row = cursor.fetchone()
            if not train_sample_row:
                return jsonify({"state": 404, "message": f"找不到模型训练ID {model_train_id} 对应的训练样本记录"}), 404
            from_sample_id = train_sample_row['from_sample_id']

            cursor.execute(
                "SELECT preprocess_method FROM tb_analysis_sample_preprocess WHERE preprocess_sample_id = %s LIMIT 1",
                (from_sample_id,))
            preprocess_row = cursor.fetchone()
            preprocess_methods = []
            if preprocess_row and preprocess_row['preprocess_method']:
                try:
                    methods_json = preprocess_row['preprocess_method']
                    preprocess_methods = json.loads(methods_json) if isinstance(methods_json, str) else methods_json
                except json.JSONDecodeError:
                    pass
                if not isinstance(preprocess_methods, list): preprocess_methods = []

            cursor.execute(
                "SELECT feature_extract, feature_extract_param FROM tb_analysis_sample_feature WHERE from_sample_id = %s LIMIT 1",
                (from_sample_id,))
            feature_row = cursor.fetchone()
            feature_extract_method, feature_extract_params = None, {}
            if feature_row:
                feature_extract_full = feature_row.get('feature_extract')
                if feature_extract_full:
                    feature_extract_method = feature_extract_full.rsplit('_', 1)[-1]
                if feature_row.get('feature_extract_param'):
                    try:
                        params_json = feature_row['feature_extract_param']
                        feature_extract_params = json.loads(params_json) if isinstance(params_json,
                                                                                       str) else params_json
                        if not isinstance(feature_extract_params, dict): feature_extract_params = {}
                    except json.JSONDecodeError:
                        feature_extract_params = {}

        # --- 硬编码 DNN 模型参数 (请根据您的训练代码调整) ---
        dnn_num_classes_hardcoded = len(apply_sample_ids)
        dnn_hidden_dims_hardcoded = [128, 64]
        dnn_dropout_rate_hardcoded = 0.1

        # --- 5. 定义方法到函数的映射 ---
        preprocess_map = {
            '异常缺失处理': Preprocessor.apply_method, '高斯滤波': Preprocessor.apply_method,
            '均值滤波': Preprocessor.apply_method, '最大最小规范化': Preprocessor.apply_method,
            'Z-score标准化': Preprocessor.apply_method, '主成分分析': Preprocessor.apply_method,
            '线性判别': Preprocessor.apply_method
        }
        feature_map = {
            METHOD_KURTOSIS: feature_extraction.kurtosis_index, METHOD_HISTOGRAM: feature_extraction.histogram_feature,
            METHOD_FFT: feature_extraction.fourier_transform, METHOD_WAVELET: feature_extraction.wavelet_transform,
            METHOD_EMD: feature_extraction.empirical_mode_decomposition,
            METHOD_WVD: feature_extraction.wigner_ville_distribution,
        }

        # --- 6. 确定窗口参数 (强制设定) ---
        window_size = 512
        step_size = 512
        is_overlapping = (step_size < window_size)

        # --- 7. 对每个输入样本执行处理流程 ---
        all_final_features_for_model = []
        window_to_sample_id_map = []
        total_samples = len(apply_sample_ids)
        processed_count = 0  # 记录实际处理的样本数

        for sample_id in apply_sample_ids:
            if sample_id not in input_data_map:
                print(f"警告: 样本ID {sample_id} 数据缺失或解析失败，跳过分析。")
                continue

            # 7.1. 预处理
            current_data_np = np.array(input_data_map[sample_id], dtype=np.float32).flatten()
            if current_data_np.size == 0: continue

            processed_data = current_data_np.copy()
            for method_spec in preprocess_methods:
                method_name = method_spec.get('method_name') if isinstance(method_spec, dict) else method_spec
                method_params = method_spec.get('params', {}) if isinstance(method_spec, dict) else {}

                # --- 优化点: 如果方法未在map中定义，则跳过而不是报错 ---
                if method_name in preprocess_map:
                    try:
                        processed_data = preprocess_map[method_name](processed_data, method_name=method_name,
                                                                     **method_params)
                        processed_data = np.array(processed_data, dtype=np.float32)
                    except Exception as e:
                        print(f"错误: 样本ID {sample_id} 在预处理 '{method_name}' 时失败: {e}")
                        processed_data = np.array([])
                        break  # 如果一个预处理步骤失败，则终止该样本的后续处理
                else:
                    print(f"信息: 样本ID {sample_id}，预处理方法 '{method_name}' 未在后端定义，已跳过。")

            if processed_data.size == 0: continue
            processed_count += 1
            # 新增预处理处理进度
            emit_process_progress(room_id, 'apply', {
                'message': f'样本 {sample_id} 预处理完成',
                'current': processed_count,
                'total': total_samples
            })

            # --- 7.2. 应用特征提取 (已优化) ---
            # --- 优化点: 如果方法未找到，则直接使用预处理后的数据作为特征 ---
            extracted_features = None
            total_feature_samples = len([sid for sid in apply_sample_ids if sid in input_data_map])
            feature_processed_count = 0
            if feature_extract_method and feature_extract_method in feature_map:
                print(f"信息: 样本ID {sample_id}，正在执行特征提取 '{feature_extract_method}'。")
                try:
                    extracted_features = feature_map[feature_extract_method](processed_data)
                    if not isinstance(extracted_features, np.ndarray):
                        extracted_features = np.array(extracted_features, dtype=np.float32)

                    # 新增特征处理进度条
                    feature_processed_count += 1
                    emit_process_progress(room_id, 'apply', {
                        'message': f'样本 {sample_id} 特征提取完成',
                        'current': feature_processed_count,
                        'total': total_feature_samples
                    })

                except Exception as e:
                    import traceback
                    print(
                        f"错误: 样本ID {sample_id} 在特征提取 '{feature_extract_method}' 时失败: {e}\n{traceback.format_exc()}")
                    continue  # 提取失败则跳过该样本
            else:
                print(
                    f"信息: 样本ID {sample_id}，未找到或未配置有效的特征提取方法 ('{feature_extract_method}')。将直接使用预处理后的数据进行分析。")
                extracted_features = processed_data  # 直接使用预处理数据

            if extracted_features is None or extracted_features.size == 0:
                print(f"警告: 样本ID {sample_id} 在特征提取阶段后结果为空，跳过。")
                continue

            print(f"DEBUG: 样本ID {sample_id} (特征提取/传递后: shape={extracted_features.shape})")

            # 7.3. 窗口化/形状调整
            final_features_for_sample = np.array([])
            if feature_extract_method == METHOD_EMD:
                if extracted_features.ndim == 2 and extracted_features.shape[1] > 0:
                    processed_1d_data = extracted_features[:, 0]
                    if processed_1d_data.size >= window_size:
                        windows, _ = create_windows_1d(processed_1d_data, 0, window_size, step_size, is_overlapping)
                        final_features_for_sample = windows
            elif feature_extract_method == METHOD_WAVELET:
                if extracted_features.ndim == 2:
                    data_to_window = extracted_features.T
                    if data_to_window.shape[1] >= window_size:
                        windows, _ = create_windows_wavelet(data_to_window, 0, window_size, step_size, is_overlapping)
                        final_features_for_sample = windows
            elif feature_extract_method == METHOD_FFT:
                processed_1d_data = extracted_features.flatten()
                if processed_1d_data.size >= window_size:
                    windows, _ = create_windows_1d(processed_1d_data, 0, window_size, step_size, is_overlapping)
                    final_features_for_sample = windows
            elif feature_extract_method in [METHOD_KURTOSIS, METHOD_HISTOGRAM, METHOD_WVD]:
                final_features_for_sample = extracted_features.reshape(1, -1)
            else:  # 其他未知方法
                final_features_for_sample = extracted_features.flatten().reshape(1, -1)

            if final_features_for_sample.size > 0:
                all_final_features_for_model.extend(final_features_for_sample)
                window_to_sample_id_map.extend([sample_id] * len(final_features_for_sample))

        if not all_final_features_for_model:
            return jsonify({"state": 400, "message": "所有样本都未能生成有效的特征向量用于预测。"}), 400

        # --- 8. 准备模型输入 ---
        X_all_samples = np.array(all_final_features_for_model, dtype=np.float32)
        if X_all_samples.ndim == 1:
            X_all_samples = X_all_samples.reshape(-1, 1)

        current_data_processed_dim = X_all_samples.shape[1]
        print(f"DEBUG: 模型输入的最终特征维度: {current_data_processed_dim}")

        # --- 9. 加载模型和Scaler ---
        model_name_base = os.path.basename(model_artifact_path_prefix).split('_')[0]
        is_dnn = (model_name_base == '深度神经网络')
        model_path = model_artifact_path_prefix + ('.pth' if is_dnn else '.joblib')
        scaler_path = model_artifact_path_prefix + '_scaler.joblib'
        # 新增：类别元数据路径
        metadata_path = model_artifact_path_prefix + '_metadata.joblib'

        if not os.path.exists(model_path):
            return jsonify({"state": 404, "message": f"在路径 {model_path} 未找到模型文件"}), 404

        # 新增：加载类别元数据
        try:
            if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    # 从元数据中获取训练时的类别数和类别列表
                    num_classes = metadata.get("num_classes", None)
                    model_classes = metadata.get("classes", None)
                    print(f"成功加载类别元数据: 类别数={num_classes}, 类别列表={model_classes}")
            else:
                    # 兼容旧模型：如果没有元数据，保留原有逻辑但警告
                    num_classes = len(apply_sample_ids)
                    model_classes = None
                    print(f"警告: 未找到类别元数据文件 {metadata_path}，将使用默认类别数")
        except Exception as e:
            return jsonify({"state": 500, "message": f"加载类别元数据失败: {e}"}), 500

        try:
            if is_dnn:
                # 根据训练逻辑设定DNN的input_dim
                if feature_extract_method in [METHOD_EMD, METHOD_FFT]:
                    dnn_model_input_dim = window_size
                else:
                    dnn_model_input_dim = current_data_processed_dim

                print(
                    f"DEBUG: Initializing GenericDNN with input_dim={dnn_model_input_dim}, num_classes={dnn_num_classes_hardcoded}")
                model = GenericDNN(
                    input_dim=int(dnn_model_input_dim),
                    num_classes=int(dnn_num_classes_hardcoded),
                    hidden_dims=dnn_hidden_dims_hardcoded,
                    dropout_rate=dnn_dropout_rate_hardcoded
                )
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
            else:
                model = joblib.load(model_path)

            scaler = None
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"成功加载Scaler: {scaler_path}")
        except Exception as e:
            return jsonify({"state": 500, "message": f"加载模型或Scaler失败: {e}"}), 500

        # --- 10. 批量预测 ---
        X_scaled = scaler.transform(X_all_samples) if scaler else X_all_samples

        if is_dnn:
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                y_pred_raw = model(X_tensor)
                if dnn_num_classes_hardcoded > 1:
                    y_pred_raw = torch.softmax(y_pred_raw, dim=1)
                raw_predictions = y_pred_raw.cpu().numpy()
        else:
            if hasattr(model, 'predict_proba'):
                raw_predictions = model.predict_proba(X_scaled)
            else:
                raw_predictions = model.predict(X_scaled)

        # --- 11. 聚合预测结果 ---
        final_predictions = []
        predictions_by_id = {}
        for i, sample_id in enumerate(window_to_sample_id_map):
            predictions_by_id.setdefault(sample_id, []).append(raw_predictions[i])

        is_regression = model_name_base in ['线性回归', '随机森林回归', '支持向量机回归'] or (
                    is_dnn and dnn_num_classes_hardcoded == 1)

        for sample_id, preds in predictions_by_id.items():
            preds_array = np.array(preds)
            result = {"sample_id": str(sample_id)}

            if model_name_base == 'K-均值聚类':
                # .item() to convert numpy int to native python int
                result["cluster_id"] = Counter(preds_array.flatten()).most_common(1)[0][0].item()
            elif is_regression:
                result["predicted_value"] = float(np.mean(preds_array))
            else:  # Classification
                avg_probs = np.mean(preds_array, axis=0)
                pred_label_idx = np.argmax(avg_probs)
                model_classes = getattr(model, 'classes_', [f"Class_{i}" for i in range(len(avg_probs))])
                result["predicted_label"] = str(model_classes[pred_label_idx])
                result["probabilities"] = {str(k): float(v) for k, v in zip(model_classes, avg_probs)}

            final_predictions.append(result)

        # --- 12. 构造成功响应 ---
        success_response = {
            "state": 200,
            "message": "分析成功",
            "data": {
                "check_result.json": {
                    "model_info": {
                        "train_id": model_train_id,
                        "train_name": model_train_name,
                        "artifact_path": model_artifact_path_prefix,
                        "performance_metrics": {}
                    },
                    "pipeline_executed": {
                        "preprocessing": [m['method_name'] if isinstance(m, dict) else m for m in preprocess_methods],
                        "feature_extraction": feature_extract_method,
                        "windowing_params": {
                            "window_size": window_size, "step_size": step_size, "is_overlapping": is_overlapping
                        }
                    },
                    "prediction": final_predictions
                }
            }
        }
        return jsonify(success_response), 200

    except psycopg2.Error as e:
        print(f"数据库错误: {e}")
        return jsonify({"state": 500, "message": f"数据库错误: {e}"}), 500
    except Exception as e:
        import traceback
        print(f"发生意外错误: {e}\n{traceback.format_exc()}")
        return jsonify({"state": 500, "message": f"发生内部服务器错误: {e}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
        print("DEBUG: Database connection closed.")

# 对应原来 @app.route('/api/analysis/apply/check/add/sample', methods=['POST'])
@apply_bp.route('/check/add/sample', methods=['POST'])
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

@apply_bp.route('/check/model', methods=['GET'])
@token_required
def get_apply_models():
    """
    获取模型训练数据列表 (用于分析应用模块的模型选择)
    - 参数：无
    - 响应：
      - 成功返回模型训练名称、model_train_id、模型构件路径
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

        # 直接查询 tb_analysis_model_train 表，获取 model_train_name, model_id, model_artifact_path
        sql_query = """
            SELECT
                model_train_name,
                model_train_id,
                model_artifact_path
            FROM
                tb_analysis_model_train
        """
        cursor.execute(sql_query)
        models = cursor.fetchall()  # 获取所有查询结果

        if not models:
            # 如果未找到任何模型训练数据，返回 state 404
            return jsonify({"state": 404, "data": [], "message": "未找到任何模型训练数据"}), 404


        # Flask 的 jsonify 通常可以直接处理 psycopg2.extras.DictRow 对象列表
        return jsonify({"state": 200, "data": models}), 200

    except Exception as e:
        # 捕获任何服务器内部错误
        print(f"获取模型训练数据失败: {str(e)}")  # 打印错误以便调试
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

