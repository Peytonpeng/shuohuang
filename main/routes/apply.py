import json
import datetime
import os
import uuid
from flask import request, jsonify, Blueprint
import main.Config.getConnection as get_db_connection
import psycopg2
import pandas as pd
from main.Config.SaveConfig import UPLOAD_FOLDER

apply = Blueprint('apply', __name__)


#5.12新增AI模型应用接口：
@apply.route('/api/analysis/apply/gather/import', methods=['POST'])
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
    cursor = None # 初始化 cursor 为 None
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
                conn.rollback() # 数据库操作失败时回滚
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
#5.12新增AI模型应用接口：
#这个接口不懂什么意思？
@apply.route('/api/analysis/apply/gather/check', methods=['GET'])
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
            }), 404 # 根据图片要求，返回 404 表示失败

        file_path = result['file_path']

        # 2. 检查文件是否存在于服务器文件系统中
        if not os.path.exists(file_path):
            return jsonify({
                "state": 404,
                "data": {
                    "success": "false",
                    "message": f"文件 '{file_path}' 在服务器上不存在或已被移动。"
                }
            }), 404 # 根据图片要求，返回 404 表示失败

        # 3. (可选) 检查文件是否为空或损坏，这里只进行简单的大小检查
        if os.path.getsize(file_path) == 0:
            return jsonify({
                "state": 400, # 文件内容问题，可以返回 400 Bad Request
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

#5.12新增AI模型应用接口：
@apply.route('/api/analysis/apply/gather/sample', methods=['GET'])
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
    create_user = request.args.get('create_user', 'admin') # 可以从请求头或session获取实际用户

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
            file_uuid_prefix = str(uuid.uuid4()) # 为当前文件解析批次生成一个 UUID 前缀

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
            conn.rollback() # 数据库操作失败时回滚
        return jsonify({"state": 500, "message": f"PostgreSQL 操作失败: {str(e)}"}), 500
    except Exception as e:
        if conn:
            conn.rollback() # 确保在其他异常时也回滚
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#5.12新增AI模型应用接口：
@apply.route('/api/analysis/apply/gather/wave', methods=['GET'])
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
                continue # 或者可以返回一个错误信息，取决于业务需求

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

#5.12新增AI模型应用接口：
@apply.route('/api/analysis/apply/check/model', methods=['GET'])
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
        models = cursor.fetchall() # 获取所有查询结果

        if not models:
            # 根据图片要求，如果未找到任何模型，返回 state 404
            return jsonify({"state": 404, "data": [], "message": "未找到任何模型数据"}), 404

        # Flask 的 jsonify 通常可以直接处理 psycopg2.extras.DictRow 对象列表
        # 如果遇到问题，可以手动转换为列表字典: [dict(model) for model in models]
        return jsonify({"state": 200, "data": models}), 200

    except Exception as e:
        # 捕获任何服务器内部错误
        print(f"获取模型数据失败: {str(e)}") # 打印错误以便调试
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


#/api/analysis/apply/check/model：这个逻辑据复杂，后面再说



#5.12新增AI模型应用接口：
@apply.route('/api/analysis/apply/check/add/sample', methods=['POST'])
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
        existing_samples = {row[0] for row in cursor.fetchall()} # 获取存在的 apply_sample_id

        if not existing_samples:
            return jsonify({
                "state": 404, # 根据图片要求，404表示失败
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
            http_status = 200 # 视为成功响应，但数据层面无更新

        return jsonify({
            "state": http_status,
            "data": {
                "success": success_status,
                "message": message
            }
        }), http_status

    except Exception as e:
        if conn:
            conn.rollback() # 发生异常时回滚数据库操作
        print(f"添加样本至样本库失败: {str(e)}") # 打印错误以便调试
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