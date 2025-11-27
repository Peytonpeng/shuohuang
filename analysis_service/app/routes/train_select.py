# app/routes/train_select.py
from flask import Blueprint, request, jsonify
import uuid
import os
import datetime
import json
import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import Error

from analysis_service.app.routes.auth import token_required
from analysis_service.app.utils.common import safe_remove, api_error
from analysis_service.config import UPLOAD_FOLDER
from analysis_service.app.db import get_db_connection
from analysis_service.app.extensions import logger

train_select_bp = Blueprint("train_select", __name__)

# 原 @app.route('/api/analysis/train/select/import', methods=['POST'])
@train_select_bp.route('/select/import', methods=['POST'])
@token_required
def import_selection_file():
    conn = None
    file_path = None  # 记录真实保存路径，方便出错时删除

    # 内部小工具：尝试多种编码校验 CSV
    def try_read_csv(path: str):
        """
        读取 CSV，仅用于校验，不返回 df。
        按多种常见编码依次尝试，全部失败则抛 UnicodeDecodeError。
        """
        encodings = ["utf-8", "utf-8-sig", "gbk"]
        last_err = None
        for enc in encodings:
            try:
                pd.read_csv(path, encoding=enc)
                return  # 成功就返回，不用真的保存 df
            except UnicodeDecodeError as e:
                last_err = e
                continue
        # 所有编码都失败，抛最后一次错误
        raise last_err or UnicodeDecodeError("unknown", b"", 0, 1, "encoding failed")

    try:
        # 1. 校验上传文件
        file = request.files.get("file")
        if file is None:
            return api_error(400, "未提供上传文件")
        if not file.filename:
            return api_error(400, "未选择文件")

        original_filename = file.filename
        file_id = str(uuid.uuid4())
        server_filename = file_id  # 服务器上用 file_id 作为文件名

        # 确保上传目录存在
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, server_filename)

        # 2. 获取数据库连接
        conn = get_db_connection()
        if conn is None:
            return api_error(500, "数据库连接失败")

        # 3. 检查文件名是否重复
        with conn.cursor() as cursor:
            check_sql = "SELECT 1 FROM tb_analysis_sample_file WHERE file_name = %s"
            cursor.execute(check_sql, (original_filename,))
            if cursor.fetchone():
                return api_error(409, f"文件名 '{original_filename}' 已存在")

        # 4. 保存文件
        file.save(file_path)

        # 5. 校验 CSV 内容/编码
        try:
            try_read_csv(file_path)
        except FileNotFoundError:
            return api_error(404, "文件未找到")
        except pd.errors.ParserError:
            safe_remove(file_path)
            return api_error(400, "CSV 解析失败，请检查文件内容与分隔符")
        except UnicodeDecodeError:
            safe_remove(file_path)
            return api_error(400, "文件编码错误，请保存为 UTF-8 编码的 CSV 后重新上传")

        # 6. 插入数据库记录
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO tb_analysis_sample_file 
                (file_id, file_name, file_path, demo, create_user, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                file_id,
                original_filename,
                file_path,
                "自动上传",
                "admin",
                datetime.datetime.now()
            ))

        conn.commit()
        return jsonify({"state": 200, "data": {"file_id": file_id}}), 200

    except UnicodeDecodeError:
        # 万一还有漏网的编码问题从内部冒出来，这里兜底一下
        safe_remove(file_path)
        return api_error(400, "文件编码错误，请保存为 UTF-8 编码的 CSV 后重新上传")
    except Exception as e:
        # 所有没单独处理的异常都归这里
        if conn:
            conn.rollback()
        safe_remove(file_path)
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500

    finally:
        if conn:
            conn.close()


# 原@app.route('/api/analysis/train/select/sample', methods=['GET'])
@train_select_bp.route('/select/sample', methods=['GET'])
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

            df = None
            # 读取 CSV 数据
            try:
                df = pd.read_csv(file_path)
                # 删除所有列都是NaN的行
                df.dropna(how='all', inplace=True)
                # 接下来再将剩余的 NaN 填充为空字符串（针对有效数据中的个别NaN）
                df.fillna('', inplace=True)

            except Exception as e:
                return jsonify({"state": 500, "message": f"读取或处理CSV文件失败: {str(e)}"}), 500

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

# 原@app.route('/api/analysis/train/select/wave', methods=['GET'])
@train_select_bp.route('/select/wave', methods=['GET'])
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

        logger.debug(f"执行的 SQL 查询: {query}")  # 打印执行的 SQL 语句
        logger.debug(f"传递的参数: {tuple(sample_ids_list)}")  # 打印传递的参数

        cursor.execute(query, tuple(sample_ids_list))
        results = cursor.fetchall()

        logger.debug(f"查询结果数量: {len(results)}")  # 打印查询结果的数量

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
                logger.error(f"JSON 解析错误的数据: {row['sample_data']}")  # 打印解析失败的数据
                continue  # 跳过格式错误的数据

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        logger.error(f"发生异常: {e}")  # 打印异常信息
        return jsonify({"state": 500, "message": f"操作失败: {str(e)}"}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# 原 @app.route('/api/analysis/train/select/add/sample', methods=['POST'])
@train_select_bp.route('/select/add/sample', methods=['POST'])
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

# 原 @app.route('/api/analysis/train/select/files', methods=['GET'])
@train_select_bp.route('/select/files', methods=['GET'])
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

# 原 @app.route('/api/analysis/train/pre/sample', methods=['GET'])
@train_select_bp.route('/pre/sample', methods=['GET'])
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
