import json

import pandas as pd
import os
import datetime
from flask import Flask, request, jsonify, Blueprint
import uuid
import main.Config.getConnection as get_db_connection
import psycopg2
select = Blueprint('select', __name__)


UPLOAD_FOLDER = "uploads"  # 服务器文件存储路径
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 创建文件夹（如果不存在）


@select.route('/api/analysis/train/select/import', methods=['POST'])
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


@select.route('/api/analysis/train/select/sample', methods=['GET'])
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
        cursor.execute("SELECT original_sample_id, sample_name FROM tb_analysis_sample_original WHERE file_id = %s", (file_id,))
        existing_samples = cursor.fetchall()

        if existing_samples:
            data_to_return = [{"sample_id": sample_id, "sample_name": sample_name} for sample_id, sample_name in existing_samples]
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


@select.route('/api/analysis/train/select/wave', methods=['GET'])
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
                print(f"JSON 解析错误的数据: {row['sample_data']}") # 打印解析失败的数据
                continue  # 跳过格式错误的数据

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        print(f"发生异常: {e}") # 打印异常信息
        return jsonify({"state": 500, "message": f"操作失败: {str(e)}"}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@select.route('/api/analysis/train/select/add/sample', methods=['POST'])
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
    cursor = None # 初始化cursor
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
        if cursor: #增加判断，避免cursor未被赋值，导致的报错
            cursor.close()
        if conn:
            conn.close()

# 5.15新增接口，保证后续所有的步骤仅仅使用某一个文件的样本
@select.route('/api/analysis/train/select/files', methods = ['GET'])
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