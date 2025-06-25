import json
import datetime
from flask import request, jsonify, Blueprint
import main.Config.getConnection as get_db_connection
import psycopg2
from main.utils.visualization import Visualizer
from main.utils.methods import Preprocessor
preProcess = Blueprint('preprocess', __name__)


@preProcess.route('/api/analysis/train/pre/sample', methods=['GET'])
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
            return jsonify({"state": 400, "message": "缺少 file_id 参数"}), 400 # 400 Bad Request

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
        cursor.execute(sql_query, (file_id,)) # 注意这里 file_id 是一个单元素的元组

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


@preProcess.route('/api/analysis/train/pre/method', methods=['GET'])
def get_methods():
    """获取所有可用的方法"""
    methods_data = [
        {"stage": "1", "stage_name": "异常缺失处理", "data": [{"method_name": "异常缺失处理"}]},
        {"stage": "2", "stage_name": "数据降噪", "data": [{"method_name": "高斯滤波"}, {"method_name": "均值滤波"}]},
        {"stage": "3", "stage_name": "数据规范化", "data": [{"method_name": "最大最小规范化"}, {"method_name": "Z-score标准化"}]},
        {"stage": "4", "stage_name": "数据降维", "data": [{"method_name": "主成分分析"}, {"method_name": "线性判别"}]}
    ]
    return jsonify({"state": 200, "data": methods_data})


@preProcess.route('/api/analysis/train/pre/confirm', methods=['POST'])
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

        if not sample_ids_str:
            return jsonify({"state": 400, "message": "缺少必要参数 sample_ids"}), 400

        # 将样本ID字符串分割成列表
        sample_ids_list = [sid.strip() for sid in sample_ids_str.split(',') if sid.strip()]

        if not sample_ids_list:
            return jsonify({"state": 400, "message": "无效的样本ID"}), 400

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
        method_parts = pre_process_methods_str.split(',')[:4] # 只取前4个
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
            return jsonify({"state": 404, "message": "未找到样本数据"}), 404

        response_data = []

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

        for sample in original_samples:
            try:
                original_id = sample["original_sample_id"]
                original_data_str = sample["sample_data"].strip()

                # 确保数据是有效的 JSON 列表
                if original_data_str.startswith('[') and original_data_str.endswith(']'):
                    current_data = json.loads(original_data_str)
                else:
                    raise json.JSONDecodeError("不是有效的 JSON 列表", original_data_str, 0)

                # 提取原始列名，用于生成 preprocess_sample_id
                original_column_name = sample["sample_name"].split('_')[-1] if '_' in sample["sample_name"] else sample["sample_name"]

                # 保留 before_process_data 结构
                before_process_data = {
                    "data": list(current_data),
                    # 假设 Visualizer 类和方法可用
                    "waveform": Visualizer.generate_waveform(current_data),
                    "spectrum": Visualizer.generate_spectrum(current_data)
                }

                processed_value = list(current_data)
                applied_methods_list_for_log = [] # 用于记录实际应用的步骤日志

                # 标记每个步骤是否已应用，确保每个步骤只应用一次
                applied_steps = {step: False for step in pipeline_steps}

                # 按照 pipeline_steps 的顺序应用方法
                for step in pipeline_steps:
                    method_for_step = None
                    # 遍历用户提交的方法列表，找到属于当前步骤的第一个方法
                    for method in pre_process_methods_from_input:
                         if method in pipeline_methods.get(step, []):
                             method_for_step = method
                             break # 找到第一个就用它

                    if method_for_step and not applied_steps[step]:
                        try:
                            processed_value = Preprocessor.apply_method(processed_value, method_for_step)
                            applied_methods_list_for_log.append(method_for_step) # 记录应用的具体方法
                            applied_steps[step] = True # 标记该步骤已应用
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
                new_preprocess_sample_id = f"{preprocess_sample_id_base}_{original_column_name}"

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
                    original_id, # 直接使用原始样本的 ID
                    applied_methods_final_json, # 存储实际应用的 methods JSON 字符串
                    json.dumps(processed_value),
                    "system", # 假设创建用户是 system
                    datetime.datetime.now()
                ))

                # 为响应数据添加条目
                response_data.append({
                    "sample_id": original_id, # 响应中仍然使用原始样本ID作为标识
                    "sample_name": sample["sample_name"],
                    "beforeprocess": before_process_data,
                    "afterprocess": after_process_data,
                    "process_log": applied_methods_list_for_log, # 返回实际应用的步骤日志（列表形式）
                    "new_preprocess_sample_id": new_preprocess_sample_id # 将新生成的预处理样本ID也返回
                })

            except json.JSONDecodeError as e:
                print(f"样本 {original_id} 的数据解析失败: {e}")
                continue # 对于单个样本的处理失败，记录错误并跳过该样本
            except ValueError as e:
                print(f"样本 {original_id} 预处理方法错误: {str(e)}")
                continue # 对于单个样本的处理失败，记录错误并跳过该样本
            except Exception as e:
                print(f"处理样本 {original_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue # 对于单个样本的处理失败，记录错误并跳过该样本

        # 提交事务：确保所有的删除和插入操作是原子性的对于整个请求而言
        # 如果中间有样本处理失败（跳过），不会影响已成功处理并准备插入/删除的数据的提交
        conn.commit()

        return jsonify({
            "state": 200,
            "data": response_data
        }), 200

    except Exception as e:
        # 捕获整个请求处理过程中的异常（如数据库连接失败，JSON解析错误等）
        if conn:
            conn.rollback() # 如果发生任何未捕获的异常，回滚整个事务
        print(f"服务器内部错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"state": 500, "message": f"服务器内部错误: {str(e)}"}), 500
    finally:
        # 确保在任何情况下都关闭游标和连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()