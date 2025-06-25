import json
import datetime
import uuid
import numpy as np
from flask import request, jsonify, Blueprint, logging
import main.Config.getConnection as get_db_connection
import psycopg2
from main.utils import feature_extraction
from main.utils.visualization import Visualizer
from main.utils.methods import Preprocessor
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
feature = Blueprint('feature', __name__)

@feature.route('/api/analysis/train/feature/fetch', methods=['GET'])
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

@feature.route('/api/analysis/train/feature/sample', methods=['GET'])
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
            return jsonify({"state": 400, "message": "缺少 file_id 参数"}), 400 # 400 Bad Request

        # 连接数据库
        conn = get_db_connection() # 假设你有一个获取数据库连接的函数
        if conn is None:
            return jsonify({"state": 500, "message": "数据库连接失败"}), 500

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 使用DictCursor

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
            sample_name = f"{sample['original_sample_name']} ({sample['preprocess_method']})" if sample.get('original_sample_name') else f"{sample['original_sample_id']} ({sample['preprocess_method']})"
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
        # 当前实现是按“原始样本出现的顺序”来决定组的顺序，组内再按各自时间排序。

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
            "state": 500, # 内部错误
            "message": f"获取样本列表失败: {str(e)}"
        }), 500
    finally:
        # 确保在任何情况下都关闭游标和连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@feature.route('/api/analysis/train/feature/confirm', methods=['POST'])
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

        if not sample_ids:
            return jsonify({"state": 400, "message": "样本ID列表为空或无效"}), 400

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

        for sample_id in sample_ids:
            if not sample_id:
                logging.warning("在样本ID列表中发现一个空ID，已跳过。")
                continue
            logging.debug(f"开始处理样本ID: {sample_id}")

            original_sample_data_str = None
            sample_type = None
            determined_legend_text = str(sample_id)

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

                cursor.execute(insert_sql, (
                    feature_sample_id, sample_id, sample_type,
                    feature_method or '无', params_json_for_db, '无',
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

        return jsonify({
            "state": 200, "message": msg_out,
            "data": {"beforeprocess": before_process_response_list, "afterprocess": after_process_response_list}
        }), 200
    except psycopg2.Error as db_err:
        if conn: conn.rollback()
        logging.error(f"数据库操作失败: {db_err}", exc_info=True)
        return jsonify({"state": 500, "message": f"数据库错误: {str(db_err)}"}), 500
    except json.JSONDecodeError as json_err:
        logging.error(f"请求体JSON解析失败: {json_err}", exc_info=True)
        return jsonify({"state": 400, "message": f"请求数据JSON格式错误: {str(json_err)}"}), 400
    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"特征确认过程发生未知错误: {e}", exc_info=True)
        return jsonify({"state": 500, "message": f"处理失败: {str(e)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@feature.route('/api/analysis/train/feature/param/get', methods=['GET'])
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

@feature.route('/api/analysis/train/feature/param/save', methods=['POST'])
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


