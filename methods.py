import numpy as np
import scipy.ndimage as ndimage
import traceback
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings


class Preprocessor:
    @staticmethod
    def handle_missing_values(data):
        """异常缺失处理"""
        try:
            # 将数据转换为numpy数组以便处理
            data_array = np.array([float(x) if x is not None and x != '' else np.nan for x in data], dtype=float)

            # 检测缺失值(NaN)和异常值(如inf)
            mask_nan = np.isnan(data_array)
            mask_inf = np.isinf(data_array)

            if np.any(mask_nan) or np.any(mask_inf):
                print(f"发现缺失值或异常值: NaN数量={np.sum(mask_nan)}, Inf数量={np.sum(mask_inf)}")

                # 处理方法1: 使用前向填充
                # 首先复制一份数组
                filled_data = data_array.copy()

                # 标记所有需要处理的索引
                mask_all = mask_nan | mask_inf

                if np.all(mask_all):
                    # 如果全是缺失值，则用0替代
                    filled_data = np.zeros_like(data_array)
                else:
                    # 计算数组的均值(忽略NaN和Inf)
                    valid_data = data_array[~mask_all]
                    mean_value = np.mean(valid_data)

                    # 前向填充
                    last_valid = mean_value  # 初始值为均值
                    for i in range(len(filled_data)):
                        if mask_all[i]:
                            filled_data[i] = last_valid
                        else:
                            last_valid = filled_data[i]

                return filled_data.tolist()

            # 如果没有缺失值或异常值，直接返回原数据
            return data_array.tolist()

        except Exception as e:
            print(f"异常缺失处理错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10] if len(data) > 10 else data}...")
            traceback.print_exc()
            raise

    @staticmethod
    def pca_reduction(data, n_components=0.95):
        """主成分分析降维

        参数:
            data: 输入数据
            n_components: 可以是组件数量或者解释方差比例(0-1之间)
        """
        try:
            # 确保输入为数值列表并转换为2D数组(样本 × 特征)
            data_array = np.array([float(x) for x in data], dtype=float)

            # 对于一维数据，我们需要将其重塑为二维
            # 由于PCA需要样本数大于特征数，我们可以用不同窗口大小创建多个样本
            window_size = max(5, int(len(data_array) * 0.1))  # 窗口大小为数据长度的10%，最小为5

            # 创建滑动窗口数据
            windowed_data = []
            for i in range(0, len(data_array) - window_size + 1, max(1, window_size // 2)):  # 50%重叠
                windowed_data.append(data_array[i:i + window_size])

            # 转换为数组
            X = np.array(windowed_data)

            if len(X) < 2:
                print("警告: 数据太少，无法执行PCA。返回原始数据。")
                return data_array.tolist()

            # 执行PCA
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)

            # 解释方差比例
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA解释方差比例: {explained_variance}")
            print(f"使用的主成分数量: {pca.n_components_}")

            # 重构数据
            X_reconstructed = pca.inverse_transform(X_reduced)

            # 对窗口化数据进行平均，得到最终的一维数据
            # 创建计数数组和累加数组
            result = np.zeros(len(data_array))
            counts = np.zeros(len(data_array))

            for i, window in enumerate(X_reconstructed):
                start_idx = i * max(1, window_size // 2)
                end_idx = start_idx + window_size
                if end_idx > len(data_array):
                    end_idx = len(data_array)

                result[start_idx:end_idx] += window[:end_idx - start_idx]
                counts[start_idx:end_idx] += 1

            # 避免除零错误
            counts[counts == 0] = 1
            result = result / counts

            return result.tolist()

        except Exception as e:
            print(f"主成分分析错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10] if len(data) > 10 else data}...")
            traceback.print_exc()
            raise

    @staticmethod
    def lda_reduction(data, n_components=1):
        """线性判别分析降维

        参数:
            data: 输入数据
            n_components: 降维后的维度
        """
        try:
            # 确保输入为数值列表
            data_array = np.array([float(x) for x in data], dtype=float)

            # 对于无监督场景下的线性判别，我们需要创建人工标签
            # 这里我们简单地将数据分成几个段落，每个段落作为一个类别
            n_segments = min(5, len(data_array) // 10)  # 最多5个段落，每段至少10个点
            n_segments = max(2, n_segments)  # 至少2个段落

            # 创建标签
            labels = np.zeros(len(data_array), dtype=int)
            segment_size = len(data_array) // n_segments
            for i in range(n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < n_segments - 1 else len(data_array)
                labels[start:end] = i

            # 为每个点创建特征：使用窗口化数据
            window_size = 5  # 使用前后共5个点作为特征
            X = []
            valid_indices = []

            for i in range(len(data_array)):
                # 如果窗口越界则跳过
                if i - window_size // 2 < 0 or i + window_size // 2 >= len(data_array):
                    continue

                # 取窗口特征
                window = data_array[i - window_size // 2:i + window_size // 2 + 1]
                X.append(window)
                valid_indices.append(i)

            X = np.array(X)
            y = labels[valid_indices]

            # 检查数据有效性
            if len(np.unique(y)) < 2:
                print("警告: 标签不足两类，无法执行LDA。返回原始数据。")
                return data_array.tolist()

            # 执行LDA
            n_components = min(n_components, len(np.unique(y)) - 1)  # LDA组件数量上限是类别数-1
            lda = LinearDiscriminantAnalysis(n_components=n_components)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_reduced = lda.fit_transform(X, y)

            # 创建映射回原始维度的函数
            # 简单方法：对每个类别取平均值
            class_means = {}
            for cls in np.unique(y):
                mask = (y == cls)
                if np.any(mask):
                    class_means[cls] = np.mean(X_reduced[mask])

            # 将LDA结果映射回原始数据长度
            result = np.zeros(len(data_array))
            for i, idx in enumerate(valid_indices):
                result[idx] = X_reduced[i][0]  # 使用第一个LDA组件

            # 处理边界点(未包含在valid_indices中的点)
            for i in range(len(data_array)):
                if i not in valid_indices:
                    # 找到最近的有效点
                    distances = np.abs(np.array(valid_indices) - i)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    result[i] = result[nearest_idx]

            # 归一化结果以匹配原始数据的大致范围
            min_original = np.min(data_array)
            max_original = np.max(data_array)
            min_result = np.min(result)
            max_result = np.max(result)

            # 避免除零错误
            if max_result != min_result:
                result = (result - min_result) / (max_result - min_result) * (
                            max_original - min_original) + min_original

            return result.tolist()

        except Exception as e:
            print(f"线性判别分析错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10] if len(data) > 10 else data}...")
            traceback.print_exc()
            raise

    @staticmethod
    def gaussian_filter(data, sigma=1):
        """高斯滤波"""
        try:
            # 确保输入为数值列表
            data_array = np.array([float(x) for x in data], dtype=float)
            result = ndimage.gaussian_filter1d(data_array, sigma=sigma)
            return result.tolist()
        except Exception as e:
            print(f"高斯滤波错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10]}...")
            traceback.print_exc()
            raise

    @staticmethod
    def moving_average(data, window_size=3):
        """均值滤波"""
        try:
            # 确保输入为数值列表
            data_array = np.array([float(x) for x in data], dtype=float)
            window = np.ones(int(window_size)) / float(window_size)
            result = np.convolve(data_array, window, mode='same')
            return result.tolist()
        except Exception as e:
            print(f"均值滤波错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10]}...")
            traceback.print_exc()
            raise

    @staticmethod
    def min_max_normalization(data):
        """最大最小规范化"""
        try:
            # 确保输入为数值列表
            data_list = [float(x) for x in data]
            min_val = min(data_list)
            max_val = max(data_list)
            if max_val == min_val:
                return [0.5] * len(data_list)
            return [(x - min_val) / (max_val - min_val) for x in data_list]
        except Exception as e:
            print(f"最大最小规范化错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10]}...")
            traceback.print_exc()
            raise

    @staticmethod
    def z_score_normalization(data):
        """Z-score标准化"""
        try:
            # 确保输入为数值列表
            data_array = np.array([float(x) for x in data], dtype=float)
            mean = np.mean(data_array)
            std = np.std(data_array)
            if std == 0:
                return [0.0] * len(data)
            return ((data_array - mean) / std).tolist()
        except Exception as e:
            print(f"Z-score标准化错误: {str(e)}")
            print(f"输入数据类型: {type(data)}, 内容: {data[:10]}...")
            traceback.print_exc()
            raise

    @staticmethod
    def apply_method(data, method_name, **kwargs):
        """应用指定的预处理方法"""
        print(f"apply_method接收到数据类型: {type(data)}")
        print(f"数据内容(前10个): {data[:10] if len(data) > 10 else data}")
        print(f"方法名称: {method_name}")
        print(f"kwargs: {kwargs}")

        # 确保data是一个列表
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise ValueError(f"输入数据必须是列表、元组或NumPy数组，而不是 {type(data)}")

        # 确保列表中每个元素都是数值
        cleaned_data = []
        for i, item in enumerate(data):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                cleaned_data.append(item)
            else:
                print(f"警告: 索引 {i} 处的项目不是数值: {item}, 类型: {type(item)}")
                # 尝试转换为浮点数
                try:
                    cleaned_data.append(float(item))
                except:
                    raise ValueError(f"无法将项目转换为数值: {item}, 类型: {type(item)}")

        method_map = {
            "异常缺失处理": Preprocessor.handle_missing_values,
            "高斯滤波": Preprocessor.gaussian_filter,
            "均值滤波": Preprocessor.moving_average,
            "最大最小规范化": Preprocessor.min_max_normalization,
            "Z-score标准化": Preprocessor.z_score_normalization,
            "主成分分析": Preprocessor.pca_reduction,
            "线性判别": Preprocessor.lda_reduction
        }

        if method_name not in method_map:
            raise ValueError(f"未知的预处理方法: {method_name}")

        # 调用相应的方法，但不传递任何参数给不需要参数的方法
        method = method_map[method_name]

        if method_name in ["最大最小规范化", "Z-score标准化", "异常缺失处理"]:
            # 这些方法不接受额外参数
            return method(cleaned_data)
        else:
            # 过滤kwargs
            import inspect
            sig = inspect.signature(method)
            valid_params = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return method(cleaned_data, **valid_params)