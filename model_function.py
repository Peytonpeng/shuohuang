import json
import time
import os
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, f1_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import TensorDataset, DataLoader
import traceback
import datetime

# --- 特征提取方法常量 ---
METHOD_EMD = "经验模态分解"
METHOD_WAVELET = "小波变换"
METHOD_FFT = "傅里叶变换"
METHOD_KURTOSIS = "峭度指标"
METHOD_HISTOGRAM = "直方图特征"

# 全局 SocketIO 实例和 emit 函数引用
socketio_instance = None
external_emit_functions = {}


def set_socketio_instance(socketio_obj):
    """设置全局 SocketIO 实例"""
    global socketio_instance
    socketio_instance = socketio_obj


def set_emit_functions(funcs):
    """设置外部 emit 函数引用"""
    global external_emit_functions
    external_emit_functions = funcs


# === emit转发函数 ===
def emit_process_progress(*args, **kwargs):
    """通用进度事件"""
    if external_emit_functions and 'training_progress' in external_emit_functions:
        return external_emit_functions['training_progress'](*args, **kwargs)


def emit_process_error(*args, **kwargs):
    """通用错误事件"""
    if external_emit_functions and 'training_error' in external_emit_functions:
        return external_emit_functions['training_error'](*args, **kwargs)


def emit_epoch_result(*args, **kwargs):
    """训练轮次结果事件"""
    if external_emit_functions and 'epoch_result' in external_emit_functions:
        return external_emit_functions['epoch_result'](*args, **kwargs)


def emit_round_result(*args, **kwargs):
    """训练回合结果事件"""
    if external_emit_functions and 'round_result' in external_emit_functions:
        return external_emit_functions['round_result'](*args, **kwargs)


def emit_process_completed(*args, **kwargs):
    """通用完成事件"""
    if external_emit_functions and 'training_completed' in external_emit_functions:
        return external_emit_functions['training_completed'](*args, **kwargs)


# --- 辅助函数：一维数据窗口化 (EMD, FFT) ---
def create_windows_1d(sequence_1d, label, window_size, step, is_overlapping=False):
    """为一维序列创建滑动窗口"""
    windowed_data = []
    windowed_labels = []
    actual_step = step if is_overlapping else window_size
    if actual_step <= 0 or window_size <= 0 or len(sequence_1d) < window_size:
        return np.array([]), np.array([])

    for j in range(0, len(sequence_1d) - window_size + 1, actual_step):
        window = sequence_1d[j:j + window_size]
        windowed_data.append(window)
        windowed_labels.append(label)

    return np.array(windowed_data, dtype=np.float32), np.array(windowed_labels, dtype=int)


# --- 辅助函数：小波数据窗口化 (2D -> N 个展平的 1D 窗口) ---
def create_windows_wavelet(sequence_2d, label, window_size, step, is_overlapping=False):
    """为二维小波数据创建滑动窗口（展平）"""
    windowed_data_flat = []
    windowed_labels = []

    if not isinstance(sequence_2d, np.ndarray):
        sequence_2d = np.array(sequence_2d)

    if sequence_2d.ndim != 2:
        print(f"小波数据不是二维 (形状: {sequence_2d.shape})。跳过。")
        return np.array([]), np.array([])

    num_channels, seq_len = sequence_2d.shape
    actual_step = step if is_overlapping else window_size

    if actual_step <= 0 or window_size <= 0 or seq_len < window_size:
        return np.array([]), np.array([])

    for j in range(0, seq_len - window_size + 1, actual_step):
        window = sequence_2d[:, j:j + window_size]  # 形状 (num_channels, window_size)
        windowed_data_flat.append(window.flatten())  # 展平为 (num_channels * window_size,)
        windowed_labels.append(label)

    return np.array(windowed_data_flat, dtype=np.float32), np.array(windowed_labels, dtype=int)


# --- 通用 DNN 模型 (包含 CNN 层) ---
class GenericDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout_rate=0.1):
        super(GenericDNN, self).__init__()

        # --- CNN 特征提取部分 ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.silu1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.silu2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- 计算卷积层输出的展平维度 ---
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            dummy_output_conv1 = self.pool1(self.silu1(self.bn1(self.conv1(dummy_input))))
            dummy_output_conv2 = self.pool2(self.silu2(self.bn2(self.conv2(dummy_output_conv1))))
            self.flattened_dim = dummy_output_conv2.view(1, -1).size(1)

        # --- 全连接分类部分 ---
        layers = []
        current_dim = self.flattened_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.pool1(self.silu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.silu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # 展平
        return self.classifier_layers(x)


# --- 通用 DNN 模型的训练函数 (集成 SocketIO) ---
def train_generic_dnn_model(train_dataloader, test_dataloader, input_dim, num_classes,
                            num_rounds=1, epochs_per_round=20, lr=0.001,
                            hidden_dims=[128, 64], dropout_rate=0.5, training_id=None,stop_event=None):
    """训练通用DNN模型"""
    start_time = time.time()  # 添加训练开始时间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始DNN训练，共 {num_rounds} 轮, 每轮 {epochs_per_round} epochs。设备: {device}")

    model = GenericDNN(input_dim, num_classes, hidden_dims, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_state = None
    best_accuracy = 0.0
    all_losses = []
    all_accuracies = []
    round_results = []

    for round_idx in range(num_rounds):
        # 检查是否收到停止信号 (在每轮开始时)
        if stop_event and stop_event.is_set():
            print(f"DNN 训练在第 {round_idx + 1} 轮时被用户中止。")
            return model, {"error": "Training stopped by user", "message": "训练已中止"}

        print(f"------ 开始训练轮次 {round_idx + 1}/{num_rounds} ------")
        round_losses = []
        round_accuracies = []

        for epoch in range(epochs_per_round):
            # 检查是否收到停止信号 (在每个epoch开始时)
            if stop_event and stop_event.is_set():
                print(f"DNN 训练在第 {round_idx + 1} 轮的第 {epoch + 1} epoch 时被用户中止。")
                return model, {"error": "Training stopped by user", "message": "训练已中止"}

            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            avg_loss = total_loss / len(train_dataloader)
            accuracy = correct / total
            round_losses.append(avg_loss)
            round_accuracies.append(accuracy)
            all_losses.append(avg_loss)
            all_accuracies.append(accuracy)

            # 每20个epoch打印一次进度
            if (epoch + 1) % 20 == 0 or epoch == epochs_per_round - 1:
                print(f"  轮次 [{round_idx + 1}/{num_rounds}], Epoch [{epoch + 1}/{epochs_per_round}] "
                      f"(总计 [{epoch + 1 + round_idx * epochs_per_round}/{num_rounds * epochs_per_round}]), "
                      f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Best Acc: {best_accuracy:.4f}")

                # 发送epoch结果
                if training_id:
                    progress_data = {
                        'round': round_idx + 1,
                        'epoch': epoch + 1,
                        'total_epochs': num_rounds * epochs_per_round,
                        'current_epoch': epoch + 1 + round_idx * epochs_per_round,
                        'loss': float(avg_loss),
                        'accuracy': float(accuracy),
                        'best_accuracy': float(best_accuracy)
                    }
                    emit_epoch_result(training_id, progress_data)

        # 计算轮次平均结果
        avg_round_loss = sum(round_losses) / len(round_losses)
        avg_round_accuracy = sum(round_accuracies) / len(round_accuracies)
        round_results.append({
            'round': round_idx + 1,
            'avg_loss': avg_round_loss,
            'avg_accuracy': avg_round_accuracy
        })

        # 发送轮次结果
        if training_id:
            round_data = {
                'round': round_idx + 1,
                'total_rounds': num_rounds,
                'avg_loss': float(avg_round_loss),
                'avg_accuracy': float(avg_round_accuracy),
                'best_accuracy': float(best_accuracy)
            }
            emit_round_result(training_id, round_data)

        # 评估当前轮次
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()

        print(f"------ 完成训练轮次 {round_idx + 1}/{num_rounds} (最佳准确率: {best_accuracy:.4f}) ------")

    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型状态，准确率: {best_accuracy:.4f}")

    return model, {
        '训练时间': time.time() - start_time,
        '准确率': best_accuracy,
        '最终Loss': all_losses[-1],
        '总轮次': num_rounds,
        '每轮Epochs': epochs_per_round,
        '总Epochs': num_rounds * epochs_per_round,
        '训练历史': {
            'rounds': round_results,
            'epochs': list(range(1, num_rounds * epochs_per_round + 1)),
            'losses': all_losses,
            'accuracies': all_accuracies,
            'best_accuracy': best_accuracy
        },
        'message': 'DNN训练成功完成'
    }


# --- 模型评估函数 (用于非 DNN 模型，添加 SocketIO 支持) ---
def evaluate_sklearn_model(model, X_train_df, X_test_df, y_train_series, y_test_series,
                           model_type, num_classes=None, training_id=None):
    """评估 Scikit-learn 模型，并通过 WebSocket 发送更新"""
    start_time = time.time()
    mse, recall, f1, accuracy = None, None, None, None
    cluster_centers_list = None
    current_message = None

    X_train, X_test = X_train_df, X_test_df
    y_train, y_test = y_train_series, y_test_series

    emit_process_progress(training_id, 'training', {
        'status': f'started_{model_type.lower()}', 'message': f'正在开始 {model_type} 训练...',
        'model_name': model_type
    })

    # --- 数据验证 ---
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)) or \
            (y_train is not None and not isinstance(y_train, (pd.Series, np.ndarray))):
        current_message = "接收到无效的训练数据或标签格式。"
    elif (isinstance(X_train, pd.DataFrame) and X_train.empty) or \
            (isinstance(X_train, np.ndarray) and X_train.size == 0):
        current_message = f"模型 {model_type} 的训练数据为空。"

    if current_message:
        print(f"错误: {current_message}")
        emit_process_error(training_id,'training', current_message)
        return {'训练时间': 0, 'message': current_message}

    # --- 训练和评估 ---
    try:
        if model_type in ['线性回归', '逻辑回归', '随机森林', '支持向量机']:
            if y_train is None or y_train.size == 0:
                current_message = f"模型 {model_type} 的训练标签为空。"
            elif len(X_train) != len(y_train):
                current_message = f"模型 {model_type} 训练数据和标签数量不匹配。"

            if current_message:
                print(f"错误: {current_message}")
                emit_process_error(training_id,'training', current_message)
                return {'训练时间': 0, 'message': current_message}

            emit_process_progress(training_id, 'training', {'status': 'fitting', 'message': f'正在拟合 {model_type}...',
                                                            'progress_percent': 30})
            model.fit(X_train, y_train)
            emit_process_progress(training_id, 'training',
                                  {'status': 'evaluating', 'message': f'正在评估 {model_type}...',
                                   'progress_percent': 70})

            if X_test.size > 0 and y_test.size > 0:
                y_pred = model.predict(X_test)
                if model_type == '线性回归':
                    mse = mean_squared_error(y_test, y_pred)
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                print(f"警告: 模型 {model_type} 的测试数据或标签为空，无法评估。")

        elif model_type == 'K-均值聚类':
            emit_process_progress(training_id, 'training', {'status': 'fitting', 'message': f'正在拟合 {model_type}...',
                                                            'progress_percent': 50})
            model.fit(X_train)
            if hasattr(model, 'cluster_centers_'):
                cluster_centers_list = model.cluster_centers_.tolist()

    except Exception as e:
        error_msg = f"模型 {model_type} 训练或评估失败: {str(e)}, {traceback.format_exc()}"
        emit_process_error(training_id, 'training', error_msg)

    train_time = time.time() - start_time
    results = {'训练时间': train_time, 'model_name': model_type}
    if mse is not None: results['均方误差'] = mse
    if recall is not None: results['召回率'] = recall
    if f1 is not None: results['F1值'] = f1
    if accuracy is not None: results['准确率'] = accuracy
    if cluster_centers_list is not None: results['聚类中心'] = cluster_centers_list
    results['message'] = current_message if current_message else f'{model_type} 评估完成'

    # 对 Sklearn 模型，我们在这里发送完成消息
    if not current_message:
        emit_process_completed(training_id, {
            'message': f'{model_type} 训练评估完成',
            'results': results
        })

    return results


# --- 模型训练主函数 (集成 SocketIO) ---
def train_model(json_data, label_column, model_name, param_data, training_id=None,stop_event=None):
    """训练模型的主入口点，处理数据并调用特定模型训练函数"""
    print(f"训练输入数据 (前200字符): {json_data[:200]}...")
    emit_process_progress(training_id, 'training',
                          {'status': 'preprocessing_start', 'message': '正在开始数据预处理...'})

    try:
        parsed_input_list = json.loads(json_data)
    except json.JSONDecodeError as e:
        error_msg = f"无效的输入JSON数据格式: {e}"
        emit_process_error(training_id,'training', error_msg)
        raise ValueError(error_msg)

    if not parsed_input_list or not isinstance(parsed_input_list, list):
        error_msg = "解析后的输入数据应为非空列表。"
        emit_process_error(training_id,'training', error_msg)
        raise ValueError(error_msg)

    all_processed_1d_samples = []
    all_corresponding_labels = []
    input_feature_dimension = None

    dnn_window_size = param_data.get("dnn_window_size", 512)
    dnn_overlapping_windows = param_data.get("dnn_overlapping_windows", False)
    dnn_step = param_data.get("dnn_step", dnn_window_size)

    total_samples_in_input = len(parsed_input_list)
    processed_sample_count = 0

    for item_idx, item in enumerate(parsed_input_list):
        if stop_event and stop_event.is_set():
            emit_process_completed(training_id, 'training_completed', {
                'status': 'stopped',
                'message': f'训练已中止 (预处理进行中，完成 {item_idx}/{total_samples_in_input} 样本)'
            })
            return None, {"error": "Training stopped by user", "message": "训练已中止"}
        original_raw_data = item.get("raw_data")
        label = item.get(label_column)
        method = item.get("method")

        # 定期发送处理进度
        if item_idx % max(1, total_samples_in_input // 10) == 0:
            emit_process_progress(training_id, 'training', {
                'status': 'preprocessing_progress',
                'message': f'正在处理输入样本 {item_idx + 1}/{total_samples_in_input}',
                'progress_percent': ((item_idx + 1) / total_samples_in_input) * 50  # 预处理占50%
            })

        if original_raw_data is None or label is None or method is None:
            print(f"警告: 跳过样本 {item_idx}，缺少关键信息。")
            continue

        current_samples_for_item = np.array([])
        current_labels_for_item = np.array([])

        try:
            label = int(label)
            data_np = np.array(original_raw_data)

            # 根据方法处理数据
            if method.endswith(METHOD_EMD) :
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    processed_1d_data = data_np[:, 0]
                    current_samples_for_item, current_labels_for_item = create_windows_1d(
                        processed_1d_data, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                else:
                    print(f"警告 ({METHOD_EMD}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method.endswith(METHOD_WAVELET):
                # 注意: 这里的转置 .T 可能需要根据你的数据实际格式调整
                if data_np.ndim == 2:
                    data_np = data_np.T
                    current_samples_for_item, current_labels_for_item = create_windows_wavelet(
                        data_np, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                else:
                    print(f"警告 ({METHOD_WAVELET}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method.endswith(METHOD_FFT):
                processed_1d_data = data_np.flatten()
                current_samples_for_item, current_labels_for_item = create_windows_1d(
                    processed_1d_data, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                # 针对 '峭度指标' 和 '直方图特征' 的修改
            elif method.endswith(METHOD_KURTOSIS):  # 单独检查峭度指标
                # 峭度通常是单个值或少数特征，确保是二维
                if data_np.ndim == 1:
                    data_np = data_np.reshape(-1, 1)
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    current_samples_for_item = data_np.astype(np.float32)
                    current_labels_for_item = np.full(len(data_np), label, dtype=int)
                else:
                    print(f"警告 ({METHOD_KURTOSIS}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method.endswith(METHOD_HISTOGRAM):  # 单独检查直方图特征
                # 直方图通常是一组特征
                if data_np.ndim == 1:
                    data_np = data_np.reshape(-1, data_np.shape[0])
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    current_samples_for_item = data_np.astype(np.float32)
                    current_labels_for_item = np.full(len(data_np), label, dtype=int)
                else:
                    print(f"警告 ({METHOD_HISTOGRAM}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            else:
                print(f"警告: 未知方法 '{method}' 用于样本 {item_idx}。")
                continue

            if current_samples_for_item.size > 0:
                all_processed_1d_samples.append(current_samples_for_item)
                all_corresponding_labels.append(current_labels_for_item)
                processed_sample_count += len(current_samples_for_item)

                if input_feature_dimension is None:
                    input_feature_dimension = current_samples_for_item.shape[1]
                elif input_feature_dimension != current_samples_for_item.shape[1]:
                    raise ValueError(
                        f"维度不匹配! {method} 产生 {current_samples_for_item.shape[1]} 维, 需要 {input_feature_dimension} 维。")
        except Exception as e:
            print(f"处理样本 {item_idx} ({method}) 时出错: {e}")
            traceback.print_exc()
            error_msg = f"处理样本 {item_idx} 时出错: {traceback.format_exc()}"
            emit_process_error(training_id,'training', error_msg)
            continue

    if not all_processed_1d_samples:
        error_msg = "没有样本被成功处理。"
        emit_process_error(training_id,'training', error_msg)
        raise ValueError(error_msg)

    X_final = np.concatenate(all_processed_1d_samples, axis=0)
    y_final = np.concatenate(all_corresponding_labels, axis=0)

    if X_final.shape[0] < 2:
        error_msg = f"有效样本数 ({len(X_final)}) 太少，无法训练。"
        emit_process_error(training_id,'training', error_msg)
        raise ValueError(error_msg)

    if input_feature_dimension is None: input_feature_dimension = X_final.shape[1]
    y_final_series = pd.Series(y_final).astype(int)
    emit_process_progress(training_id, 'training',
                          {'status': 'preprocessing_complete', 'message': f'预处理完成，生成 {len(X_final)} 样本。',
                           'progress_percent': 50})

    # --- 模型参数 ---
    #DNN:
    num_classes_param = param_data.get("num_classes", len(y_final_series.unique()))
    num_rounds_param = param_data.get("num_rounds", 5)
    lr_param = param_data.get("lr", 0.001)
    epochs_per_round_param = param_data.get("epochs_per_round", 200)
    #随机森林：
    n_estimators_param = param_data.get("n_estimators", 100)
    #逻辑回归：
    lr_max_iter = param_data.get("lr_max_iter", 1000)
    lr_C = param_data.get("lr_C", 1.0) # Inverse of regularization strength
    lr_solver = param_data.get("lr_solver", 'lbfgs') # 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
    #支持向量机：
    svc_C = param_data.get("svc_C", 1.0) # Regularization parameter
    svc_kernel = param_data.get("svc_kernel", 'rbf') # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    svc_gamma = param_data.get("svc_gamma", 'scale') # Kernel coefficient ('scale' or 'auto' or float)
    #线性回归：
    lr_fit_intercept = param_data.get("lr_fit_intercept", True)  # Whether to calculate the intercept
    #聚类：
    kmeans_n_clusters_param = param_data.get("kmeans_n_clusters", num_classes_param if num_classes_param > 0 else 2)



    dnn_hidden_dims = param_data.get("dnn_hidden_dims", [128, 64])
    dnn_dropout_rate = param_data.get("dnn_dropout_rate", 0.5)

    # --- 划分数据集 ---
    stratify_option = y_final_series if len(y_final_series.unique()) > 1 and all(
        y_final_series.value_counts() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final_series, test_size=0.2, random_state=42,
                                                        stratify=stratify_option)

    # --- 模型配置 ---
    models_config = {
        '线性回归': LinearRegression(fit_intercept=lr_fit_intercept),
        '逻辑回归': LogisticRegression(max_iter=lr_max_iter, C=lr_C, solver=lr_solver, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=n_estimators_param, random_state=42),
        '支持向量机': SVC(probability=True, C=svc_C, kernel=svc_kernel, gamma=svc_gamma, random_state=42),
        'K-均值聚类': KMeans(n_clusters=kmeans_n_clusters_param, n_init='auto', random_state=42),
        '深度神经网络': 'generic_dnn_placeholder'
    }

    if model_name not in models_config:
        error_msg = f"不支持的模型名称: {model_name}"
        emit_process_error(training_id,'training', error_msg)
        raise ValueError(error_msg)

    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_base = f"{model_name.replace(' ', '_')}_{timestamp}"  # 使用时间戳
    model_path = os.path.join(model_dir, model_filename_base)

    print(f"开始训练模型: {model_name}, 输入维度: {input_feature_dimension}, 训练样本数: {len(X_train)}")
    model_instance, result_metrics = None, {}

    # --- 调用特定模型训练 ---
    if model_name == '深度神经网络':
        if X_train.shape[0] == 0:
            result_metrics = {'训练时间': 0, '准确率': 0, 'message': '训练数据为空，DNN无法训练。'}
            emit_process_error(training_id,'training', result_metrics['message'])
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) if len(
                train_dataset) > 0 else None
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) if len(test_dataset) > 0 else None

            model_instance, result_metrics = train_generic_dnn_model(
                train_dataloader, test_dataloader, input_feature_dimension, num_classes_param,
                num_rounds_param, epochs_per_round_param, lr_param,
                dnn_hidden_dims, dnn_dropout_rate, training_id
            )
            if model_instance:
                try:
                    torch.save(model_instance.state_dict(), f"{model_path}.pth")
                    joblib.dump(scaler, f"{model_path}_scaler.joblib")
                    print(f"DNN 模型和 Scaler 已保存到: {model_path}.pth / .joblib")
                except Exception as e:
                    print(f"保存 DNN 模型或 Scaler 失败: {e}")
    else:
        model_instance = models_config[model_name]
        result_metrics = evaluate_sklearn_model(
            model_instance, X_train, X_test, y_train, y_test,
            model_name, num_classes_param, training_id
        )
        if model_instance and '失败' not in result_metrics.get('message', ''):
            try:
                joblib.dump(model_instance, f"{model_path}.joblib")
                print(f"模型 {model_name} 已保存到: {model_path}.joblib")
            except Exception as e:
                print(f"保存模型 {model_name} 失败: {e}")

    print(f"模型 {model_name} 训练完成。结果: {result_metrics}")
    return model_instance, result_metrics











