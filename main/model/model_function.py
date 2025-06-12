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
external_emit_functions = None


def set_socketio_instance(socketio_obj):
    """设置全局 SocketIO 实例"""
    global socketio_instance
    socketio_instance = socketio_obj


def set_emit_functions(emit_funcs_dict):
    """设置外部 emit 函数引用"""
    global external_emit_functions
    external_emit_functions = emit_funcs_dict


# --- WebSocket Emitter 辅助函数 ---
def _emit_event_to_websocket(event_name, training_id, data_dict):
    """通过 WebSocket 发送事件的辅助函数"""
    if external_emit_functions and training_id:
        # 使用外部提供的 emit 函数
        # emit_func = external_emit_functions.get(event_name.replace('training_', '').replace('_result', ''))
        emit_func = external_emit_functions.get(event_name)
        if emit_func:
            try:
                emit_func(training_id, data_dict)
            except Exception as e:
                print(f"外部 emit 函数调用失败 '{event_name}': {e}")
                traceback.print_exc()
        else:
            print(f"未找到对应的 emit 函数: {event_name}")
    elif socketio_instance and training_id:
        # 备用：直接使用 socketio 实例
        payload_to_send = {"session_id": training_id, "timestamp": datetime.datetime.now().isoformat()}
        payload_to_send.update(data_dict)
        try:
            socketio_instance.emit(event_name, payload_to_send, namespace='/training', room=training_id)
        except Exception as e:
            print(f"SocketIO 发送事件 '{event_name}' 失败: {e}")
            traceback.print_exc()
    else:
        print(f"WebSocket 发送失败：未设置 emit 函数或 training_id 为空 (training_id: {training_id})")


# --- WebSocket Emitter 具体函数 ---
def emit_training_progress(training_id, data):
    """发送通用训练进度消息"""
    _emit_event_to_websocket('training_progress', training_id, data)


def emit_epoch_result(training_id, data):
    """发送单个 epoch 结果"""
    _emit_event_to_websocket('epoch_result', training_id, data)


def emit_round_result(training_id, data):
    """发送轮次结果"""
    _emit_event_to_websocket('round_result', training_id, data)


def emit_training_completed(training_id, data):
    """发送训练完成消息"""
    _emit_event_to_websocket('training_completed', training_id, data)


def emit_training_error(training_id, error_msg, details=None):
    """发送训练错误消息"""
    payload = {'error': str(error_msg)}
    if details:
        payload['details'] = str(details)
    _emit_event_to_websocket('training_error', training_id, payload)


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
                            hidden_dims=[128, 64], dropout_rate=0.5, training_id=None):
    """训练通用 DNN 模型，并通过 WebSocket 发送实时更新"""
    error_message = None
    if input_dim <= 0:
        error_message = f"DNN 输入维度 ({input_dim}) 必须大于 0。"
    elif num_classes <= 0:
        error_message = f"DNN 分类类别数 ({num_classes}) 必须大于 0。"
    total_epochs = num_rounds * epochs_per_round
    if not error_message and total_epochs <= 0:
        error_message = f"总训练 epochs ({total_epochs}) 必须大于 0。轮次: {num_rounds}, 每轮epochs: {epochs_per_round}"

    if error_message:
        print(f"错误: {error_message}")
        emit_training_error(training_id, error_message)  # 发送错误
        return None, {"训练时间": 0, "准确率": 0, "message": error_message}

    model = GenericDNN(input_dim, num_classes, hidden_dims, dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    print(f"开始DNN训练，共 {num_rounds} 轮, 每轮 {epochs_per_round} epochs。设备: {device}")

    # 发送训练开始进度
    emit_training_progress(training_id, {
        'status': 'started_dnn', 'message': '正在开始DNN训练...',
        'total_rounds': num_rounds, 'epochs_per_round': epochs_per_round, 'total_epochs': total_epochs,
    })

    best_accuracy = 0.0
    best_model_state = None
    global_epoch_counter = 0
    last_epoch_loss = 0.0
    training_history = {'rounds': [], 'epochs': [], 'losses': [], 'accuracies': [], 'best_accuracy': 0.0}

    try:
        for round_idx in range(num_rounds):
            print(f"------ 开始训练轮次 {round_idx + 1}/{num_rounds} ------")
            emit_round_result(training_id, {  # 发送轮次开始
                'status': 'started', 'current_round': round_idx + 1, 'total_rounds': num_rounds
            })

            round_losses = []
            round_accuracies = []

            for epoch_idx_in_round in range(epochs_per_round):
                actual_epoch_num = global_epoch_counter
                model.train()
                running_loss = 0.0

                if not train_dataloader:
                    print(f"警告: 训练数据加载器为空，跳过 Epoch {actual_epoch_num + 1}。")
                    epoch_loss = 0.0
                else:
                    for i, (inputs, labels) in enumerate(train_dataloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    epoch_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
                last_epoch_loss = epoch_loss

                current_accuracy = 0.0
                if test_dataloader is not None and len(test_dataloader) > 0:
                    model.eval()
                    all_predictions_epoch, all_true_labels_epoch = [], []
                    with torch.no_grad():
                        for inputs_test, labels_test in test_dataloader:
                            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                            outputs_test = model(inputs_test)
                            _, predicted_test = torch.max(outputs_test.data, 1)
                            all_predictions_epoch.extend(predicted_test.cpu().numpy())
                            all_true_labels_epoch.extend(labels_test.cpu().numpy())
                    if len(all_true_labels_epoch) > 0:
                        current_accuracy = accuracy_score(all_true_labels_epoch, all_predictions_epoch)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_model_state = model.state_dict()

                round_losses.append(epoch_loss)
                round_accuracies.append(current_accuracy)
                training_history['epochs'].append(actual_epoch_num + 1)
                training_history['losses'].append(epoch_loss)
                training_history['accuracies'].append(current_accuracy)
                training_history['best_accuracy'] = best_accuracy

                # --- 发送 Epoch 结果 ---
                progress_data = {
                    'current_round': round_idx + 1, 'total_rounds': num_rounds,
                    'epoch_in_round': epoch_idx_in_round + 1, 'epochs_per_round': epochs_per_round,
                    'global_epoch': actual_epoch_num + 1, 'total_epochs': total_epochs,
                    'loss': float(epoch_loss), 'accuracy': float(current_accuracy),
                    'best_accuracy': float(best_accuracy),
                    'progress_percent': ((actual_epoch_num + 1) / total_epochs) * 100
                }
                emit_epoch_result(training_id, progress_data)
                # --- 结束发送 ---

                log_frequency = max(1, epochs_per_round // 5)
                if (epoch_idx_in_round + 1) % log_frequency == 0 or (epoch_idx_in_round + 1) == epochs_per_round:
                    print(
                        f'  轮次 [{round_idx + 1}/{num_rounds}], Epoch [{epoch_idx_in_round + 1}/{epochs_per_round}] '
                        f'(总计 [{actual_epoch_num + 1}/{total_epochs}]), '
                        f'Loss: {epoch_loss:.4f}, Acc: {current_accuracy:.4f}, Best Acc: {best_accuracy:.4f}')
                global_epoch_counter += 1

            round_avg_loss = np.mean(round_losses) if round_losses else 0.0
            round_avg_accuracy = np.mean(round_accuracies) if round_accuracies else 0.0
            training_history['rounds'].append({
                'round': round_idx + 1, 'avg_loss': round_avg_loss, 'avg_accuracy': round_avg_accuracy,
            })

            emit_round_result(training_id, {  # 发送轮次结束
                'status': 'completed', 'current_round': round_idx + 1,
                'round_avg_loss': float(round_avg_loss), 'round_avg_accuracy': float(round_avg_accuracy),
                'best_accuracy_so_far': float(best_accuracy)
            })
            print(f"------ 完成训练轮次 {round_idx + 1}/{num_rounds} (最佳准确率: {best_accuracy:.4f}) ------")

    except Exception as e:
        train_time = time.time() - start_time
        error_msg = f"DNN 训练过程中发生错误: {str(e)}"
        print(f"错误: {error_msg}\n{traceback.format_exc()}")
        emit_training_error(training_id, error_msg, traceback.format_exc())
        return None, {"训练时间": train_time, "准确率": best_accuracy, "message": error_msg,
                      "训练历史": training_history}

    train_time = time.time() - start_time
    print(f"DNN训练总耗时: {train_time:.2f} 秒")

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型状态，准确率: {best_accuracy:.4f}")
    else:
        print("警告: 未找到最佳模型状态。")

    results = {
        '训练时间': train_time, '准确率': best_accuracy,
        '最终Loss': last_epoch_loss if global_epoch_counter > 0 else None,
        '总轮次': num_rounds, '每轮Epochs': epochs_per_round, '总Epochs': total_epochs,
        '训练历史': training_history, 'message': 'DNN训练成功完成'
    }

    # DNN 训练完成，发送完成消息
    emit_training_completed(training_id, {
        'message': 'DNN训练成功完成',
        'results': results
    })

    return model, results


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

    emit_training_progress(training_id, {
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
        emit_training_error(training_id, current_message)
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
                emit_training_error(training_id, current_message)
                return {'训练时间': 0, 'message': current_message}

            emit_training_progress(training_id, {'status': 'fitting', 'message': f'正在拟合 {model_type}...',
                                                 'progress_percent': 30})
            model.fit(X_train, y_train)
            emit_training_progress(training_id, {'status': 'evaluating', 'message': f'正在评估 {model_type}...',
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
            emit_training_progress(training_id, {'status': 'fitting', 'message': f'正在拟合 {model_type}...',
                                                 'progress_percent': 50})
            model.fit(X_train)
            if hasattr(model, 'cluster_centers_'):
                cluster_centers_list = model.cluster_centers_.tolist()

    except Exception as e:
        current_message = f"模型 {model_type} 训练或评估失败: {str(e)}"
        print(f"错误: {current_message}\n{traceback.format_exc()}")
        emit_training_error(training_id, current_message, traceback.format_exc())

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
        emit_training_completed(training_id, {
            'message': f'{model_type} 训练评估完成',
            'results': results
        })

    return results


# --- 模型训练主函数 (集成 SocketIO) ---
def train_model(json_data, label_column, model_name, param_data, training_id=None):
    """训练模型的主入口点，处理数据并调用特定模型训练函数"""
    print(f"训练输入数据 (前200字符): {json_data[:200]}...")
    emit_training_progress(training_id, {'status': 'preprocessing_start', 'message': '正在开始数据预处理...'})

    try:
        parsed_input_list = json.loads(json_data)
    except json.JSONDecodeError as e:
        error_msg = f"无效的输入JSON数据格式: {e}"
        emit_training_error(training_id, error_msg)
        raise ValueError(error_msg)

    if not parsed_input_list or not isinstance(parsed_input_list, list):
        error_msg = "解析后的输入数据应为非空列表。"
        emit_training_error(training_id, error_msg)
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
        original_raw_data = item.get("raw_data")
        label = item.get(label_column)
        method = item.get("method")

        # 定期发送预处理进度
        if item_idx % max(1, total_samples_in_input // 10) == 0:
            emit_training_progress(training_id, {
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
            if method == METHOD_EMD:
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    processed_1d_data = data_np[:, 0]
                    current_samples_for_item, current_labels_for_item = create_windows_1d(
                        processed_1d_data, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                else:
                    print(f"警告 ({METHOD_EMD}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method == METHOD_WAVELET:
                # 注意: 这里的转置 .T 可能需要根据你的数据实际格式调整
                if data_np.ndim == 2:
                    data_np = data_np.T
                    current_samples_for_item, current_labels_for_item = create_windows_wavelet(
                        data_np, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                else:
                    print(f"警告 ({METHOD_WAVELET}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method == METHOD_FFT:
                processed_1d_data = data_np.flatten()
                current_samples_for_item, current_labels_for_item = create_windows_1d(
                    processed_1d_data, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
            elif method in [METHOD_KURTOSIS, METHOD_HISTOGRAM]:  # 假设这些是预计算好的特征
                if data_np.ndim == 1: data_np = data_np.reshape(-1,
                                                                1 if method == METHOD_KURTOSIS else data_np.shape[0])
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    current_samples_for_item = data_np.astype(np.float32)
                    current_labels_for_item = np.full(len(data_np), label, dtype=int)
                else:
                    print(f"警告 ({method}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
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
            emit_training_error(training_id, f"处理样本 {item_idx} 时出错", traceback.format_exc())
            continue

    if not all_processed_1d_samples:
        error_msg = "没有样本被成功处理。"
        emit_training_error(training_id, error_msg)
        raise ValueError(error_msg)

    X_final = np.concatenate(all_processed_1d_samples, axis=0)
    y_final = np.concatenate(all_corresponding_labels, axis=0)

    if X_final.shape[0] < 2:
        error_msg = f"有效样本数 ({len(X_final)}) 太少，无法训练。"
        emit_training_error(training_id, error_msg)
        raise ValueError(error_msg)

    if input_feature_dimension is None: input_feature_dimension = X_final.shape[1]
    y_final_series = pd.Series(y_final).astype(int)
    emit_training_progress(training_id, {'status': 'preprocessing_complete', 'message': f'预处理完成，生成 {len(X_final)} 样本。', 'progress_percent': 50})

    # --- 模型参数 ---
    num_classes_param = param_data.get("num_classes", len(y_final_series.unique()))
    lr_param = param_data.get("lr", 0.001)
    n_estimators_param = param_data.get("n_estimators", 100)
    num_rounds_param = param_data.get("num_rounds", 1)
    epochs_per_round_param = param_data.get("epochs_per_round", 100)
    dnn_hidden_dims = param_data.get("dnn_hidden_dims", [128, 64])
    dnn_dropout_rate = param_data.get("dnn_dropout_rate", 0.5)
    kmeans_n_clusters_param = param_data.get("kmeans_n_clusters", num_classes_param if num_classes_param > 0 else 2)

    # --- 划分数据集 ---
    stratify_option = y_final_series if len(y_final_series.unique()) > 1 and all(y_final_series.value_counts() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final_series, test_size=0.2, random_state=42, stratify=stratify_option)

    # --- 模型配置 ---
    models_config = {
        '线性回归': LinearRegression(),
        '逻辑回归': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=n_estimators_param, random_state=42),
        '支持向量机': SVC(probability=True, random_state=42),
        'K-均值聚类': KMeans(n_clusters=kmeans_n_clusters_param, n_init='auto', random_state=42),
        '深度神经网络': 'generic_dnn_placeholder'
    }

    if model_name not in models_config:
        error_msg = f"不支持的模型名称: {model_name}"
        emit_training_error(training_id, error_msg)
        raise ValueError(error_msg)

    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_base = f"{model_name.replace(' ', '_')}_{timestamp}" # 使用时间戳
    model_path = os.path.join(model_dir, model_filename_base)

    print(f"开始训练模型: {model_name}, 输入维度: {input_feature_dimension}, 训练样本数: {len(X_train)}")
    model_instance, result_metrics = None, {}

    # --- 调用特定模型训练 ---
    if model_name == '深度神经网络':
        if X_train.shape[0] == 0:
            result_metrics = {'训练时间': 0, '准确率': 0, 'message': '训练数据为空，DNN无法训练。'}
            emit_training_error(training_id, result_metrics['message'])
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
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) if len(train_dataset) > 0 else None
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
