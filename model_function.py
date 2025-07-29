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
from sklearn.metrics import recall_score, f1_score, mean_squared_error, accuracy_score, silhouette_score
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


# # --- 通用 DNN 模型 (包含 CNN 层) ---
# class GenericDNN(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout_rate=0.1):
#         super(GenericDNN, self).__init__()
#
#         # --- CNN 特征提取部分 ---
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#
#         # --- 计算卷积层输出的展平维度 ---
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 1, input_dim)
#             dummy_output_conv1 = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
#             dummy_output_conv2 = self.pool2(self.relu2(self.bn2(self.conv2(dummy_output_conv1))))
#             self.flattened_dim = dummy_output_conv2.view(1, -1).size(1)
#
#         # --- 全连接分类部分 ---
#         layers = []
#         current_dim = self.flattened_dim
#         for h_dim in hidden_dims:
#             layers.append(nn.Linear(current_dim, h_dim))
#             layers.append(nn.ReLU())
#             if dropout_rate > 0:
#                 layers.append(nn.Dropout(dropout_rate))
#             current_dim = h_dim
#         layers.append(nn.Linear(current_dim, num_classes))
#         self.classifier_layers = nn.Sequential(*layers)
#
#         # --- 权重初始化 ---
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         """改进的权重初始化"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # 添加数值稳定性检查
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("警告: 输入包含NaN或Inf值")
#
#         x = x.unsqueeze(1)  # 增加通道维度
#         x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#         x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#         x = x.view(x.size(0), -1)  # 展平
#
#         # 添加数值检查
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("警告: 卷积层输出包含NaN或Inf值")
#
#         output = self.classifier_layers(x)
#         return output



# --- 简化的两层 CNN 模型 ---
class GenericDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout_rate=0.3): # 建议将默认 dropout_rate 提高一点，例如 0.3
        super(GenericDNN, self).__init__()

        # --- CNN 特征提取部分 ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- 计算卷积层输出的展平维度 ---
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            dummy_output_conv1 = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            dummy_output_conv2 = self.pool2(self.relu2(self.bn2(self.conv2(dummy_output_conv1))))
            self.flattened_dim = dummy_output_conv2.view(1, -1).size(1)

        # --- 新增：Dropout 层 ---
        # 只有当 dropout_rate > 0 时才创建 Dropout 层
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # --- 输出层 (直接连接到分类) ---
        self.output_layer = nn.Linear(self.flattened_dim, num_classes)

        # --- 权重初始化 ---
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化，应用于卷积层、线性层和BatchNorm层"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 添加通道维度：从 (batch_size, input_dim) 变为 (batch_size, 1, input_dim)
        x = x.unsqueeze(1)

        # 通过第一层卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 通过第二层卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 展平特征图，为全连接层做准备
        x = x.view(x.size(0), -1)

        # --- 新增：应用 Dropout ---
        # 只有在训练模式下 Dropout 才会起作用，评估模式下会被自动禁用
        x = self.dropout(x)

        # 通过输出层进行分类
        output = self.output_layer(x)
        return output


# --- 通用 DNN 模型的训练函数 (集成 SocketIO) ---
def train_generic_dnn_model(train_dataloader, val_dataloader, test_dataloader, input_dim, num_classes,
                            num_rounds=1, epochs_per_round=20, lr=0.001,
                            hidden_dims=[128, 64], dropout_rate=0.3, training_id=None, stop_event=None):
    """训练通用DNN模型 (每个epoch后都进行验证评估，训练结束后进行测试评估)"""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始DNN训练，共 {num_rounds} 轮, 每轮 {epochs_per_round} epochs。设备: {device}")

    model = GenericDNN(input_dim, num_classes, hidden_dims, dropout_rate).to(device)

    # --- 改进的损失函数和优化器 ---
    # 使用标签平滑来提高稳定性
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, pred, target):
            log_prob = torch.log_softmax(pred, dim=-1)
            weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
            weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
            loss = (-weight * log_prob).sum(dim=-1).mean()
            return loss

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

    # 使用AdamW优化器，更稳定
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


    best_model_state = None
    best_val_accuracy = 0.0 # 基于验证集准确率选择最佳模型
    all_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    round_results = []
    final_test_results = {} # 用于存储最终测试集结果

    for round_idx in range(num_rounds):
        if stop_event and stop_event.is_set():
            print(f"DNN 训练在第 {round_idx + 1} 轮时被用户中止。")
            return model, {"error": "Training stopped by user", "message": "训练已中止"}

        print(f"------ 开始训练轮次 {round_idx + 1}/{num_rounds} ------")

        round_losses = []
        round_train_accuracies = []
        round_val_accuracies = []

        for epoch in range(epochs_per_round):
            if stop_event and stop_event.is_set():
                print(f"DNN 训练在第 {round_idx + 1} 轮的第 {epoch + 1} epoch 时被用户中止。")
                return model, {"error": "Training stopped by user", "message": "训练已中止"}

            # --- 1. 训练模式 ---
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)

            avg_loss = total_loss / len(train_dataloader)
            train_accuracy = train_correct / train_total

            round_losses.append(avg_loss)
            round_train_accuracies.append(train_accuracy)
            all_losses.append(avg_loss)
            all_train_accuracies.append(train_accuracy)

            # --- 2. 验证模式 (每个epoch后立即执行) ---
            model.eval()
            val_correct = 0
            val_total = 0
            val_accuracy = 0.0

            if val_dataloader:
                with torch.no_grad():
                    for data, target in val_dataloader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        val_correct += pred.eq(target.view_as(pred)).sum().item()
                        val_total += target.size(0)

                if val_total > 0:
                    val_accuracy = val_correct / val_total

            round_val_accuracies.append(val_accuracy)
            all_val_accuracies.append(val_accuracy)

            # --- 3. 更新最佳模型 (基于验证集准确率) ---
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                print(f"  * 新的最佳验证准确率: {best_val_accuracy:.4f} *")

            # --- 4. 打印和发送进度 ---
            total_epoch_count = epoch + 1 + round_idx * epochs_per_round
            if (epoch + 1) % 20 == 0 or epoch == epochs_per_round - 1:
                print(f"  轮次 [{round_idx + 1}/{num_rounds}], Epoch [{epoch + 1}/{epochs_per_round}] "
                      f"(总计 [{total_epoch_count}/{num_rounds * epochs_per_round}]), "
                      f"Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Best Val Acc: {best_val_accuracy:.4f}")

                if training_id:
                    progress_data = {
                    'round': round_idx + 1,
                    'epoch': epoch + 1,
                    'total_epochs': num_rounds * epochs_per_round,
                    'current_epoch': total_epoch_count,
                    'loss': float(avg_loss),
                    'accuracy': float(train_accuracy),
                    'val_accuracy': float(val_accuracy),
                    'best_val_accuracy': float(best_val_accuracy)
                    }
                    emit_epoch_result(training_id, progress_data)

        # --- 回合结束，计算并发送回合平均结果 ---
        avg_round_loss = sum(round_losses) / len(round_losses) if round_losses else 0
        avg_round_train_accuracy = sum(round_train_accuracies) / len(
            round_train_accuracies) if round_train_accuracies else 0
        avg_round_val_accuracy = sum(round_val_accuracies) / len(
            round_val_accuracies) if round_val_accuracies else 0

        round_summary = {
            'round': round_idx + 1,
            'avg_loss': avg_round_loss,
            'avg_train_accuracy': avg_round_train_accuracy,
            'avg_val_accuracy': avg_round_val_accuracy
        }
        round_results.append(round_summary)

        print(f"------ 完成训练轮次 {round_idx + 1}/{num_rounds} (当前最佳验证准确率: {best_val_accuracy:.4f}) ------")

        if training_id:
            round_data = {
                'round': round_idx + 1,
                'total_rounds': num_rounds,
                'avg_loss': float(avg_round_loss),
                'avg_accuracy': float(avg_round_train_accuracy),
                'avg_val_accuracy': float(avg_round_val_accuracy),
                'best_val_accuracy': float(best_val_accuracy)
            }
            emit_round_result(training_id, round_data)

    # --- 所有训练轮次结束后，加载最佳模型并对测试集进行最终评估 ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型状态 (基于验证集)，准备对测试集进行最终评估。")

        if test_dataloader:
            model.eval()
            test_correct = 0
            test_total = 0
            y_pred_list = []
            y_true_list = []

            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
                    y_pred_list.extend(pred.cpu().numpy().flatten())
                    y_true_list.extend(target.cpu().numpy().flatten())

            final_test_accuracy = test_correct / test_total if test_total > 0 else 0.0
            final_test_recall = recall_score(y_true_list, y_pred_list, average='macro', zero_division=0)
            final_test_f1 = f1_score(y_true_list, y_pred_list, average='macro', zero_division=0)

            final_test_results = {
                'accuracy': float(final_test_accuracy),
                'recall': float(final_test_recall),
                'f1': float(final_test_f1),
                'message': 'DNN 最终测试集评估完成'
            }
            print(f"最终测试集结果 - 准确率: {final_test_accuracy:.4f}, 召回率: {final_test_recall:.4f}, F1值: {final_test_f1:.4f}")
        else:
            final_test_results = {'message': '没有提供测试集，跳过最终测试评估。'}
            print("没有提供测试集，跳过 DNN 最终测试评估。")

    else:
        final_test_results = {'message': '没有找到最佳模型状态，无法进行最终测试评估。'}
        print("没有找到最佳模型状态，无法进行 DNN 最终测试评估。")


    return model, {
        '训练时间': time.time() - start_time,
        '最佳验证准确率': best_val_accuracy,
        '最终Loss': all_losses[-1] if all_losses else 0,
        '总轮次': num_rounds,
        '每轮Epochs': epochs_per_round,
        '总Epochs': num_rounds * epochs_per_round,
        '训练历史': {
            'rounds': round_results,
            'epochs': list(range(1, num_rounds * epochs_per_round + 1)),
            'losses': all_losses,
            'train_accuracies': all_train_accuracies,
            'val_accuracies': all_val_accuracies,
            'best_val_accuracy': best_val_accuracy
        },
        '最终测试结果': final_test_results, # 新增最终测试结果
        'message': 'DNN训练成功完成'
    }


# --- 模型评估函数 (用于非 DNN 模型，添加 SocketIO 支持) ---
# 此函数现在只用于在训练结束后对测试集进行评估，不再负责训练过程
def evaluate_sklearn_model_on_test_set(model, X_test_df, y_test_series, model_type, num_classes=None):
    """评估 Scikit-learn 模型在独立测试集上的表现"""
    start_time = time.time()
    mse, recall, f1, accuracy, silhouette_score_val = None, None, None, None, None
    cluster_centers_list = None
    current_message = "测试集评估完成"

    X_test, y_test = X_test_df, y_test_series

    if X_test.size == 0 or (y_test is not None and y_test.size == 0 and model_type != 'K-均值聚类'):
        current_message = f"模型 {model_type} 的测试数据或标签为空，无法评估。"
        print(f"警告: {current_message}")
        return {'测试时间': 0, 'message': current_message}

    try:
        if model_type in ['线性回归', '逻辑回归', '随机森林', '支持向量机']:
            y_pred = model.predict(X_test)
            if model_type == '线性回归':
                mse = mean_squared_error(y_test, y_pred)
            else:
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        elif model_type == 'K-均值聚类':
            # K-Means 在测试集上做预测（分配到最近的簇），然后评估
            labels_pred_test = model.predict(X_test) # 使用训练好的模型对测试集进行预测

            if X_test.shape[0] > 1 and model.n_clusters > 1 and len(np.unique(labels_pred_test)) > 1:
                try:
                    silhouette_score_val = silhouette_score(X_test, labels_pred_test)
                except Exception as e:
                    print(f"计算 K-Means 轮廓系数时出错: {e}")
            else:
                print("警告: 测试数据不足或聚类数量不满足要求，跳过 K-Means 轮廓系数计算。")

            if hasattr(model, 'cluster_centers_'):
                cluster_centers_list = model.cluster_centers_.tolist()

    except Exception as e:
        current_message = f"模型 {model_type} 测试评估失败: {str(e)}, {traceback.format_exc()}"
        print(f"错误: {current_message}")
        return {'测试时间': 0, 'message': current_message}

    test_time = time.time() - start_time
    results = {'测试时间': test_time, 'model_name': model_type}
    if mse is not None: results['均方误差'] = mse
    if recall is not None: results['召回率'] = recall
    if f1 is not None: results['F1值'] = f1
    if accuracy is not None: results['准确率'] = accuracy
    if cluster_centers_list is not None: results['聚类中心'] = cluster_centers_list
    if silhouette_score_val is not None: results['轮廓系数'] = float(silhouette_score_val)
    results['message'] = current_message

    return results


# --- 模型训练主函数 (集成 SocketIO) ---
def train_model(json_data, label_column, model_name, param_data, training_id=None, stop_event=None):
    start_time = time.time()
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
            print(f"数据预处理在处理第 {item_idx + 1} 个样本时被中止。")
            return None, {"error": "Training stopped by user", "message": "训练在预处理阶段已中止"}, None

        progress_percent = 10 + (item_idx / total_samples_in_input) * 40
        emit_process_progress(
            training_id, 'training', {
                'status': 'preprocessing_progress',
                'message': f'正在处理输入样本 {item_idx + 1}/{total_samples_in_input}',
                'progress_percent': progress_percent
            })

        original_raw_data = item.get("raw_data")
        label = item.get(label_column)
        method = item.get("method")

        if original_raw_data is None or label is None or method is None:
            print(f"警告: 跳过样本 {item_idx}，缺少关键信息。")
            continue

        current_samples_for_item = np.array([])
        current_labels_for_item = np.array([])

        try:
            label = int(label)
            data_np = np.array(original_raw_data)

            if method.endswith(METHOD_EMD) :
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    processed_1d_data = data_np[:, 0]
                    current_samples_for_item, current_labels_for_item = create_windows_1d(
                        processed_1d_data, label, dnn_window_size, dnn_step, dnn_overlapping_windows)
                else:
                    print(f"警告 ({METHOD_EMD}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method.endswith(METHOD_WAVELET):
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
            elif method.endswith(METHOD_KURTOSIS):
                if data_np.ndim == 1:
                    data_np = data_np.reshape(-1, 1)
                if data_np.ndim == 2 and data_np.shape[1] > 0:
                    current_samples_for_item = data_np.astype(np.float32)
                    current_labels_for_item = np.full(len(data_np), label, dtype=int)
                else:
                    print(f"警告 ({METHOD_KURTOSIS}): 样本 {item_idx} 形状 {data_np.shape} 不符。")
            elif method.endswith(METHOD_HISTOGRAM):
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
    num_classes_param = param_data.get("num_classes", len(y_final_series.unique()))
    num_rounds_param = param_data.get("num_rounds", 5)
    lr_param = param_data.get("lr", 0.001)
    epochs_per_round_param = param_data.get("epochs_per_round", 200)
    n_estimators_param = param_data.get("n_estimators", 100)
    lr_max_iter = param_data.get("lr_max_iter", 1000)
    lr_C = param_data.get("lr_C", 1.0)
    lr_solver = param_data.get("lr_solver", 'lbfgs')
    svc_C = param_data.get("svc_C", 1.0)
    svc_kernel = param_data.get("svc_kernel", 'rbf')
    svc_gamma = param_data.get("svc_gamma", 'scale')
    lr_fit_intercept = param_data.get("lr_fit_intercept", True)
    kmeans_n_clusters_param = param_data.get("kmeans_n_clusters", 5)
    dnn_hidden_dims = param_data.get("dnn_hidden_dims", [128, 64])
    dnn_dropout_rate = param_data.get("dnn_dropout_rate", 0.3)

    # --- 划分数据集：70% 训练，20% 验证，10% 测试 ---
    stratify_option = y_final_series if len(y_final_series.unique()) > 1 and all(
        y_final_series.value_counts() >= 2) else None

    # 第一次划分：70% 训练，30% 临时（验证+测试）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_final, y_final_series, test_size=0.3, random_state=42, stratify=stratify_option
    )

    # 第二次划分：从临时集中划分，使得验证集占总体的 20%，测试集占总体的 10%
    # 计算 test_size = 0.1 / 0.3 = 1/3 (临时集中的 1/3 作为测试集)
    test_size_from_temp = 0.1 / 0.3
    stratify_temp_option = y_temp if len(y_temp.unique()) > 1 and all(
        y_temp.value_counts() >= 2) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size_from_temp, random_state=42, stratify=stratify_temp_option
    )

    print(f"数据划分完成：训练集 {len(X_train)} 样本，验证集 {len(X_val)} 样本，测试集 {len(X_test)} 样本。")
    emit_process_progress(training_id, 'training',
                          {'status': 'data_split_complete', 'message': '数据划分完成',
                           'train_samples': len(X_train), 'val_samples': len(X_val), 'test_samples': len(X_test),
                           'progress_percent': 60})


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
    model_filename_base = f"{model_name.replace(' ', '_')}_{timestamp}"
    model_path = os.path.join(model_dir, model_filename_base)

    print(f"开始训练模型: {model_name}, 输入维度: {input_feature_dimension}, 训练样本数: {len(X_train)}")
    model_instance, result_metrics = None, {}
    final_test_results = {} # 初始化最终测试结果


    # --- 调用特定模型训练 ---
    if model_name == '深度神经网络':
        if X_train.shape[0] == 0:
            result_metrics = {'训练时间': 0, '最佳验证准确率': 0, 'message': '训练数据为空，DNN无法训练。'}
            emit_process_error(training_id,'training', result_metrics['message'])
        else:
            # DNN 需要标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test) # 测试集也需要标准化，但用训练集的fit参数

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) if len(train_dataset) > 0 else None
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False) if len(val_dataset) > 0 else None
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) if len(test_dataset) > 0 else None


            model_instance, result_metrics = train_generic_dnn_model(
                train_dataloader, val_dataloader, test_dataloader, input_feature_dimension, num_classes_param,
                num_rounds_param, epochs_per_round_param, lr_param,
                dnn_hidden_dims, dnn_dropout_rate, training_id, stop_event
            )
            final_test_results = result_metrics.get('最终测试结果', {}) # 从DNN训练结果中获取最终测试结果

            if model_instance:
                try:
                    torch.save(model_instance.state_dict(), f"{model_path}.pth")
                    joblib.dump(scaler, f"{model_path}_scaler.joblib")
                    # -------------------- 新增：保存类别信息 --------------------
                    # 获取训练数据中的所有类别（按出现顺序排序，确保一致性）
                    unique_classes = sorted(y_final_series.unique().tolist())
                    num_classes = len(unique_classes)
                    # 保存类别元数据（包含类别列表和数量）
                    metadata = {
                        "classes": unique_classes,  # 类别列表（如 [0,1,2] 或 ['Class_0', 'Class_1', 'Class_2']）
                        "num_classes": num_classes,
                        "label_column": label_column  # 标签列名称（可选，用于追溯）
                    }
                    joblib.dump(metadata, f"{model_path}_metadata.joblib")
                    # ---------------------------------------------------------

                    print(f"DNN 模型、Scaler 和类别元数据已保存到: {model_path}.pth / .joblib / _metadata.joblib")
                except Exception as e:
                    print(f"保存 DNN 模型或 Scaler 失败: {e}")
    else: # Scikit-learn 模型
        model_instance = models_config[model_name]

        # 为需要标准化的Scikit-learn模型进行标准化
        scaler = None
        X_train_for_model = X_train
        X_val_for_model = X_val
        X_test_for_model = X_test

        if model_name in ['逻辑回归', '支持向量机', 'K-均值聚类']: # 根据模型特性决定是否需要标准化
            scaler = StandardScaler()
            X_train_for_model = scaler.fit_transform(X_train)
            X_val_for_model = scaler.transform(X_val) # 验证集和测试集使用训练集fit的scaler
            X_test_for_model = scaler.transform(X_test)
            try:
                joblib.dump(scaler, f"{model_path}_scaler.joblib")
                print(f"Scaler 已为 {model_name} 保存到: {model_path}_scaler.joblib")
            except Exception as e:
                print(f"保存 {model_name} 的 Scaler 失败: {e}")

        # --- 训练 Scikit-learn 模型 ---
        emit_process_progress(training_id, 'training', {
            'status': f'started_{model_name.lower()}', 'message': f'正在开始 {model_name} 训练...',
            'model_name': model_name, 'progress_percent': 70
        })

        current_message = None
        try:
            if model_name in ['线性回归', '逻辑回归', '随机森林', '支持向量机']:
                if y_train.size == 0:
                    current_message = f"模型 {model_name} 的训练标签为空。"
                elif len(X_train_for_model) != len(y_train):
                    current_message = f"模型 {model_name} 训练数据和标签数量不匹配。"
                if current_message:
                    raise ValueError(current_message)

                if stop_event and stop_event.is_set():
                    print(f"模型 {model_name} 训练在开始拟合前被中止。")
                    return model_instance, {'训练时间': 0, 'message': '训练已中止'}, None

                model_instance.fit(X_train_for_model, y_train)
                emit_process_progress(training_id, 'training',
                                      {'status': 'training_complete', 'message': f'{model_name} 训练完成。',
                                       'progress_percent': 90})

                # Scikit-learn模型没有epoch概念，这里模拟一个简单的训练结果
                y_pred_train = model_instance.predict(X_train_for_model)
                train_acc = accuracy_score(y_train, y_pred_train) if model_name != '线性回归' else None

                # 验证集评估（为了选择最佳模型或监控过拟合，但这里没有迭代训练，所以只是展示）
                val_acc = None
                if X_val_for_model.size > 0 and (model_name == 'K-均值聚类' or y_val.size > 0):
                    if model_name != 'K-均值聚类':
                        y_pred_val = model_instance.predict(X_val_for_model)
                        val_acc = accuracy_score(y_val, y_pred_val)
                    else:
                        # K-Means 对验证集进行预测，然后评估轮廓系数
                        labels_pred_val = model_instance.predict(X_val_for_model)
                        if X_val_for_model.shape[0] > 1 and model_instance.n_clusters > 1 and len(np.unique(labels_pred_val)) > 1:
                            val_acc = silhouette_score(X_val_for_model, labels_pred_val)
                        else:
                            val_acc = 0.0 # 无法计算

                result_metrics = {
                    '训练时间': time.time() - start_time,
                    '训练准确率': train_acc,
                    '验证准确率': val_acc,
                    'message': f'{model_name} 训练完成'
                }
                if model_name == 'K-均值聚类':
                    result_metrics['验证轮廓系数'] = val_acc
                    if hasattr(model_instance, 'cluster_centers_'):
                        result_metrics['聚类中心'] = model_instance.cluster_centers_.tolist()

            elif model_name == 'K-均值聚类':
                if stop_event and stop_event.is_set():
                    print(f"模型 {model_name} 训练在开始拟合前被中止。")
                    return model_instance, {'训练时间': 0, 'message': '训练已中止'}, None

                model_instance.fit(X_train_for_model) # K-Means 无监督，只fit X_train
                emit_process_progress(training_id, 'training',
                                      {'status': 'training_complete', 'message': f'{model_name} 训练完成。',
                                       'progress_percent': 90})

                train_silhouette = None
                labels_pred_train = model_instance.labels_
                if X_train_for_model.shape[0] > 1 and model_instance.n_clusters > 1 and len(np.unique(labels_pred_train)) > 1:
                    train_silhouette = silhouette_score(X_train_for_model, labels_pred_train)

                val_silhouette = None
                if X_val_for_model.size > 0:
                    labels_pred_val = model_instance.predict(X_val_for_model)
                    if X_val_for_model.shape[0] > 1 and model_instance.n_clusters > 1 and len(np.unique(labels_pred_val)) > 1:
                        val_silhouette = silhouette_score(X_val_for_model, labels_pred_val)

                result_metrics = {
                    '训练时间': time.time() - start_time,
                    '训练轮廓系数': train_silhouette,
                    '验证轮廓系数': val_silhouette,
                    'message': f'{model_name} 训练完成'
                }
                if hasattr(model_instance, 'cluster_centers_'):
                    result_metrics['聚类中心'] = model_instance.cluster_centers_.tolist()

        except Exception as e:
            error_msg = f"模型 {model_name} 训练失败: {str(e)}, {traceback.format_exc()}"
            emit_process_error(training_id, 'training', error_msg)
            return None, {'训练时间': 0, 'message': error_msg}, None

        # --- Scikit-learn 模型训练后，进行最终测试集评估 ---
        print(f"开始对 {model_name} 模型进行最终测试集评估...")
        final_test_results = evaluate_sklearn_model_on_test_set(
            model_instance, X_test_for_model, y_test, model_name
        )
        print(f"{model_name} 最终测试集结果: {final_test_results}")

        if model_instance and '失败' not in result_metrics.get('message', ''):
            try:
                joblib.dump(model_instance, f"{model_path}.joblib")
                print(f"模型 {model_name} 已保存到: {model_path}.joblib")
            except Exception as e:
                print(f"保存模型 {model_name} 失败: {e}")

    # 将最终测试结果添加到总结果中
    result_metrics['最终测试结果'] = final_test_results

    print(f"模型 {model_name} 训练完成。结果: {result_metrics}")
    emit_process_completed(training_id,'training', {
        'message': f'{model_name} 训练评估完成',
        'results': result_metrics
    })
    return model_instance, result_metrics, model_path