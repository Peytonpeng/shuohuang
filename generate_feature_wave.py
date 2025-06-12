import base64
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_extraction(features, method_name, feature_names=None):
    """
    通用绘图函数，根据特征提取方法自动选择合适的可视化方式，并返回 Base64 编码的图像。

    参数：
    features: np.ndarray
        特征提取后的数据。
    method_name: str
        进行的特征提取方法名称。
    feature_names: list, optional
        特征名称列表，默认使用索引。

    返回：
    - Base64 编码的图像字符串
    """
    features = np.array(features)
    shape = features.shape

    plt.figure(figsize=(10, 5))
    plt.title(f'{method_name} Visualization')

    # 针对不同特征类型选择合适的绘图方式
    if len(shape) == 1 or shape[0] == 1:  # 1D 数据，如 Kurtosis
        plt.bar(range(len(features.flatten())), features.flatten(), color='skyblue')
        plt.xlabel('Feature Index')
        plt.ylabel(method_name)

    elif shape[0] > 1 and shape[1] < 50:  # 2D 小规模数据，如傅里叶变换
        for i in range(shape[1]):
            plt.plot(features[:, i], label=f'Feature {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Magnitude')
        plt.legend()

    else:  # 2D 大规模数据，如时频分析
        sns.heatmap(features, cmap='viridis', cbar=True)
        plt.xlabel('Time / Frequency')
        plt.ylabel('Feature / Decomposition Level')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64


# --- 生成波形图 ---
def generate_waveform(timestamp_data, data, title):
    """
    生成波形图，并返回 base64 编码的图像。

    参数：
    - timestamp_data: 时间戳数据（Series）
    - data: 需要绘制的数据（DataFrame）
    - title: 图表标题

    返回：
    - Base64 编码的图像字符串
    """
    plt.figure(figsize=(12, 6))
    for col in data.columns:
        plt.plot(timestamp_data, data[col], label=col)
    plt.title(title)
    plt.xlabel('时间戳')
    plt.ylabel('数值')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    waveform_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return waveform_base64


# --- 生成二维散点图 ---
def generate_2d_scatter_plot(data, title):
    """
    生成二维散点图，并返回 base64 编码的图像。

    参数：
    - data: 需要绘制的数据（DataFrame），假设只有两列
    - title: 图表标题

    返回：
    - Base64 编码的图像字符串
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.title(title)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    scatter_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return scatter_base64


# --- 生成三维散点图 ---
def generate_3d_scatter_plot(data, title):
    """
    生成三维散点图，并返回 base64 编码的图像。

    参数：
    - data: 需要绘制的数据（DataFrame），假设只有三列
    - title: 图表标题

    返回：
    - Base64 编码的图像字符串
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    ax.set_title(title)
    ax.set_xlabel('主成分1')
    ax.set_ylabel('主成分2')
    ax.set_zlabel('主成分3')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    scatter_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return scatter_base64