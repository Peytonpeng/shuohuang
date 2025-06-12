import numpy as np
from scipy.stats import kurtosis
from scipy.fftpack import fft
import pywt
import emd

#峭度指标
import numpy as np
from scipy.stats import kurtosis

def histogram_feature(signal, bins=10, window_size=512, step_size=None):
    """
    对一维信号计算直方图特征。如果提供了 window_size，则进行小窗口化处理。

    参数：
    signal: np.ndarray，一维数组
        输入信号。
    bins: int，直方图的区间数
        直方图的柱子数量。
    window_size: int, optional
        滑动窗口的大小。如果为 None，则对整个信号计算直方图。
    step_size: int, optional
        滑动窗口的步长。仅在 window_size 不为 None 时有效。如果为 None，则默认为 window_size。

    返回：
    np.ndarray
        如果未进行窗口化，返回一维直方图特征向量（长度为 bins）。
        如果进行窗口化，返回二维数组，每行是一个窗口的直方图特征向量。
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维数组。")

    if window_size is None:
        # 不进行窗口化，直接计算整个信号的直方图
        hist, _ = np.histogram(signal, bins=bins, density=True)
        return hist
    else:
        # 进行窗口化处理
        if step_size is None:
            step_size = window_size # 如果未指定步长，默认为不重叠窗口

        features = []
        for i in range(0, len(signal) - window_size + 1, step_size):
            window = signal[i:i + window_size]
            hist, _ = np.histogram(window, bins=bins, density=True)
            features.append(hist)
        return np.array(features)

def kurtosis_index(data, window_size=512, step_size=None):
    """
    计算输入数据（或其每个小窗口）的峰度（Kurtosis Index）。

    参数：
    data: np.ndarray, 形状 (n_samples,) 或 (n_samples, n_features)
        输入的数据矩阵。如果是单个信号，建议传入一维数组。
    window_size: int, optional
        滑动窗口的大小。如果为 None，则对整个信号计算峰度。
    step_size: int, optional
        滑动窗口的步长。仅在 window_size 不为 None 时有效。如果为 None，则默认为 window_size。

    返回：
    np.ndarray
        如果未进行窗口化，返回形状为 (1, 1) 或 (1, n_features) 的峰度值二维数组。
        如果进行窗口化，返回二维数组，每行是一个窗口的峰度值。
    """
    data = np.asarray(data)

    if window_size is None:
        # 不进行窗口化，直接计算整个信号或多特征数据的峰度
        # 确保对于单个信号返回 (1,1) 的形状
        if data.ndim == 1:
            kurtosis_value = kurtosis(data, axis=0)
            return np.array([[kurtosis_value]])
        elif data.ndim == 2:
            kurtosis_values = kurtosis(data, axis=0)
            return kurtosis_values.reshape(1, -1)
        else:
            raise ValueError("输入数据维度不正确。")
    else:
        # 进行窗口化处理
        if data.ndim != 1:
            raise ValueError("进行窗口化时，输入数据必须是一维数组。")

        if step_size is None:
            step_size = window_size # 如果未指定步长，默认为不重叠窗口

        features = []
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i + window_size]
            kurtosis_value = kurtosis(window, axis=0)
            features.append([kurtosis_value]) # 使用列表包裹，以便转换为二维数组
        return np.squeeze(np.array(features)).reshape(-1, 1)



#傅里叶变换
def fourier_transform(data):
    """
    计算输入数据每个特征的傅里叶变换。

    参数：
    data: np.ndarray, 形状 (n_samples, n_features)
        输入的数据矩阵。

    返回：
    fourier_features: np.ndarray, 形状 (n_samples, n_features)
        每个特征的傅里叶变换后的幅度谱。
    """
    return np.abs(fft(data, axis=0))

#小波变换
def wavelet_transform(signal, wavelet_name='db4', level=None):
    """
    对输入信号进行小波变换，返回二维数组。

    参数:
    signal (array-like): 输入的一维信号。
    wavelet_name (str): 小波基的名称，默认为 'db4'（Daubechies 4 小波）。
    level (int, 可选): 分解的层数。如果为 None，则自动确定最大层数。

    返回:
    np.ndarray: 包含近似系数和细节系数的二维数组。
    """
    # 检查信号是否为一维
    if len(signal.shape) != 1:
        raise ValueError("输入信号必须是一维的。")

    # 如果没有指定分解层数，则自动确定最大层数
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name))

    # 进行小波分解
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)

    # 找到最长系数的长度
    max_len = max(len(c) for c in coeffs)

    # 填充系数到相同长度
    padded_coeffs = [np.pad(c, (0, max_len - len(c)), mode='constant') for c in coeffs]

    # 将系数堆叠为二维数组并转置
    result = np.array(padded_coeffs).T  # 转置操作，从(分量, 时间点)变为(时间点, 分量)

    return result

#经验模态分解
def empirical_mode_decomposition(signal, max_imfs=None, sift_thresh=1e-8):
    """
    对输入信号进行经验模态分解。

    参数:
    signal (array-like): 输入的一维信号。
    max_imfs (int, 可选): 要提取的最大本征模态函数（IMF）数量。如果为 None，则提取尽可能多的 IMF。
    sift_thresh (float, 可选): 筛选过程的终止阈值，默认为 1e-8。


    返回:
    array: 包含所有本征模态函数的二维数组，每一行代表一个 IMF。
    """
    # 进行经验模态分解
    imfs = emd.sift.sift(signal, max_imfs=max_imfs, sift_thresh=sift_thresh)
    return imfs

#Wigner-Ville分布
def wigner_ville_distribution(signal):
    """
    计算输入信号的 Wigner - Ville 分布。

    参数:
    signal (array-like): 输入的一维信号。

    返回:
    array: 二维数组，表示 Wigner - Ville 分布。
    """
    N = len(signal)
    t = np.arange(N)
    wvd = np.zeros((N, N), dtype=complex)

    for tau in range(-N // 2, N // 2):
        for n in range(N):
            if 0 <= n + tau // 2 < N and 0 <= n - tau // 2 < N:
                wvd[n, tau + N // 2] = signal[n + tau // 2] * np.conj(signal[n - tau // 2])

    wvd = np.fft.fft(wvd, axis=1)
    wvd = np.fft.fftshift(wvd, axes=1)

    return np.real(wvd)

#Fisher判别法
def fisher_discriminant(data, labels):
    """
    使用Fisher判别法进行特征选择。

    参数:
    data -- 样本数据，形状为 (样本数, 特征数) 的二维numpy数组
    labels -- 样本标签，一维numpy数组

    返回:
    W -- Fisher判别向量，形状为 (特征数, 1)
    """
    unique_labels = np.unique(labels)
    num_features = data.shape[1]

    # 计算每个类的均值向量
    mean_vectors = []
    for label in unique_labels:
        mean_vectors.append(np.mean(data[labels == label], axis=0))

    # 类间散度矩阵 Sb
    overall_mean = np.mean(data, axis=0)
    Sb = np.zeros((num_features, num_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = data[labels == unique_labels[i], :].shape[0]
        mean_vec = mean_vec.reshape(num_features, 1)  # make column vector
        overall_mean = overall_mean.reshape(num_features, 1)  # make column vector
        Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # 类内散度矩阵 Sw
    Sw = np.zeros((num_features, num_features))
    for label, mv in zip(unique_labels, mean_vectors):
        class_sc_mat = np.zeros((num_features, num_features))  # scatter matrix for every class
        for row in data[labels == label]:
            row, mv = row.reshape(num_features, 1), mv.reshape(num_features, 1)  # make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        Sw += class_sc_mat

    # 求解Sw^(-1)Sb的特征值和特征向量
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    # 将特征值和对应的特征向量排序
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

    # 最佳投影方向是具有最大特征值的特征向量
    W = sorted_eigen_vectors[:, 0].reshape(num_features, 1)

    return W

#Relief算法
def relief_feature_selection(X, y, num_neighbors=10):
    """
    使用Relief算法进行特征选择。

    参数:
    X -- 样本数据，形状为 (样本数, 特征数) 的二维numpy数组
    y -- 样本标签，一维numpy数组（0 或 1）
    num_neighbors -- 邻居数量，默认值为10

    返回:
    feature_weights -- 每个特征的权重，形状为 (特征数,) 的一维numpy数组
    """
    num_samples, num_features = X.shape
    feature_weights = np.zeros(num_features)

    # 对于每个样本寻找其同类和异类的最近邻居
    for i in range(num_samples):
        distances = np.sum((X - X[i]) ** 2, axis=1)
        # 排除自身点
        distances[i] = np.inf

        # 找到num_neighbors个最近的同类和异类邻居
        same_class_distances = np.where(y == y[i], distances, np.inf)
        diff_class_distances = np.where(y != y[i], distances, np.inf)

        same_class_neighbors = np.argsort(same_class_distances)[:num_neighbors]
        diff_class_neighbors = np.argsort(diff_class_distances)[:num_neighbors]

        # 更新特征权重
        for j in range(num_features):
            feature_diff_same = np.sum(np.abs(X[same_class_neighbors, j] - X[i, j]))
            feature_diff_diff = np.sum(np.abs(X[diff_class_neighbors, j] - X[i, j]))

            feature_weights[j] -= feature_diff_same / num_neighbors
            feature_weights[j] += feature_diff_diff / num_neighbors

    return feature_weights