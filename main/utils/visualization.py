import numpy as np
from scipy import fft


class Visualizer:
    @staticmethod
    def generate_waveform(data):
        """
        生成波形图数据

        参数:
        data: 字典(包含数值字段)或数组

        返回:
        波形图数据
        """
        try:
            # 如果是字典，提取数值字段
            if isinstance(data, dict):
                numeric_values = []
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not isinstance(v, dict):
                        numeric_values.append(v)
                data = numeric_values

            # 转换为numpy数组
            data_array = np.array(data, dtype=float)

            if len(data_array.shape) == 1:
                # 一维数据
                return {
                    "x": list(range(len(data_array))),
                    "y": data_array.tolist()
                }
            else:
                # 二维数据 - 每列一个通道
                result = []
                for i in range(data_array.shape[1]):
                    channel_data = data_array[:, i]
                    result.append({
                        "x": list(range(len(channel_data))),
                        "y": channel_data.tolist()
                    })
                return result
        except Exception as e:
            print(f"生成波形图数据时出错: {str(e)}")
            # 返回一个空的波形图以避免完全失败
            return {"x": [], "y": []}

    @staticmethod
    def generate_spectrum(data):
        """
        生成频谱图数据

        参数:
        data: 字典(包含数值字段)或数组

        返回:
        频谱图数据
        """
        try:
            # 如果是字典，提取数值字段
            if isinstance(data, dict):
                numeric_values = []
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not isinstance(v, dict):
                        numeric_values.append(v)
                data = numeric_values

            # 如果没有足够的数值，返回空结果
            if len(data) < 2:
                return {"freq": [], "amplitude": []}

            # 转换为numpy数组
            data_array = np.array(data, dtype=float)

            if len(data_array.shape) == 1:
                # 一维数据
                n = len(data_array)
                yf = fft.fft(data_array)
                xf = np.linspace(0, 1.0 / (2.0), n // 2)
                return {
                    "freq": xf.tolist(),
                    "amplitude": np.abs(yf[:n // 2]).tolist()
                }
            else:
                # 二维数据 - 每列一个通道
                result = []
                for i in range(data_array.shape[1]):
                    channel_data = data_array[:, i]
                    n = len(channel_data)
                    yf = fft.fft(channel_data)
                    xf = np.linspace(0, 1.0 / (2.0), n // 2)
                    result.append({
                        "freq": xf.tolist(),
                        "amplitude": np.abs(yf[:n // 2]).tolist()
                    })
                return result
        except Exception as e:
            print(f"生成频谱图数据时出错: {str(e)}")
            # 返回一个空的频谱图以避免完全失败
            return {"freq": [], "amplitude": []}