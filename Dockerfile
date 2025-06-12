# ✅ 使用官方 Miniconda 镜像（体积比 Anaconda 小，干净）
FROM continuumio/miniconda3

# ✅ 设置工作目录
WORKDIR /app

# ✅ 拷贝环境配置文件和项目代码
COPY environment.yaml .
COPY . /app

# ✅ 创建 Conda 环境（名称来自 environment.yaml）
RUN conda env create -f environment.yaml

# ✅ 确保 Conda 环境的路径被注册（否则某些依赖可能找不到）
ENV PATH /opt/conda/envs/pytorch/bin:$PATH

# ✅ 设置默认 shell（后续命令都在 pytorch 环境中执行）
SHELL ["conda", "run", "--no-capture-output", "-n", "pytorch", "/bin/bash", "-c"]

# ✅ 开放端口（假设 Flask 项目用的是 5000）
EXPOSE 5000

# ✅ 启动服务（确保 app.py 中有 host='0.0.0.0'）
CMD ["python", "app.py"]

