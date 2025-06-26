# 使用官方提供的 Python 开发镜像作为基础镜像
FROM python:3.12

# 将工作目录切换为 /app
WORKDIR /app

# 将当前目录下的所有内容复制到 /app 下

ADD . /app

# 使用 pip 命令安装这个应用所需要的依赖
# RUN pip install --trusted-host pypi.python.org -r requirements.txt
# RUN pip3 install --trusted-host https://pypi.tuna.tsinghua.edu.cn/simple -r ./requirements.txt
RUN pip install --no-index --find-links=./offline_packages -r ./requirements.txt
#RUN pip install -r ./requirements.txt
#RUN pip install --upgrade pip
#RUN pip install Flask
# 国内的源更快

# 允许外界访问容器的 12345 端口
EXPOSE 5001

# 设置环境变量
ENV NAME World

# 设置容器进程为：python app.py，即：这个 Python 应用的启动命令
CMD ["python3", "app.py"]
# CMD 前面 隐式的包含了 ENTRYPOINT ， /bin/sh -c
