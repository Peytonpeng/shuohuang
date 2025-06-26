
### 部署说明
#### 准备（本地操作）
pip3 freeze > requirements.txt
pip3 download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 312 -d ./offline_packages -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

#### python镜像离线安装（服务器操作）

docker load < python.3.12.tar

#### 制作analysis:v1 镜像（服务器操作）

>上传代码至文件夹
> 
>docker build -t analysis:v1
> 
>docker-compose up -d
