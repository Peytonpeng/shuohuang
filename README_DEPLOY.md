
### 部署说明
#### 准备,在有网的机器上运行
```
1.制作镜像包
   pip3 freeze > requirements.txt
   docker build -t sh_analysis:v1 .
   docker save -o sh_analysis.v1.tar sh_analysis:v1
   gzip sh_analysis.v1.tar
   sh_analysis.v1.tar.gz 到 离线服务器
2. sh_analysis.v1.tar.gz
gzip -d sh_analysis.v1.tar.gz
docker load < sh_analysis.v1.tar
docker compose up -d

pip3 download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 312 -d ./offline_packages -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip3 download -d ./offline_packages -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

```
