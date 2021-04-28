FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
MAINTAINER Berry Yan <strayberry0w0@gmail.com>

## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . / 
WORKDIR / 

## 安装python依赖包 
RUN set -ex && pip3 install -r ./code/requirements.txt

## 挂载点
VOLUME ["/tcdata"]

## 镜像启动后统一执行 sh run.sh 
CMD ["sh", "run.sh"]