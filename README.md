# PAKDD2021 2nd ALIBABA CLOUD AIOPS COMPETITION
团队名称：练气十级

## 解决方案及算法
1. 数据处理: 参照 baseline , 聚合时间粒度改为 7 min
2. 模型训练: XgBoost 算法 （./model/basic_model.py）
3. 结果预测: inner join 两张测试数据, kernel_log 和 address_log, 读取模型进行预测 （./code/main.py）

## 代码运行说明
运行文件[main.py](./code/main.py)
```
# 安装环境后, 执行 main.py 预测
$ pip install -r ./code/requirements.txt
$ python ./code/main.py
```
