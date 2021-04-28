import os
import pickle
import pandas as pd

PARENT_FOLDER = '../data/round1_b_test' # test data dir
kernel_log_test_data_path = 'memory_sample_kernel_log_round1_b1_test.csv'
address_log_test_data_path = 'memory_sample_address_log_round1_b1_test.csv'

def etl(path, agg_time):
    import random
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data['collect_time'] = pd.to_datetime(data['collect_time']).dt.ceil(agg_time)
    group_data = data.groupby(['serial_number','collect_time'],as_index=False).agg('sum')
    return group_data

# 设置聚合时间粒度
AGG_VALUE = 7
AGG_UNIT = 'min'
AGG_TIME = str(AGG_VALUE)+AGG_UNIT


# 测试数据
kernel_log_test = etl(kernel_log_test_data_path, AGG_TIME)
address_log_test = etl(address_log_test_data_path, AGG_TIME)
group_data_test = pd.merge(address_log_test, kernel_log_test, how='inner',on=['serial_number','collect_time'])

group_min_sn_test = pd.DataFrame(group_data_test[['serial_number','collect_time']])
group_min_test = group_data_test.drop(['serial_number', 'collect_time', 'memory', 'rankid', 'bankid', 'row', 'col', 'manufacturer_y', 'vendor_y', 'manufacturer_x', 'vendor_x'], axis=1)


# 模型预测
model = pickle.load(open("../user_data/model_data/model.dat", "rb"))
res = model.predict(group_min_test)
group_min_sn_test['predict']=res

# 保存结果
group_min_sn_test=group_min_sn_test[group_min_sn_test['predict']==1]
group_min_sn_res = group_min_sn_test.drop('predict',axis=1)
group_min_sn_res = group_min_sn_res.drop_duplicates(subset='serial_number', keep='first')
print(group_min_sn_res)
group_min_sn_res.to_csv('../prediction_result/predictions.csv', header=False, index=False)