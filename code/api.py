import json
import pickle
import sys
import pandas as pd
from pandas.core.frame import DataFrame
from ai_hub import inferServer


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        self.group_min_sn_test = None
        print("init_myInfer")

    def etl(self, test_data, agg_time, type):
        from numpy import nan
        if isinstance(test_data, list):
            data = DataFrame(test_data)
            
            if type == 0:
                data.columns = ['serial_number','manufacturer','vendor','collect_time','mca_id','transaction']
            elif type == 1:
                data.columns = ['collect_time','1_hwerr_f','1_hwerr_e','2_hwerr_c','2_sel','3_hwerr_n','2_hwerr_s','3_hwerr_m','1_hwerr_st','1_hw_mem_c','3_hwerr_p','2_hwerr_ce','3_hwerr_as','1_ke','2_hwerr_p','3_hwerr_kp','1_hwerr_fl','3_hwerr_r','_hwerr_cd','3_sup_mce_note','3_cmci_sub','3_cmci_det','3_hwerr_pi','3_hwerr_o','3_hwerr_mce_l','serial_number','manufacturer','vendor']
            elif type == 2:
                data.columns = ['serial_number','manufacturer','vendor','memory','rankid','bankid','collect_time','row','col']

            data[-1] = pd.to_datetime(data['collect_time']).dt.ceil(agg_time)
            group_data = data.groupby(['serial_number','collect_time'], as_index=False).agg('sum')
            return group_data
        else:
            return None

    #数据前处理
    def pre_process(self, request):
        print("my_pre_process")
        
        #json process
        json_data = request.get_json()
        mce_log_test_data = json_data.get('mce_log')
        kernel_log_test_data = json_data.get('kernel_log')
        address_log_test_data = json_data.get('address_log')
        
        # 设置聚合时间粒度
        AGG_VALUE = 30
        AGG_UNIT = 'S'
        AGG_TIME = str(AGG_VALUE)+AGG_UNIT

        # 测试数据
        mce_log_test = self.etl(mce_log_test_data, AGG_TIME, 0)
        kernel_log_test = self.etl(kernel_log_test_data, AGG_TIME, 1)
        address_log_test = self.etl(address_log_test_data, AGG_TIME, 2)

        group_data_test = pd.merge(address_log_test, kernel_log_test, how='outer',on=['serial_number','collect_time'])

        self.group_min_sn_test = pd.DataFrame(group_data_test[['serial_number','collect_time']])
        group_min_test = group_data_test.drop(['serial_number', 'collect_time', 'memory', 'rankid', 'bankid', 'row', 'col', 'manufacturer_y', 'vendor_y', 'manufacturer_x', 'vendor_x'], axis=1)
        print(group_min_test)
        return group_min_test
    
    #数据后处理
    def post_process(self, ret):
        print("post_process")
        #如果这一分钟内没有数据，或者没有预测为1的数据，请返回[]
        if ret.shape[0] == 0:
            return []
        #请将预测结果以json格式返回，具体格式参照赛题
        self.group_min_sn_test['predict'] = ret
        self.group_min_sn_test.rename(columns={'collect_time': 'pti'}, inplace=True)
        self.group_min_sn_test = self.group_min_sn_test[self.group_min_sn_test['predict'] == 1]
        group_min_sn_res = self.group_min_sn_test.drop('predict', axis=1)
        group_min_sn_res = group_min_sn_res.drop_duplicates(subset='serial_number', keep='first')

        processed_data = group_min_sn_res[['serial_number', 'pti']].to_json(orient='records')
        return processed_data
    
    #模型预测：默认执行self.model(preprocess_data)，一般不用重写
    #如需自定义，可覆盖重写
    def predict(self, group_min_test):
        print("predict")
        ret = self.model.predict(group_min_test)
        return ret

if __name__ == "__main__":
    model = pickle.load(open("./user_data/model_data/model.dat", "rb"))
    #his_data_mce = pd.read_csv('./tcdata/memory_sample_mce_log_round2_a_his.csv')
    my_infer = myInfer(model)
    my_infer.run(debuge=True) #默认为("127.0.0.1", 80)
