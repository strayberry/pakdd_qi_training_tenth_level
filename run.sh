#bin/bash 
#打印GPU信息 
nvidia-smi 
#训练model
python3 ./model/basic_model.py
#运行server
python3 ./code/api.py