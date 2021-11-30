import os
import json

program_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(program_path,"../../Datasets/TVM/") 
prefix_dataset_fold_path = os.path.join(program_path,"../../Datasets/") 

device_json_path = os.path.join(program_path,"../../Datasets/devices.json") 

devices = None
with open(device_json_path ,'r') as f:
    devices = json.load(f)