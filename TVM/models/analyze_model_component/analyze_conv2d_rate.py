import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import ast
import analyzes.TVM.config as config

program_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(config.dataset_path,"models_component/dataset_time_analyze.json") 
datas = {}

with open(json_path,'r') as f:
    datas = json.load(f)

plot_data = [[],[],[],[]]

for model_name in datas.keys():
    if model_name in ["accuracy","loss","allow_loss_rate"]:
        continue
    model_total_accuracy = 0.0
    for model_input_shape in datas[model_name].keys():
        if model_input_shape in ["accuracy","loss"]:
            continue
        model_input_total_accuracy = 0.0
        for batch_size in datas[model_name][model_input_shape].keys():
            if batch_size in ["accuracy","loss"] or "real_runtimes" not in datas[model_name][model_input_shape][batch_size].keys() or "nn.conv2d" not in datas[model_name][model_input_shape][batch_size].keys() or "runtimes" not in datas[model_name][model_input_shape][batch_size]["nn.conv2d"].keys():
                continue
            
            conv2d_runtime = float(datas[model_name][model_input_shape][batch_size]["nn.conv2d"]["runtimes"])
            real_runtime = float(datas[model_name][model_input_shape][batch_size]["real_runtimes"])
            pre_runtime = float(datas[model_name][model_input_shape][batch_size]["runtimes"])
            plot_data[0].append(conv2d_runtime)
            plot_data[1].append(real_runtime)
            plot_data[2].append(pre_runtime)

 # 创建存储目录
fold_path = os.path.join(program_path,"images")
if not os.path.exists(fold_path):
    os.makedirs(fold_path)

np_conv2d_runtime = np.array(plot_data[0])
np_real_runtime = np.array(plot_data[1])
np_pre_runtime = np.array(plot_data[2])
index_sort = np_conv2d_runtime.argsort()

plt.plot(np_real_runtime[index_sort],color = 'green',label="real")
plt.plot(np_pre_runtime[index_sort],color = 'red',label="pre")
plt.legend()
plt.ylabel('runtime')
plt.savefig(os.path.join(fold_path,"pre-real.png"))
plt.close()

print((np_conv2d_runtime[index_sort]/np_real_runtime[index_sort])[:10])
plt.plot(np_conv2d_runtime[index_sort]/np_real_runtime[index_sort])
plt.ylabel('conv2d-runtimes/real-runtime')
plt.savefig(os.path.join(fold_path,"conv2d_ratio.png"))
plt.close()

print((np_pre_runtime[index_sort]/np_real_runtime[index_sort])[:10])
plt.plot(np_pre_runtime[index_sort]/np_real_runtime[index_sort])
plt.ylabel('pre-runtime / real-runtime')
plt.savefig(os.path.join(fold_path,"pre_ratio.png"))
plt.close()