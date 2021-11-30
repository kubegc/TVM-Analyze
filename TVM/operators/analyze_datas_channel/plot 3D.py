import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import ast
import analyzes.TVM.config as config

def calc_datas_count(dimens: tuple, shapes: tuple):
    shape = list(shapes[dimens[1][0]])
    shape[dimens[1][1]]=1

    datas_count = 1
    for i in shape:
        datas_count*=i
    return datas_count

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题

program_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(config.dataset_path,"datasets_channel/dataset.json") 
datas = {}
picture_index = 1

with open(json_path,'r') as f:
    datas = json.load(f)

device = config.devices

for device_name in datas.keys():
    if device_name=="count":
        continue
    
    for op_name in datas[device_name].keys():
        if op_name=="count":
            continue

        for device_id in datas[device_name][op_name].keys():
            if device_id=="count":
                continue
            hardware_name = device[device_name][device_id]

            datas_runtime = [[],[],[]]      # size,batch_size,runtime
            # plt.figure(picture_index)
            # picture_index += 1
            for shapes_dimensionality in datas[device_name][op_name][device_id].keys():
                if shapes_dimensionality=="count":
                    continue
                
                # 创建存储目录
                fold_path = os.path.join(os.path.join(os.path.join(os.path.join(program_path,"images"),op_name),shapes_dimensionality),"3D").replace(" ","").replace("(","[").replace(")","]")
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                # 开始绘图
                
                #plt.figure(figsize=(19,10))
                
                # picture_count = int(datas[device_name][op_name][device_id][shapes_dimensionality]["count"])

                # img_index=0
                for shape in datas[device_name][op_name][device_id][shapes_dimensionality].keys():
                    if shape=="count":
                        continue
                    
                    value = datas[device_name][op_name][device_id][shapes_dimensionality][shape]
                    with open(os.path.join(config.prefix_dataset_fold_path, value["file_path"])) as f:
                        line = f.readline()
                        while line is not None and len(line)>0 :
                            datas_runtime[0].append(calc_datas_count(ast.literal_eval(shapes_dimensionality),ast.literal_eval(shape)))
                            datas_runtime[1].append(int(line.split(",")[0]))
                            datas_runtime[2].append(float(line.split(",")[1]))
                            line=f.readline()

            np_data_size = np.array(datas_runtime[0])
            np_batch_size = np.array(datas_runtime[1])
            np_runtime = np.array(datas_runtime[2])
            index_sort = np.array(np_data_size).argsort()

            # 绘制3D图像
            fig = plt.figure()
            ax = plt.axes(projection= '3d')
            ax.plot_trisurf(np_data_size[index_sort],np_batch_size[index_sort],np_runtime[index_sort],cmap='rainbow')
            # plt.legend() # 显示图例
            ax.set_xlabel('data size')
            ax.set_ylabel('batch size')
            ax.set_zlabel('run-time')
            plt.savefig(os.path.join(fold_path,hardware_name+"-3D.png"))
            plt.close()