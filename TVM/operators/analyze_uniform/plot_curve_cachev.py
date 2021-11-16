from numpy.lib.function_base import kaiser
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import json
import os
import math
import analyzes.TVM.config as config

# 自定义函数，curve_fit支持自定义函数的形式进行拟合，这里定义的是指数函数的形式
def func(x, k1,k2,m1,m2,m3,x0,x1):
    return np.piecewise(x, [x <= x0, np.logical_and(x0<x, x<= x1),x>x1],[lambda x:m1*x, lambda x:k1*m1*x + (1-k1)*m2*x, lambda x:k1*m1*x + (1-k1)*(k2*m2*x+(1-k2)*m3*x)])

def curve_datas(datas):
    xdata = np.array(datas[0])
    ydata = np.array(datas[1],dtype=np.float64)*10000000
    popt, pcov = curve_fit(func, xdata,ydata,maxfev=400000) 

    return [list(np.array(datas[0])), list(func(np.array(datas[0]), *popt)/10000000)]

def calc_k(datas, offset=15):
    ks=[[],[]]
    for i in range(len(datas[1])-offset):
        ks[0].append(datas[0][i])
        ks[1].append(abs((datas[1][i+offset]-datas[1][i])/(datas[0][i+offset]-datas[0][i])))

        # if i==0:
        #     ks[1].append(datas[1][i]/datas[0][i])
        # else:
        #     ks[1].append(abs((datas[1][i]-datas[1][i-1])/(datas[0][i]-datas[0][i-1])))

    return ks

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题

program_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(config.dataset_path,"datasets/dataset.json") 
datas = {}
picture_index = 1

with open(json_path,'r') as f:
    datas = json.load(f)

device = {"dell04":{"0":"GTX-2080Ti"},"dell03":{"0":"Tesla-T4"},"dell01":{"0":"Tesla-T4"},"dellh01":{"0":"Tesla-K40c"}}

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

            for shapes_dimensionality in datas[device_name][op_name][device_id].keys():
                if shapes_dimensionality=="count":
                    continue
                
                # 创建存储目录
                fold_path = os.path.join(os.path.join(os.path.join(program_path,"images"),op_name),shapes_dimensionality).replace(" ","").replace("(","[").replace(")","]")
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                # 开始绘图
                plt.figure(picture_index)
                plt.figure(figsize=(50,50))
                picture_index += 1
                picture_count = int(datas[device_name][op_name][device_id][shapes_dimensionality]["count"])

                img_index=0
                for shape in datas[device_name][op_name][device_id][shapes_dimensionality].keys():
                    if shape=="count":
                        continue

                    img_index+=1
                    value = datas[device_name][op_name][device_id][shapes_dimensionality][shape]
                    data=[[],[]]
                    with open(os.path.join(config.prefix_dataset_fold_path, value["file_path"])) as f:
                        line = f.readline()
                        while line is not None and len(line)>0 :
                            data[0].append(int(line.split(",")[0]))
                            data[1].append(float(line.split(",")[1]))
                            line=f.readline()
                    img = plt.subplot(math.ceil(picture_count/10),10 , img_index)
                    
                    # data[0].insert(0,0)
                    # data[1].insert(0,0)
                    # data = calc_k(data)
                    data_curve = curve_datas(data)

                    plt.plot(*data,color="gray")
                    plt.plot(*data_curve,color="green")
                    # plt.legend() # 显示图例
                    plt.xlabel('batch size')
                    plt.ylabel('run-time')
                    img.set_title(shape)
                plt.savefig(os.path.join(fold_path,hardware_name+"--curve_cachev.png"))