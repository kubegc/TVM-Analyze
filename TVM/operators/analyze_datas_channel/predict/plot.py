from operator import truediv
from numpy.lib.function_base import average, kaiser
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import json
import os
import math
import analyzes.TVM.config as config
import ast

# 自定义函数，curve_fit支持自定义函数的形式进行拟合，这里定义的是指数函数的形式
def func(x,a0, a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c2,c3,c4,c5,d0,d1,d2,d3,d4,d5):
    # return (k0*x[0]+b0)*(k1*x[1]+b1)*(k2*x[2]+b2)*(k3*x[3]+b3)
    return (a5*x[0]**5+a4*x[0]**4+a3*x[0]**3+a2*x[0]**2+a1*x[0]+a0)*(b5*x[1]**5+b4*x[1]**4+b3*x[1]**3+b2*x[1]**2+b1*x[1]+b0)*(c5*x[2]**5+c4*x[2]**4+c3*x[2]**3+c2*x[2]**2+c1*x[2]+c0)*(d5*x[3]**5+d4*x[3]**4+d3*x[3]**3+d2*x[3]**2+d1*x[3]+d0)
    # return k2*x**3+k1*x**2+k0*x+b0

def curve_datas(datas):
    # xdata = np.array(datas[0])
    # ydata = np.array(datas[1],dtype=np.float64)*10000000
    popt, pcov = curve_fit(func, (np.array(datas[0]),np.array(datas[1]),np.array(datas[2]),np.array(datas[3])),np.array(datas[4],dtype=np.float64)*10000000) 

    return list(func((np.array(datas[0]),np.array(datas[1]),np.array(datas[2]),np.array(datas[3])), *popt)/10000000)

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
json_path = os.path.join(config.dataset_path,"datasets_channel/dataset.json") 
datas = {}
ops=["add"]

with open(json_path,'r') as f:
    datas = json.load(f)

device = config.devices

data = [[],[],[],[],[]]
for device_name in datas.keys():
    if device_name=="count":
        continue
    
    for op_name in (datas[device_name].keys() if len(ops)>0 else ops):
        if op_name=="count":
            continue
        
        for device_id in datas[device_name][op_name].keys():
            if device_id=="count":
                continue
            hardware_name = device[device_name][device_id]

            for shapes_dimensionality in datas[device_name][op_name][device_id].keys():
                if shapes_dimensionality=="count":
                    continue

                for shape in datas[device_name][op_name][device_id][shapes_dimensionality].keys():
                    if shape=="count":
                        continue

                    value = datas[device_name][op_name][device_id][shapes_dimensionality][shape]
                    
                    with open(os.path.join(config.prefix_dataset_fold_path, value["file_path"])) as f:
                        line = f.readline()
                        while line is not None and len(line)>0 :
                            real_shape_str = shape.replace("-1",str(line.split(",")[0]))
                            real_shape = ast.literal_eval(real_shape_str)[0]
                            data[0].append(real_shape[0])
                            data[1].append(real_shape[1])
                            data[2].append(real_shape[2])
                            data[3].append(real_shape[3])
                            data[4].append(float(line.split(",")[1]))
                            line=f.readline()

accu_total = 0.0
accurates = []
loss = []
values = []
for pre,real in zip(curve_datas(data),data[4]):
    values.append(pre)
    loss.append(abs(pre-real))
    accurates.append(abs(pre-real)/real)
    accu_total += abs(pre-real)/real

print("avg accurate = ",accu_total/len(data[4]))

plt.plot(accurates)
# plt.legend() # 显示图例
plt.xlabel('')
plt.ylabel('accurates')
plt.savefig(os.path.join(program_path,"accurate.png"))
plt.close()

plt.plot(loss)
# plt.legend() # 显示图例
plt.xlabel('')
plt.ylabel('loss')
plt.savefig(os.path.join(program_path,"loss.png"))
plt.close()

plt.plot(data[4],color="gray")
plt.plot(values,color="green")
# plt.legend() # 显示图例
plt.xlabel('')
plt.ylabel('runtime')
plt.savefig(os.path.join(program_path,"runtime.png"))
plt.close()