from matplotlib import colors
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import math
import analyzes.config as config

def mycolor(index)->str:
    colors_ = ['red','green','blue','c','m','y','k']

    if index<len(colors_) and index>=0:
        return colors_[index]
    else:
        return 'w'

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

bench_device_name = "dell04"
device_names = list(datas.keys())
device_names.remove(bench_device_name)
device_names.remove("count")
device_names.insert(0,bench_device_name)

for op_name in datas[bench_device_name].keys():
    if op_name=="count":
        continue

    for device_id in datas[bench_device_name][op_name].keys():
        if device_id=="count":
            continue
        hardware_name = device[bench_device_name][device_id]

        for shapes_dimensionality in datas[bench_device_name][op_name][device_id].keys():
            if shapes_dimensionality=="count":
                continue
            
            # 创建存储目录
            fold_path = os.path.join(os.path.join(os.path.join(program_path,"images"),op_name),shapes_dimensionality).replace(" ","").replace("(","[").replace(")","]")
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)

            # 开始绘图
            plt.figure(picture_index)
            picture_index += 1
            plt.figure(figsize=(50,50))

            picture_count = int(datas[bench_device_name][op_name][device_id][shapes_dimensionality]["count"])
            img_index=0

            for shape in datas[bench_device_name][op_name][device_id][shapes_dimensionality].keys():
                if shape=="count":
                    continue
                
                # 读取基准值
                bench_value = datas[bench_device_name][op_name][device_id][shapes_dimensionality][shape]
                bench_data=[[],[]]
                with open(os.path.join(program_path,"../"+bench_value["file_path"])) as f:
                    line = f.readline()
                    while line is not None and len(line)>0 :
                        bench_data[0].append(int(line.split(",")[0]))
                        bench_data[1].append(float(line.split(",")[1]))
                        line=f.readline()

                img_index+=1
                img = plt.subplot(math.ceil(picture_count/10),10 , img_index)
                # 绘制比例图
                plot_flag=0
                for device_name in device_names:
                    if shape not in datas[device_name][op_name][device_id][shapes_dimensionality]:
                        break
                    value = datas[device_name][op_name][device_id][shapes_dimensionality][shape]
                    data=[[],[]]
                    radio_data=[[],[]]
                    with open(os.path.join(program_path,"../"+value["file_path"])) as f:
                        index = 0
                        line = f.readline()
                        while line is not None and len(line)>0 :
                            data[0].append(int(line.split(",")[0]))
                            radio_data[0].append(int(line.split(",")[0]))
                            data[1].append(float(line.split(",")[1]))
                            # 计算比值
                            radio_data[1].append(float(line.split(",")[1])/bench_data[1][index])
                            index+=1
                            line=f.readline()

                    plt.plot(*radio_data,color=mycolor(device_names.index(device_name)),label=device[device_name][str(device_id)])
                    plot_flag+=1
                    plt.legend() # 显示图例
                    plt.xlabel('batch size')
                    plt.ylabel('runtime radio')
                    img.set_title(shape)
                if plot_flag==len(device_names):
                    plt.savefig(os.path.join(fold_path,"bench_"+device[bench_device_name][str(device_id)]+".png"))