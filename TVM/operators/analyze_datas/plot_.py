import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import ast
import analyzes.TVM.config as config

def mycolor(index)->str:
    colors_ = ['red','green','blue','c','m','y','k']

    if index<len(colors_) and index>=0:
        return colors_[index]
    else:
        return 'w'

def calc_datas_count(dimens: tuple, shapes: tuple, batch_size: int):
    shape = list(shapes[dimens[1][0]])
    shape[dimens[1][1]]=batch_size

    datas_count = 1
    for i in shape:
        datas_count*=i
    return datas_count

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
op_names = ["add","multiply","nn.bias_add","nn.relu","sigmoid","subtract","tanh"]
bench_op = "add"


for device_name in datas.keys():
    if device_name=="count":
        continue

    # 测试基准op
    bench_datas_runtime = [[],[]]

    for shapes_dimensionality in datas[device_name][bench_op]["0"].keys():
        if shapes_dimensionality=="count":
            continue
        
        # 创建存储目录
        fold_path = os.path.join(os.path.join(program_path,"images"),"total").replace(" ","").replace("(","[").replace(")","]")
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        for shape in datas[device_name][bench_op]["0"][shapes_dimensionality].keys():
            if shape=="count":
                continue
            
            value = datas[device_name][bench_op]["0"][shapes_dimensionality][shape]
            with open(os.path.join(config.prefix_dataset_fold_path, value["file_path"])) as f:
                line = f.readline()
                while line is not None and len(line)>0 :
                    bench_datas_runtime[0].append(calc_datas_count(ast.literal_eval(shapes_dimensionality),ast.literal_eval(shape),int(line.split(",")[0])))
                    bench_datas_runtime[1].append(float(line.split(",")[1]))
                    line=f.readline()

    np_data_size = np.array(bench_datas_runtime[0])
    np_runtime = np.array(bench_datas_runtime[1])
    index_sort = np.array(np_data_size).argsort()
    bench_datas_runtime = [np_data_size[index_sort],np_runtime[index_sort]]
    # 基准op测试完毕
    
    for device_id in datas[device_name][bench_op].keys():
        if device_id=="count":
            continue
        hardware_name = device[device_name][device_id]

        plt.figure(0)
        op_index = 0
        for op_name in op_names:
            datas_runtime = [[],[]]
            for shapes_dimensionality in datas[device_name][op_name][device_id].keys():
                if shapes_dimensionality=="count":
                    continue
                
                # 创建存储目录
                fold_path = os.path.join(os.path.join(program_path,"images"),"total").replace(" ","").replace("(","[").replace(")","]")
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                for shape in datas[device_name][op_name][device_id][shapes_dimensionality].keys():
                    if shape=="count":
                        continue
                    
                    value = datas[device_name][op_name][device_id][shapes_dimensionality][shape]
                    with open(os.path.join(config.prefix_dataset_fold_path, value["file_path"])) as f:
                        line = f.readline()
                        while line is not None and len(line)>0 :
                            datas_runtime[0].append(calc_datas_count(ast.literal_eval(shapes_dimensionality),ast.literal_eval(shape),int(line.split(",")[0])))
                            datas_runtime[1].append(float(line.split(",")[1]))
                            line=f.readline()

            np_data_size = np.array(datas_runtime[0])
            np_runtime = np.array(datas_runtime[1])
            index_sort = np.array(np_data_size).argsort()

            plt.figure(0)
            plt.plot(np_data_size[index_sort],np_runtime[index_sort],color=mycolor(op_index),label=op_name)
            plt.figure(1)
            plt.plot(np_data_size[index_sort],np_runtime[index_sort]/bench_datas_runtime[1],color=mycolor(op_index),label=op_name)
            op_index+=1
        
        plt.figure(0)
        plt.legend() # 显示图例
        plt.xlabel('data size')
        plt.ylabel('run-time')
        # plt.set_title(shape)
        plt.savefig(os.path.join(fold_path,hardware_name+".png"))
        plt.close()

        plt.figure(1)
        plt.legend() # 显示图例
        plt.xlabel('data size')
        plt.ylabel('run-time/add')
        # plt.set_title(shape)
        plt.savefig(os.path.join(fold_path,hardware_name+"-bench_add.png"))
        plt.close()