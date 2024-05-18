import re
import numpy as np
# 用于存储提取的数字的数组
accuracies = []

# 打开包含输出的文件
with open('***cifar10_beta0.05_fedexe_ned_klloss0.1_cal_logist0.005_global_local_perclass_acc_1round.out', 'r') as file:
    for line in file:
        # 使用正则表达式来匹配行中的数字部分
        match = re.search(r'Client Model Test Accuracy on class 9 : (\d+\.\d+)', line)
        if match:
            accuracy = float(match.group(1))  # 提取匹配的数字并转换为浮点数
            accuracies.append(accuracy)  # 将提取的数字添加到数组中

# 打印提取的数字数组
wise_acc = np.mean(accuracies)
print(accuracies)
print(wise_acc)

