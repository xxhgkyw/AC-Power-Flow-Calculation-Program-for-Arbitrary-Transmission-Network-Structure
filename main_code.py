# 尝试编写通用的交流潮流计算程序,目前以10机39节点为例
# SB=100MVA,需要将功率换算为p.u.;而V和Z直接按p.u.给出了
# 取31号节点为平衡节点, 其余所有节点都视作PQ节点
import numpy as np
import pandas as pd
import supplement_file

# 输入算例信息
# 1、各节点DG、负荷的信息(pdf中表4),各列依次为：节点编号、DG有功、无功出力、有功、无功负荷
DG_and_load = np.array([
    [1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [3, 0, 0, 322.0, 2.4],
    [4, 0, 0, 500.0, 184.0],
    [5, 0, 0, 0, 0],
    [6, 0, 0, 0, 0],
    [7, 0, 0, 233.8, 84.0],
    [8, 0, 0, 522.0, 176.0],
    [9, 0, 0, 0, 0],
    [10, 0, 0, 0, 0],
    [11, 0, 0, 0, 0],
    [12, 0, 0, 8.5, 88.0],
    [13, 0, 0, 0, 0],
    [14, 0, 0, 0, 0],
    [15, 0, 0, 320.0, 153.0],
    [16, 0, 0, 329.4, 32.3],
    [17, 0, 0, 0, 0],
    [18, 0, 0, 158.0, 30.0],
    [19, 0, 0, 0, 0],
    [20, 0, 0, 680.0, 103.0],
    [21, 0, 0, 274.0, 115.0],
    [22, 0, 0, 0, 0],
    [23, 0, 0, 247.5, 84.6],
    [24, 0, 0, 308.6, -92.2],
    [25, 0, 0, 224.0, 47.2],
    [26, 0, 0, 139.0, 17.0],
    [27, 0, 0, 281.0, 75.0],
    [28, 0, 0, 206.0, 27.6],
    [29, 0, 0, 283.3, 26.9],
    [30, 250.0, 143.7, 0, 0],
    [31, 573.1, 207.1, 9.2, 4.6],   # 31号节点被定为平衡节点
    [32, 650.0, 205.4, 0, 0],
    [33, 632.0, 108.6, 0, 0],
    [34, 508.0, 166.8, 0, 0],
    [35, 650.0, 211.6, 0, 0],
    [36, 560.0, 100.4, 0, 0],
    [37, 540.0, 2.4, 0, 0],
    [38, 830.0, 22.5, 0, 0],
    [39, 1000.0, 89.3, 1104.0, 250.0]
])
DG_and_load[:, 1:5] = DG_and_load[:, 1:5] / 100   # 将功率转化为p.u
# 2、支路矩阵(pdf中表2),各列依次为：支路始、末节点、Rij、Xij、B/2(p.u.)
branch_information = np.array([
    [1, 2, 0.0035, 0.0411, 0.3494],
    [1, 39, 0.0010, 0.0250, 0.3750],
    [2, 3, 0.0013, 0.0151, 0.1286],
    [2, 25, 0.0070, 0.0086, 0.0730],
    [3, 4, 0.0013, 0.0213, 0.1107],
    [3, 18, 0.0011, 0.0133, 0.1069],
    [4, 5, 0.0008, 0.0128, 0.0671],
    [4, 14, 0.0008, 0.0129, 0.0691],
    [5, 6, 0.0002, 0.0026, 0.0217],
    [5, 8, 0.0008, 0.0112, 0.0738],
    [6, 7, 0.0006, 0.0092, 0.0565],
    [6, 11, 0.0007, 0.0082, 0.0695],
    [7, 8, 0.0004, 0.0046, 0.0390],
    [8, 9, 0.0023, 0.0363, 0.1902],
    [9, 39, 0.0010, 0.0250, 0.6000],
    [10, 11, 0.0004, 0.0043, 0.0365],
    [10, 13, 0.0004, 0.0043, 0.0365],
    [13, 14, 0.0009, 0.0101, 0.0862],
    [14, 15, 0.0018, 0.0217, 0.1830],
    [15, 16, 0.0009, 0.0094, 0.0855],
    [16, 17, 0.0007, 0.0089, 0.0671],
    [16, 19, 0.0016, 0.0195, 0.1520],
    [16, 21, 0.0008, 0.0135, 0.1274],
    [16, 24, 0.0003, 0.0059, 0.0340],
    [17, 18, 0.0007, 0.0082, 0.0660],
    [17, 27, 0.0013, 0.0173, 0.1608],
    [21, 22, 0.0008, 0.0140, 0.1283],
    [22, 23, 0.0006, 0.0096, 0.0923],
    [23, 24, 0.0022, 0.0350, 0.1805],
    [25, 26, 0.0032, 0.0323, 0.2565],
    [26, 27, 0.0014, 0.0147, 0.1198],
    [26, 28, 0.0043, 0.0474, 0.3901],
    [26, 29, 0.0057, 0.0625, 0.5145],
    [28, 29, 0.0014, 0.0151, 0.1245]
])
# 3、变压器矩阵(pdf中表3),各列依次为：支路始、末节点、Rij、Xij、kji
transformer_information = np.array([
    [11, 12, 0.0016, 0.0435, 1.006],
    [13, 12, 0.0016, 0.0435, 1.006],
    [31, 6, 0.0000, 0.0250, 1.070],
    [32, 10, 0.0000, 0.0200, 1.070],
    [33, 19, 0.0007, 0.0142, 1.070],
    [34, 20, 0.0009, 0.0180, 1.009],
    [35, 22, 0.0000, 0.0143, 1.025],
    [36, 23, 0.0005, 0.0272, 1.000],
    [37, 25, 0.0006, 0.0232, 1.025],
    [30, 2, 0.0000, 0.0181, 1.025],
    [38, 29, 0.0008, 0.0156, 1.025],
    [20, 19, 0.0007, 0.0138, 1.060]
])
# 4、DG1~DG10: 所在节点编号、最大无功出力、最小无功出力、暂态电抗(p.u.)、机端电压V(p.u.)
DG_information = np.array([
    [39, 400., 140., 0.006, 1.03],
    [31, 300., -100., 0.0647, 0.982],   # 这也是规定的平衡节点
    [32, 300., 150., 0.0531, 0.983],
    [33, 250., 0., 0.0436, 0.997],
    [34, 167., 0., 0.132, 1.012],
    [35, 300., -100., 0.05, 1.049],
    [36, 240., 0., 0.049, 1.063],
    [37, 250., 0., 0.057, 1.028],
    [38, 300., -150., 0.057, 1.026],
    [30, 300., -100., 0.031, 1.047]
])
DG_information[:, 1:3] = DG_information[:, 1:3] / 100  # 将功率转化为p.u.

# 设置收敛精度
episilon = 5.0
# 设置迭代次数
iteration_num = 20
# 设置电压迭代步长
alpha_V = 0.001
# 设置相角迭代步长
alpha_angle = 0.5
# 告知平衡节点编号、PV节点
slack_bus = 31
PV_bus = [30, 34, 38]  # 根据Qdg_real实际情况,将一些PV节点转化为PQ节点

# 【1】初始化实例,进行潮流计算
calculator = supplement_file.AC_Power_Flow_Calculation1(DG_and_load, branch_information, transformer_information,
                DG_information, slack_bus, PV_bus, episilon, iteration_num, alpha_V, alpha_angle)
V, angle, Sij = calculator.power_flow_calculation()
print(V)
print(angle)
print(Sij)

# 【2】初始化实例,进行短路计算
# 从excel中读取潮流计算的结果[已经经过加工]
excel_file_path = '潮流计算结果.xlsx'
df_v = pd.read_excel(excel_file_path, sheet_name='V', header=None)
df_angle = pd.read_excel(excel_file_path, sheet_name='angle', header=None)
df_pij = pd.read_excel(excel_file_path, sheet_name='Pij', header=None)
df_qij = pd.read_excel(excel_file_path, sheet_name='Qij', header=None)
V_final = df_v.iloc[:, -1].tolist()     # 读取最后作为潮流结果的V_final
angle_final = df_angle.iloc[:, -1].tolist()     # 读取最后作为潮流结果的angle_final
Pij_final = df_pij.iloc[:, -1].tolist()     # 读取最后作为潮流结果的Pij_final
Qij_final = df_qij.iloc[:, -1].tolist()     # 读取最后作为潮流结果的Qij_final

# 告知发生故障的节点
problem_node = [2, 16]

# 进行短路计算,返回：短路点电流幅值、相角;所有节点故障时的电压幅值、相角; 各支路故障时首端有功、无功
If, angle_If, V_short_circuit, angle_short_circuit, Sij_short_circuit = (
    calculator.short_circuited_analysis(np.array(V_final), np.array(angle_final), problem_node))
print(If)
print(angle_If)
print(V_short_circuit)
print(angle_short_circuit)
print(Sij_short_circuit)
