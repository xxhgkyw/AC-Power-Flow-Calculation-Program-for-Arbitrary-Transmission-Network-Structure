# 本脚本存放封装函数
import numpy as np
import math

def get_row_number1(J, S):
    """查找以J为末节点的支路行索引(会返回一个数组,可能有很多)"""
    condition = (S[:, 1] == J)
    row_number = np.where(condition)[0]
    return row_number

def get_row_number2(I, J, S):
    """输入支路(I,J)、需要查找的列表S,返回(I,J)在S中的行索引"""
    condition = (S[:, 0] == I) & (S[:, 1] == J)  # 布尔查找
    row_number = np.where(condition)[0]
    return row_number

def get_row_number3(I, S):
    """查找所有以I为始节点的支路行索引(会返回一个数组,可能有很多)"""
    condition = (S[:, 0] == I)  # 布尔查找
    row_number = np.where(condition)[0]
    return row_number

class AC_Power_Flow_Calculation1:        # 交流潮流计算的类
    def __init__(self, DG_and_load, branch_information, transformer_information, DG_information,
                 slack_bus, PV_bus, eps, iteration_num, alpha_V, alpha_angle):
        self.DG_and_load = DG_and_load
        self.branch_information = branch_information
        self.transformer_information = transformer_information
        self.DG_information = DG_information
        self.eps = eps  # 收敛精度
        self.iteration_num = iteration_num  # 迭代次数
        self.alpha_V = alpha_V  # 设置电压迭代步长
        self.alpha_angle = alpha_angle  # 设置相角迭代步长

        self.node_num = int(len(self.DG_and_load))   # 网络节点个数,目前是39节点
        self.slack_bus = int(slack_bus)  # 平衡节点
        self.PV_bus = PV_bus    # PV节点构成的列表

        # 初始化节点电压、相角、支路潮流矩阵,后续不断更新(平衡节点、PV节点的数据保持不变)
        self.V = np.ones(self.node_num)
        for line_idx in range(len(self.DG_information)):    # 把DG所在节点的V初值按机端电压来设置[涵盖了PV节点]
            node = int(self.DG_information[line_idx][0])    # DG所在节点编号
            self.V[node - 1] = self.DG_information[line_idx][4]
        self.angle = np.zeros(self.node_num)
        self.Sij = np.zeros((len(self.branch_information) + len(self.transformer_information), 4))  # 4列依次为：始、末节点编号、首端有功、无功
        self.Sij[:len(self.branch_information), 0:2] = self.branch_information[:, 0:2]      # 复制始末节点
        self.Sij[len(self.branch_information):, 0:2] = self.transformer_information[:, 0:2]

        # 初始化节点导纳矩阵的实部、虚部[均为常量]
        self.G = np.zeros((self.node_num, self.node_num))   # 39*39
        self.B = np.zeros((self.node_num, self.node_num))

        # 获得节点注入功率矩阵,3列依次为：节点编号、有功注入、无功注入[均为常量]
        self.Si = self.power_injection()

    # 初始化节点注入功率矩阵
    def power_injection(self):
        Si = np.zeros((self.node_num, 3))
        for i in range(self.node_num):  # 从0开始索引所有节点
            Si[i][1] = self.DG_and_load[i][1] - self.DG_and_load[i][3]  # DG有功出力-有功负荷
            Si[i][2] = self.DG_and_load[i][2] - self.DG_and_load[i][4]  # DG无功出力-无功负荷
        return Si

    # 潮流计算主函数
    def power_flow_calculation(self):
        iteration_num = 0
        converged = False

        # 计算节点导纳矩阵self.G、self.B
        self.Y_matrix_calculation()

        while not converged:
            iteration_num += 1
            if iteration_num > self.iteration_num:
                break

            # 1、获得迭代方程中PQ节点的有功常数向量PQ_column_vector_P、偏导子矩阵PQ_sub_matrix_P(二维numpy数组)
            PQ_column_vector_P, PQ_sub_matrix_P = self.PQ_active_power_matrix_derivation()

            # 2、获得迭代方程中PQ节点的无功常数向量PQ_column_vector_Q、偏导子矩阵PQ_sub_matrix_Q(二维numpy数组)
            PQ_column_vector_Q, PQ_sub_matrix_Q = self.PQ_reactive_power_matrix_derivation()

            # 3、获得迭代方程中PV节点的有功常数向量PV_column_vector_P、偏导子矩阵PV_sub_matrix_P(二维numpy数组)
            PV_column_vector_P, PV_sub_matrix_P = self.PV_active_power_matrix_derivation()

            # 4、拼凑得到迭代方程,并求出PQ节点的V、angle增量, 修正self.V、self.angle
            self.renewal(PQ_column_vector_P, PQ_sub_matrix_P, PQ_column_vector_Q, PQ_sub_matrix_Q,
                         PV_column_vector_P, PV_sub_matrix_P)

            # 5、根据功率平衡方程与0的接近程度,判断收敛与否,更新converged
            converged = self._check_convergence(PQ_column_vector_P, PQ_column_vector_Q, PV_column_vector_P)

            # 打印迭代信息
            print(f"迭代次数: {iteration_num}")

        if iteration_num < self.iteration_num:
            print("潮流计算收敛！")
        else:
            print("潮流计算结束,达到最大迭代次数")

        # 计算平衡节点、各PV节点的实际注入功率与原先所给数据
        # (1)平衡节点
        Pi_real_slack_bus, Qi_real_slack_bus, Pdg_real_slack_bus, Qdg_real_slack_bus \
            = self.calculate_bus_power_injection(self.slack_bus)
        print("平衡节点的Pi_real(p.u.)：", Pi_real_slack_bus)
        print("平衡节点的Qi_real(p.u.)：", Qi_real_slack_bus)
        print("平衡节点的Pdg_real(p.u.)：", Pdg_real_slack_bus)
        print("平衡节点的Qdg_real(p.u.)：", Qdg_real_slack_bus)
        # (2)各PV节点
        for i in self.PV_bus:
            node_num = int(i)
            Pi_real_bus, Qi_real_bus, Pdg_real, Qdg_real= self.calculate_bus_power_injection(node_num)
            # print(node_num, "号节点的Pdg_real(p.u.)：", Pdg_real)
            print(node_num, "号节点的Qdg_real(p.u.)：", Qdg_real)

        return self.V, self.angle, self.Sij

    def Y_matrix_calculation(self):
        # 1、先确定self.Y中的互导纳
        for line_idx in range(len(self.branch_information)):    # 逐行遍历branch_information
            from_node = int(self.branch_information[line_idx][0])    # 支路首节点编号
            to_node = int(self.branch_information[line_idx][1])      # 支路末节点编号
            rij = self.branch_information[line_idx][2]      # 支路阻抗(p.u.)
            xij = self.branch_information[line_idx][3]
            gij = rij / (rij ** 2 + xij ** 2)   # 换算为导纳(p.u.)
            bij = -xij / (rij ** 2 + xij ** 2)
            # 更新节点导纳矩阵,注意互导纳与真实值差一个负号
            self.G[from_node - 1][to_node - 1] = -gij
            self.B[from_node - 1][to_node - 1] = -bij

        for line_idx in range(len(self.transformer_information)):   # 逐行遍历transformer_information
            from_node = int(self.transformer_information[line_idx][0])    # 支路首节点编号
            to_node = int(self.transformer_information[line_idx][1])      # 支路末节点编号
            rij = self.transformer_information[line_idx][2]      # 支路阻抗(p.u.)
            xij = self.transformer_information[line_idx][3]
            gij = rij / (rij ** 2 + xij ** 2)   # 换算为导纳(p.u.)
            bij = -xij / (rij ** 2 + xij ** 2)
            # 更新节点导纳矩阵,注意互导纳与真实值差一个负号
            self.G[from_node - 1][to_node - 1] = -gij
            self.B[from_node - 1][to_node - 1] = -bij

        # 2、求self.G、self.B中的自导纳
        for line_idx in range(len(self.G)):
            sum_line_G = np.sum(self.G[line_idx])  # 求self.G该行的和,取负号后,赋值给对角元
            sum_line_B = np.sum(self.B[line_idx])  # 求self.B该行的和,取负号后,赋值给对角元
            self.G[line_idx][line_idx] = -sum_line_G
            self.B[line_idx][line_idx] = -sum_line_B

        for line_idx in range(len(self.branch_information)):    # 逐行遍历branch_information
            from_node = int(self.branch_information[line_idx][0])    # 支路首节点编号
            to_node = int(self.branch_information[line_idx][1])      # 支路末节点编号
            parallel_half_B = self.branch_information[line_idx][4]   # 线路对地并联支路电纳
            # 始末节点的自导纳修正
            self.B[from_node - 1][from_node - 1] += parallel_half_B     # 叠加在始、末节点的自导纳上
            self.B[to_node - 1][to_node - 1] += parallel_half_B

        for line_idx in range(len(self.transformer_information)):   # 逐行遍历transformation_information
            from_node = int(self.transformer_information[line_idx][0])    # 支路首节点编号
            to_node = int(self.transformer_information[line_idx][1])      # 支路末节点编号
            rij = self.transformer_information[line_idx][2]      # 支路阻抗(p.u.)
            xij = self.transformer_information[line_idx][3]
            gij = rij / (rij ** 2 + xij ** 2)   # 换算为导纳(p.u.)
            bij = -xij / (rij ** 2 + xij ** 2)
            kji = self.transformer_information[line_idx][4]     # 变压器变比kji(始→末按照1:kji的格式)
            # 首节点的自导纳修正
            self.G[from_node - 1][from_node - 1] += (kji - 1) / kji * gij
            self.B[from_node - 1][from_node - 1] += (kji - 1) / kji * bij
            # 末节点的自导纳修正
            self.G[to_node - 1][to_node - 1] += (1 - kji) / kji ** 2 * gij
            self.B[to_node - 1][to_node - 1] += (1 - kji) / kji ** 2 * bij

    # 【1】计算PQ节点迭代方程中的有功子块[快速解耦法]
    def PQ_active_power_matrix_derivation(self):
        PQ_column_vector_P = []    # 常数列向量的有功部分,由于i是按顺序遍历的,故里面的节点编号必然是从小到大的顺序,且跳过平衡节点
        PQ_sub_matrix_P = []   # 有功部分的雅克比矩阵子矩阵

        for i in range(1, self.node_num + 1):  # 遍历所有PQ节点,研究【有功方程、Pi】相关的内容
            if (i == self.slack_bus) or (i in self.PV_bus): continue   # 不包括平衡节点、PV节点

            # 1、计算常数向量,每个i对应其中的一个元素
            sum = 0
            sum += self.V[i - 1] ** 2 * self.G[i - 1][i - 1] - self.Si[i - 1][1]    # 非求和部分
            for k in range(1, self.node_num + 1):   # 求和部分,这里是基于KVL的关系,所以会涉及平衡节点的V、angle
                if k == i: continue         # 已经把k=i的部分独立到求和号外面去了
                sum += self.V[i - 1] * self.V[k - 1] * (
                            self.G[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                            + self.B[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                )
            PQ_column_vector_P.append(sum) # 压入常数列向量

            # 2、计算PQ节点有功-相角子矩阵的某一行,只计算对相角的偏导数[注:快速解耦,不考虑有功-电压关系]
            per_line = []   # 注:由于j是按顺序的,故append进去的节点编号肯定也是按从小到大的顺序排列的,且中间跳过平衡节点
            for j in range(1, self.node_num + 1):   # 遍历所有节点, 研究【各相角变量】的偏导数
                if j == self.slack_bus: continue    # 不包括平衡节点
                else:                               # 排除平衡节点后,再讨论j=i、j≠i的偏导数情况
                    if j != i:                      # 节点j(≠i)变量anglej的偏导数
                        element = self.V[i - 1] * self.V[j - 1] * (
                                self.G[i - 1][j - 1] * math.sin(self.angle[i - 1] - self.angle[j - 1])
                                - self.B[i - 1][j - 1] * math.cos(self.angle[i - 1] - self.angle[j - 1])
                        )
                        per_line.append(element)
                    else:                           # 节点j(=i)变量anglej的偏导数
                        sum = 0
                        for k in range(1, self.node_num + 1):   # 为避免混淆,这里用k遍历偏导数中求和项; 这里会涉及平衡节点的V、angle
                            if k == i: continue
                            sum += self.V[i - 1] * self.V[k - 1] * (
                                        -self.G[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                                        + self.B[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                            )
                        per_line.append(sum)
            PQ_sub_matrix_P.append(per_line)        # 将一行的偏导数压入矩阵

        # PQ节点的有功常数向量、偏导数子矩阵均已填充完毕,以numpy数组形式返回出去
        return np.array(PQ_column_vector_P), np.array(PQ_sub_matrix_P)

    # 【2】计算PQ节点迭代方程中的无功子块[快速解耦法]
    def PQ_reactive_power_matrix_derivation(self):
        PQ_column_vector_Q = []  # 常数列向量的无功部分,由于i是按顺序遍历的,故里面的节点编号必然是从小到大的顺序
        PQ_sub_matrix_Q = []  # 无功部分的雅克比矩阵子矩阵

        for i in range(1, self.node_num + 1):  # 遍历所有节点,研究每个PQ节点【无功方程、Qi】相关的内容
            if (i == self.slack_bus) or (i in self.PV_bus): continue  # 不包括平衡节点、PV节点

            # 1、计算PQ节点的无功部分的常数向量,每个i对应其中的一个元素
            sum = 0
            sum += -self.V[i - 1] ** 2 * self.B[i - 1][i - 1] - self.Si[i - 1][2]  # 非求和部分
            for k in range(1, self.node_num + 1):  # 求和部分,这里是基于KVL的关系,所以会涉及平衡节点的V、angle
                if k == i: continue  # 已经把k=i的部分独立到求和号外面去了
                sum += self.V[i - 1] * self.V[k - 1] * (
                        self.G[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                        - self.B[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                )
            PQ_column_vector_Q.append(sum)

            # 2、计算PQ节点的无功-电压子矩阵的某一行
            per_line = []  # 注:由于j是按顺序的,故append进去的节点编号肯定也是按从小到大的顺序排列的,且中间跳过平衡节点
            for j in range(1, self.node_num + 1):  # 遍历所有节点, 研究PQ节点【各V变量】的偏导数
                if (j == self.slack_bus) or (j in self.PV_bus): continue   # 不包括平衡节点、PV节点的V
                else:                              # 排除以上节点后,再讨论j=i、j≠i的偏导数
                    if j != i:                     # 节点j(≠i)变量Vj,偏导数如下
                        element = self.V[i - 1] * (
                                self.G[i - 1][j - 1] * math.sin(self.angle[i - 1] - self.angle[j - 1])
                                - self.B[i - 1][j - 1] * math.cos(self.angle[i - 1] - self.angle[j - 1])
                        )
                        per_line.append(element)
                    else:                          # 节点j(=i)变量Vj,偏导数如下
                        sum = -2 * self.V[i - 1] * self.B[i - 1][i - 1]
                        for k in range(1, self.node_num + 1):  # 为避免混淆,这里用k遍历偏导数中求和项; 这里会涉及平衡节点的V、angle
                            if k == i: continue
                            sum += self.V[k - 1] * (
                                    self.G[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                                    - self.B[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                            )
                        per_line.append(sum)
            PQ_sub_matrix_Q.append(per_line)  # 计算完一行偏导数,压入矩阵

        # 在最外层循环结束后,有功常数向量、偏导数子矩阵均已填充完毕,以numpy数组形式返回出去
        return np.array(PQ_column_vector_Q), np.array(PQ_sub_matrix_Q)

    # 【3】计算PV节点迭代方程中的有功子块[快速解耦法]
    def PV_active_power_matrix_derivation(self):
        PV_column_vector_P = []  # PV节点的常数列向量,下面的遍历顺序可以保证里面的节点编号是从小到大的顺序,且跳过平衡节点
        PV_sub_matrix_P = []     # 有功部分的雅克比矩阵子矩阵

        for i in range(1, self.node_num + 1):  # 遍历所有PV节点,研究【有功方程、Pi】相关的内容
            if i not in self.PV_bus: continue  # 只研究PV节点的

            # 1、计算常数向量,每个i对应其中的一个元素
            sum = 0
            sum += self.V[i - 1] ** 2 * self.G[i - 1][i - 1] - self.Si[i - 1][1]  # 非求和部分
            for k in range(1, self.node_num + 1):  # 求和部分,这里是基于KVL的关系,所以会涉及平衡节点的V、angle
                if k == i: continue  # 已经把k=i的部分独立到求和号外面去了
                sum += self.V[i - 1] * self.V[k - 1] * (
                        self.G[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                        + self.B[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                )
            PV_column_vector_P.append(sum)  # 压入常数列向量

            # 2、计算PV节点有功-相角子矩阵的某一行,只计算对相角的偏导数[注:快速解耦,不考虑有功-电压关系]
            per_line = []  # 注:由于j是按顺序的,故append进去的节点编号肯定也是按从小到大的顺序排列的
            for j in range(1, self.node_num + 1):  # 遍历所有节点的angle
                if j == self.slack_bus: continue   # 不包括平衡节点
                else:                              # 排除平衡节点后,再讨论j=i、j≠i的偏导数
                    if j != i:                     # 节点j(≠i)变量anglej的偏导数
                        element = self.V[i - 1] * self.V[j - 1] * (
                                self.G[i - 1][j - 1] * math.sin(self.angle[i - 1] - self.angle[j - 1])
                                - self.B[i - 1][j - 1] * math.cos(self.angle[i - 1] - self.angle[j - 1])
                        )
                        per_line.append(element)
                    else:                          # 节点j(=i)变量anglej的偏导数
                        sum = 0
                        for k in range(1, self.node_num + 1):  # 为避免混淆,这里用k遍历偏导数中求和项; 这里会涉及平衡节点的V、angle
                            if k == i: continue
                            sum += self.V[i - 1] * self.V[k - 1] * (
                                    -self.G[i - 1][k - 1] * math.sin(self.angle[i - 1] - self.angle[k - 1])
                                    + self.B[i - 1][k - 1] * math.cos(self.angle[i - 1] - self.angle[k - 1])
                            )
                        per_line.append(sum)
            PV_sub_matrix_P.append(per_line)  # 将一行的偏导数压入矩阵

        # PQ节点的有功常数向量、偏导数子矩阵均已填充完毕,以numpy数组形式返回出去
        return np.array(PV_column_vector_P), np.array(PV_sub_matrix_P)

    def renewal(self, PQ_column_vector_P, PQ_sub_matrix_P, PQ_column_vector_Q, PQ_sub_matrix_Q,
                         PV_column_vector_P, PV_sub_matrix_P):
        # 拼凑矩阵
        column_vector = np.concatenate((PQ_column_vector_P, PQ_column_vector_Q, PV_column_vector_P)) # 常数向量
        if len(PV_column_vector_P) == 0:  # 如果PV_column_vector是空的(即全部都是PQ节点)
            Jaccobi_matrix = np.block([  # [J]按快速解耦法,只考虑有功-相角[PQ、PV节点的]、无功-电压的[只包括PQ节点]关系
                [PQ_sub_matrix_P, np.zeros((PQ_sub_matrix_P.shape[0], PQ_sub_matrix_Q.shape[1]), dtype=PQ_sub_matrix_P.dtype)],
                [np.zeros((PQ_sub_matrix_Q.shape[0], PQ_sub_matrix_P.shape[1]), dtype=PQ_sub_matrix_Q.dtype), PQ_sub_matrix_Q]
            ])
        else:
            Jaccobi_matrix = np.block([     # [J]按快速解耦法,只考虑有功-相角[PQ、PV节点的]、无功-电压的[只包括PQ节点]关系
                [PQ_sub_matrix_P, np.zeros((PQ_sub_matrix_P.shape[0], PQ_sub_matrix_Q.shape[1]), dtype=PQ_sub_matrix_P.dtype)],
                [np.zeros((PQ_sub_matrix_Q.shape[0], PQ_sub_matrix_P.shape[1]), dtype=PQ_sub_matrix_Q.dtype), PQ_sub_matrix_Q],
                [PV_sub_matrix_P, np.zeros((PV_sub_matrix_P.shape[0], PQ_sub_matrix_Q.shape[1]),dtype=PV_sub_matrix_P.dtype)]
            ])
        # 解方程得到的增量按顺序为：delta_angle(不含平衡节点)、delta_V(不含平衡节点、PV节点)
        delta_all = -(np.linalg.pinv(Jaccobi_matrix)).dot(column_vector)    # 采用伪逆pinv
        delta_angle = self.alpha_angle * np.concatenate((delta_all[:self.slack_bus - 1], [0.0],    # 插入平衡节点的相角变化量,重新拼接
                                      delta_all[self.slack_bus - 1:self.node_num - 1]))   # 最后注意要-1,因为只有38个元素
        # 各节点电压增量向量,下面这种做法能通用
        delta_V = np.zeros(self.node_num)
        if not self.PV_bus:     # 如果没有PV节点
            delta_V[self.slack_bus - 1] = 1e-6
        else:       # 如果有PV节点
            for i in np.concatenate(([self.slack_bus], self.PV_bus)): delta_V[i - 1] = 1e-6
        zero_indices = np.where(delta_V == 0)[0]    # 找到仍为0的索引(此时只剩PQ节点的位置了)

        for i, a in enumerate(np.array(delta_all[self.node_num - 1:])): # 把delta_all剩下的元素按索引从小到大顺序装填
            delta_V[zero_indices[i]] = a

        if not self.PV_bus:     # 如果没有PV节点,只把平衡节点改回去即可
            delta_V[self.slack_bus - 1] = 0.0
        else:   # 如果有PV节点,则把它和平衡节点一起改回去
            for i in np.concatenate(([self.slack_bus], self.PV_bus)): delta_V[i - 1] = 0.0
        delta_V = self.alpha_V * delta_V

        # 根据增量,修正self.V、self.angle
        for i in range(1, self.node_num + 1):   # 逐个节点遍历
            self.V[i - 1] += delta_V[i - 1]
            self.angle[i - 1] += delta_angle[i - 1]

        # 根据最新self.V、self.angle,计算self.Sij
        for line_idx in range(len(self.Sij)):   # 逐行确定
            from_node = int(self.Sij[line_idx][0])   # 支路始末节点
            to_node = int(self.Sij[line_idx][1])
            gij = -self.G[from_node - 1][to_node - 1]
            bij = -self.B[from_node - 1][to_node - 1]

            Pij = self.V[from_node - 1] ** 2 * gij - self.V[from_node - 1] * self.V[to_node - 1] * (
                        gij * math.cos(self.angle[from_node - 1] - self.angle[to_node - 1])
                        + bij * math.sin(self.angle[from_node - 1] - self.angle[to_node - 1])
            )
            Qij = -self.V[from_node - 1] ** 2 * bij - self.V[from_node - 1] * self.V[to_node - 1] * (
                        gij * math.sin(self.angle[from_node - 1] - self.angle[to_node - 1])
                        - bij * math.cos(self.angle[from_node - 1] - self.angle[to_node - 1])
            )
            self.Sij[line_idx][2] = Pij     # 存入self.Sij矩阵
            self.Sij[line_idx][3] = Qij

    # 根据常数向量与0的接近程度判断收敛性
    def _check_convergence(self, PQ_column_vector_P, PQ_column_vector_Q, PV_column_vector_P):
        PQ_col_P_max = max(abs(PQ_column_vector_P))
        PQ_col_Q_max = max(abs(PQ_column_vector_Q))
        if not self.PV_bus:     # 如果没有PV节点
            if max(PQ_col_P_max, PQ_col_Q_max) < self.eps:
                return True
            return False
        else:       # 如果有PV节点
            PV_col_P_max = max(abs(PV_column_vector_P))
            if max(PQ_col_P_max, PQ_col_Q_max, PV_col_P_max) < self.eps:
                return True
            return False

    # 通用函数：计算某个节点的实际注入功率(可以是平衡节点、PV节点)
    def calculate_bus_power_injection(self, node_num):
        # 找出所有以平衡节点为末节点、首节点的支路(可能有很多)
        branch_idx1 = get_row_number1(int(node_num), self.Sij)
        branch_idx2 = get_row_number3(int(node_num), self.Sij)
        # 有功部分
        sum_inflow_P = 0  # 流入等效大节点的有功之和(首端功率之和)
        sum_outflow_P = 0   # 流出等效大节点的有功之和
        sum_branch_cunsumed_P = 0   # 支路消耗的有功之和
        sum_parallel_gi0 = self.G[int(node_num) - 1][int(node_num) - 1]   # 对地并联支路电导总和,从自导开始往下减
        # 无功部分
        sum_inflow_Q = 0
        sum_outflow_Q = 0
        sum_branch_cunsumed_Q = 0
        sum_parallel_bi0 = self.B[int(node_num) - 1][int(node_num) - 1]   # 对地并联支路的电纳综合,从自纳开始往下减

        for branch_idx in branch_idx1:
            from_node = int(self.Sij[branch_idx][0])     # 该支路的首节点
            to_node = int(self.Sij[branch_idx][1])       # 该支路的末节点
            Pij = self.Sij[branch_idx][2]   # 支路的首端流入有功
            Qij = self.Sij[branch_idx][3]   # 支路的首端流入无功
            gij = -self.G[from_node - 1][to_node - 1]   # 该支路的导纳信息
            bij = -self.B[from_node - 1][to_node - 1]
            rij = gij / (gij ** 2 + bij ** 2)
            xij = -bij / (gij ** 2 + bij ** 2)

            sum_inflow_P += Pij
            sum_branch_cunsumed_P += (Pij ** 2 + Qij ** 2) / (self.V[from_node - 1] ** 2) * rij
            sum_parallel_gi0 -= gij

            sum_inflow_Q += Qij
            sum_branch_cunsumed_Q += (Pij ** 2 + Qij ** 2) / (self.V[from_node - 1] ** 2) * xij
            sum_parallel_bi0 -= bij

        # 找出所有以该节点为首节点的支路(可能有很多)
        for branch_idx in branch_idx2:
            from_node = int(self.Sij[branch_idx][0])
            to_node = int(self.Sij[branch_idx][1])
            Pij = self.Sij[branch_idx][2]
            Qij = self.Sij[branch_idx][3]
            gij = -self.G[from_node - 1][to_node - 1]
            bij = -self.B[from_node - 1][to_node - 1]
            rij = gij / (gij ** 2 + bij ** 2)
            xij = -bij / (gij ** 2 + bij ** 2)

            sum_outflow_P += Pij
            sum_parallel_gi0 -= gij

            sum_outflow_Q += Qij
            sum_parallel_bi0 -= bij

        sum_parallel_consumed_P = self.V[int(node_num) - 1] ** 2 * sum_parallel_gi0
        sum_parallel_consumed_Q = self.V[int(node_num) - 1] ** 2 * sum_parallel_bi0

        # 至此,可以根据能量守恒确定平衡节点真实注入有功、无功了
        Pi_real_bus = - sum_inflow_P + sum_outflow_P + sum_branch_cunsumed_P + sum_parallel_consumed_P
        Qi_real_bus = - sum_inflow_Q + sum_outflow_Q + sum_branch_cunsumed_Q + sum_parallel_consumed_Q

        # 进一步得到该节点DG机组的有功、无功出力
        Pdg_real_bus = Pi_real_bus + self.DG_and_load[node_num - 1][3]
        Qdg_real_bus = Qi_real_bus + self.DG_and_load[node_num - 1][4]

        return Pi_real_bus, Qi_real_bus, Pdg_real_bus, Qdg_real_bus

    # 进行短路计算
    def short_circuited_analysis(self, V_final, angle_final, problem_node):
        # 【1】对节点导纳矩阵进行修正
        G_new = self.G.copy()
        B_new = self.B.copy()
        # 添加DG暂态电抗
        for idx in range(len(self.DG_information)):
            DG_node = int(self.DG_information[idx][0])   # DG所在节点编号
            y_dg_temporal = -1 / self.DG_information[idx][3]     # 该DG的暂态电抗的倒数[注意有个负号]
            B_new[DG_node - 1][DG_node - 1] += y_dg_temporal    # 叠加到自纳矩阵中
        # 添加各节点负荷阻抗
        for idx in range(len(self.DG_and_load)):
            node = int(self.DG_and_load[idx][0])    # 某节点编号
            P_load = self.DG_and_load[idx][3]   # 有功负荷
            Q_load = self.DG_and_load[idx][4]   # 无功负荷
            G_new[node - 1][node - 1] += P_load / V_final[node - 1] ** 2    # 修正G_new
            B_new[node - 1][node - 1] += -Q_load / V_final[node - 1] ** 2   # 修正B_new

        # 【2】# 获得每个故障点在Z'中对应的R_k、X_k
        R_k = []    # 将每个短路点的向量都压入,形成一个二维数组
        X_k = []
        coefficient_matrix = np.block([     # 拼接得到系数矩阵
            [G_new, -B_new],
            [B_new, G_new]
        ])
        for k in problem_node:
            e = np.zeros(self.node_num); e[int(k) - 1] = 1  # 得到第k个元素为1、其余元素为0的向量
            coefficient_vector = np.concatenate((e, np.zeros(self.node_num)))
            solution = (np.linalg.pinv(coefficient_matrix)).dot(coefficient_vector)
            R_k.append(solution[:self.node_num])  # 获得Z'第k列的实部向量,压入
            X_k.append(solution[self.node_num:])  # 获得Z'第k列的虚部向量,压入

        # 【3】计算每个故障点的短路电流幅值、相角
        If = []
        angle_If = []
        for i in range(len(R_k)):
            k = int(problem_node[i])    # 故障节点的编号
            R_kk = R_k[i][k - 1]  # 第k列的第k个元素
            X_kk = X_k[i][k - 1]
            denominator = (R_kk ** 2 + X_kk ** 2) / V_final[k - 1]  # 分母
            If_real = (R_kk * math.cos(angle_final[k - 1]) + X_kk * math.sin(angle_final[k - 1])) / denominator
            If_imag = (R_kk * math.sin(angle_final[k - 1]) - X_kk * math.cos(angle_final[k - 1])) / denominator
            amplitude = math.sqrt(If_real ** 2 + If_imag ** 2)  # 电流幅值
            phase = math.atan(If_imag / If_real)    # 电流相角(弧度制)
            If.append(amplitude)
            angle_If.append(phase)

        # 【4】计算每个节点在故障点发生短路后的电压幅值、相角
        V_short_circuit = np.ones(self.node_num)
        angle_short_circuit = np.zeros(self.node_num)
        for i in range(1, self.node_num + 1):
            Vi_steady = V_final[i - 1]  # 潮流稳态电压
            anglei_steady = angle_final[i - 1]  # 潮流稳态相角

            # Vi_real、Vi_imag按照增量叠加的方式,依次叠加上所有故障点的影响
            Vi_real = Vi_steady * math.cos(anglei_steady)   # 实部的初始值
            Vi_imag = Vi_steady * math.sin(anglei_steady)   # 虚部的初始值
            for j in range(len(R_k)):  # 遍历所有故障节点
                k = int(problem_node[j])    # 故障节点的编号
                R_ik = R_k[j][i - 1]   # 获得(i,k)元的实部、虚部
                X_ik = X_k[j][i - 1]
                If_k = If[j]    # 获得k号节点(故障点)的短路电流幅值
                angle_If_k = angle_If[j]    # 获得k号节点(故障点)的短路电流相角

                Vi_real += (-R_ik * If_k * math.cos(angle_If_k) + X_ik * If_k * math.sin(angle_If_k))
                Vi_imag += -(R_ik * If_k * math.sin(angle_If_k) + X_ik * If_k * math.cos(angle_If_k))

            Vi_amplitude = math.sqrt(Vi_real ** 2 + Vi_imag ** 2)   # 该节点在网络发生故障后的电压幅值
            Vi_phase = math.atan(Vi_imag / Vi_real)   # 该节点在网络发生故障后的相角
            V_short_circuit[i - 1] = Vi_amplitude   # 将其更新到指定向量中
            angle_short_circuit[i - 1] = Vi_phase

        # 【5】在确定V_short_circuit、angle_short_circuit后,可以进一步导出各支路功率情况
        # 先获得合并线路、变压器的支路首端功率矩阵&阻抗信息
        Sij_short_circuit = np.zeros((len(self.branch_information) + len(self.transformer_information), 4))  # 4列依次为：始、末节点编号、首端有功、无功
        Sij_short_circuit[:len(self.branch_information), 0:2] = self.branch_information[:, 0:2]  # 复制始末节点信息
        Sij_short_circuit[len(self.branch_information):, 0:2] = self.transformer_information[:, 0:2]
        branch_and_transformer = np.zeros((len(self.branch_information) + len(self.transformer_information), 4))
        branch_and_transformer[:len(self.branch_information), 0:2] = self.branch_information[:, 0:2]  # 复制始末节点
        branch_and_transformer[:len(self.branch_information), 2:4] = self.branch_information[:, 2:4]  # 复制支路阻抗
        branch_and_transformer[len(self.branch_information):, 0:2] = self.transformer_information[:, 0:2]  # 复制始末节点
        branch_and_transformer[len(self.branch_information):, 2:4] = self.transformer_information[:, 2:4]  # 复制支路阻抗
        # 按照公式导出实际功率
        for branch_idx in range(len(Sij_short_circuit)):
            # 获取当前支路信息
            from_node = int(Sij_short_circuit[branch_idx][0])  # 某支路始节点
            to_node = int(Sij_short_circuit[branch_idx][1])  # 某支路末节点
            rij = branch_and_transformer[branch_idx][2]  # 该支路阻抗信息
            xij = branch_and_transformer[branch_idx][3]
            gij = rij / (rij ** 2 + xij ** 2)  # 转化为gij、bij
            bij = -xij / (rij ** 2 + xij ** 2)

            # 计算Pij、Qij
            Pij_short_circuit = V_short_circuit[from_node - 1] ** 2 * gij - V_short_circuit[from_node - 1] * V_short_circuit[to_node - 1] * (
                    gij * math.cos(angle_short_circuit[from_node - 1] - angle_short_circuit[to_node - 1])
                    + bij * math.sin(angle_short_circuit[from_node - 1] - angle_short_circuit[to_node - 1])
            )
            Qij_short_circuit = -V_short_circuit[from_node - 1] ** 2 * bij - V_short_circuit[from_node - 1] * V_short_circuit[to_node - 1] * (
                    gij * math.sin(angle_short_circuit[from_node - 1] - angle_short_circuit[to_node - 1])
                    - bij * math.cos(angle_short_circuit[from_node - 1] - angle_short_circuit[to_node - 1])
            )
            Sij_short_circuit[branch_idx][2] = Pij_short_circuit
            Sij_short_circuit[branch_idx][3] = Qij_short_circuit

        # 返回：短路点电流幅值、相角;所有节点故障时的电压幅值、相角; 各支路故障时首端有功、无功
        return If, angle_If, V_short_circuit, angle_short_circuit, Sij_short_circuit
