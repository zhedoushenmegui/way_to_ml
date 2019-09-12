from collections import Counter
import math


class DecisionTree:
    @staticmethod
    # 计算gini
    def cal_gini(data):
        if len(data) == 0:
            return 0
        c = Counter(data)
        l = len(data)
        return 1 - sum([(c.get(k) / l) ** 2 for k in c])

    @staticmethod
    # 计算信息熵
    def cal_entropy(data):
        if len(data) == 0:
            return 0
        c = Counter(data)
        l = len(data)
        return -sum([c.get(k) / l * math.log2(c.get(k) / l) for k in c])

