import numpy as np


def inverse_hessian_mat(x):
    # 计算hessian矩阵，并求其逆矩阵
    x_hessian = np.dot(x.T, x)
    inverse_hessian = np.linalg.inv(x_hessian)
    return inverse_hessian