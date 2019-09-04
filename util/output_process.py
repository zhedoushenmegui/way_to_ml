import os
import json
import arrow
import shutil
import pandas as pd
import pickle
import sys
"""
模型导出, 模型相关内容导出, 可视化
"""


project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('.')), '../'))
sys.path.append(project_path)


def dump_feature_map(feature_map, file_path):
    """
    ### 导出fmap xgb
    :param feature_map:
    :param file_path:
    :return:
    """
    with open(file_path, "w") as file:
        file.writelines(map(lambda x: '\t'.join([str(_) for _ in x]) + '\n', feature_map))


def dump_label_encode_map(label_encode_map, file_path):
    """

    :param label_encode_map:
    :param file_path:
    :return:
    """
    json.dump(label_encode_map, open(file_path, "w"))


def dump_XGBModel(xgb_model, file_path):
    """
    导出xgb 模型
    :param xgb_model:
    :param file_path:
    :return:
    """
    xgb_model.dump_model(file_path, with_stats=True)


def save_XGBModel(xgb_model, file_path):
    xgb_model.save_model(file_path)


def dump_XGB_score(score, file_path):
    with open(file_path, "w") as file:
        file.writelines(map(lambda x: str(x) + '\n', score))


def jpmml_conv(bmodel, fmap, output):
    """xgb model 生成 jpmml 文件
    :param bmodel: 二进制模型文件名
    :param fmap: feature map 文件名
    :param output: jpmml 路径和文件名
    :type bmodel: str
    :type fmap: str
    :type output: str
    """
    cmd = f'java -jar {project_path}/util/jpmml-xgboost.jar --model-input {bmodel} --fmap-input {fmap} --target-name ' \
          f'mpg --pmml-output {output} '
    os.system(cmd)
