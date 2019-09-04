import datetime
import random
import arrow
import time


# 时间相加 分钟
def dt_add(ori, num, rtype=1):
    oridt = datetime.datetime.strptime(ori, '%Y-%m-%d %H:%M:%S')
    retdt = oridt + datetime.timedelta(minutes=num)
    if rtype == 1:
        return retdt
    return retdt.strftime("%Y-%m-%d %H:%M:%S")


# 日期相加 天
def date_add(ori, num, rtype=1):
    oridt = datetime.datetime.strptime(ori, '%Y-%m-%d %H:%M:%S')
    retdt = oridt + datetime.timedelta(days=num)
    if rtype == 1:
        return retdt
    return retdt.strftime("%Y-%m-%d %H:%M:%S")


# 随机指定预约时间
def gen_appointment_time_random(create_time, random_start=1, random_end=10):
    tp = random.randint(random_start, random_end)
    if tp < 4:
        retdt = dt_add(create_time, random.randint(30, 90))
    elif tp < 9:
        retdt = dt_add(create_time, random.randint(90, 540))
    else:
        retdt = dt_add(create_time, random.randint(540, 1200))
    """21点以后, 8点以前的单, 统一修改"""
    if 8 <= retdt.hour < 21:
        return retdt.strftime("%Y-%m-%d %H:%M:%S")
    if retdt.hour >= 21:
        return (retdt + datetime.timedelta(hours=32-retdt.hour)).strftime("%Y-%m-%d %H:%M:%S")
    return (retdt + datetime.timedelta(hours=8-retdt.hour)).strftime("%Y-%m-%d %H:%M:%S")


def time_sub(dt1, dt2):
    """
    返回秒
    :param dt1:
    :param dt2:
    :return:
    """
    dt_format = '%Y-%m-%d %H:%M:%S'
    tmp = datetime.datetime.strptime(dt1, dt_format) - datetime.datetime.strptime(dt2, dt_format)
    return tmp.days * 86400 + tmp.seconds


def std_now():
    return arrow.now().format('YYYY-MM-DD HH:mm:ss')


def file_time(**kwargs):
    return arrow.now().format('YYYYMMDDHH')


if __name__ == '__main__':
    print(std_now())