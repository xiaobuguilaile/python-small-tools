# -*-coding:utf-8 -*-

'''
@File       : time_count.py
@Author     : HW Shen
@Date       : 2020/7/23
@Desc       :
'''
import time


# 引入装饰器，用于计算时间
def get_time(func):
    def wrapper(*args):
        start = time.time()
        func(*args)
        end = time.time()
        print('used time : {}'.format(end-start))
    return wrapper


@get_time
def test():
    time.sleep(3)

if __name__ == '__main__':
    test()