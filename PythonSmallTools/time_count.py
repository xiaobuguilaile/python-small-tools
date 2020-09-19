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


def timmer(func):
    def deco(*args, **kwargs):
        print('\n函数：\033[32;1m{_funcname_}()\033[0m 开始运行：'.format(_funcname_=func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('函数: \033[32;1m{_funcname_}()\033[0m 运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res

    return deco


@timmer
def test():
    time.sleep(3)

if __name__ == '__main__':
    test()