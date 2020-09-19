# -*-coding:utf-8 -*-

'''
@File       : text_multiply_split.py
@Author     : HW Shen
@Date       : 2020/9/10
@Desc       : 正则按照多个字符分隔
'''

# python中字符串自带的split方法一次只能使用一个字符对字符串进行分割，但是python的正则模块则可以实现多个字符分割

import re

re.split('[_#|]', 'this_is#a|test')

# 返回的是一个列表（list），输出结果如下：
# ['this', 'is', 'a', 'test']