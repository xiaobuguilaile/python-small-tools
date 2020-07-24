# -*-coding:utf-8 -*-

'''
@File       : re_formula.py
@Author     : HW Shen
@Date       : 2020/7/23
@Desc       :
'''


import re


# IP地址正则表达式
def re_IP():
    str = "255.255.168.3ahfoweh 255.255.1.10"
    pattrern = re.compile(r'(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)')
    res = pattrern.findall(str)
    print(res)


# 腾讯QQ号正则表达式
def re_QQ():
    str = "ashfiowhei357037797"
    pattrern = re.compile(r'[1-9]([0-9]{5,11})')
    res = pattrern.findall(str)
    print(res)


# 国内固话号码正则表达式
def re_ChineseLocalPhoneNum():
    str = "255.255.168.3ahfoweh 255.255.1.10 357037797 021-6"
    pattrern = re.compile(r'[0-9-()（）]{7,18}')
    res = pattrern.findall(str)
    print(res)


if __name__ == '__main__':
    re_IP()
    re_QQ()
    re_ChineseLocalPhoneNum()