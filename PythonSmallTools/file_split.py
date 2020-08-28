# -*-coding:utf-8 -*-

'''
@File       : file_split.py
@Author     : HW Shen
@Date       : 2020/7/26
@Desc       :
'''

from datetime import datetime


def Main():

    source_dir = 'ftp_Title_Content_Corpus20200726.csv'
    target_dir = 'origin/'  # 接受拆分文件的文件夹

    flag = 0   # 计数器，用于控制多少行分成一个文件
    name = 1  # 文件名
    dataList = []  # 存放数据
    print("开始。。。。。")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(source_dir, 'r', encoding='utf-8') as f_source:
        for line in f_source:
            flag += 1
            dataList.append(line)
            if flag == 100000:
                with open(target_dir + "ftp_" + str(name) + ".txt", 'w+', encoding='utf-8') as f_target:
                    for data in dataList:
                        f_target.write(data)
                name += 1
                flag = 0
                dataList = []

    # 处理最后一批行数少于10万的
    with open(target_dir + "ftp_" + str(name) + ".txt", 'w+', encoding='utf-8') as f_target:
        for data in dataList:
            f_target.write(data)

    print("完成。。。。。")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    Main()

    import glob

    # 查看并读取拆分后的所有文件
    txts = glob.glob('../data/origin/*.txt')
    for txt in txts:
        print(txt)