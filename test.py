# -*-coding:utf-8 -*-

'''
@File       : test.py
@Author     : HW Shen
@Date       : 2020/8/6
@Desc       :
'''


def _match(str1, str2):
    str1 = str1.replace("有限公司", "").replace("贸易", "").replace("集团", "")
    if str1[:2] in str2:
        return True
    return False


f1 = open("会后满意度.csv", encoding="utf-8")
f2 = open("签到名单.csv", encoding="utf-8")

content1 = f1.readlines()
content2 = f2.readlines()

print("content1: ", content1[1], end="")
# print(content1[1].strip().split(",")[0])
# print(content1[1].strip().split(",")[2])

print("content2: ", content2[1], end="")
# print(content2[1].strip().split(",")[2])
# print(content2[1].strip().split(",")[1])


if __name__ == '__main__':
    res = [False] * len(content2)

    for i in range(1, len(content2)):
        res[i] = False
        item2 = content2[i].strip().split(",")
        print("item2: ", item2)
        for j in range(1, len(content1)):
            item1 = content1[j].strip().split(",")
            print("item1: ", item1)
            if item2[2][0] == item1[0][0] and _match(item2[1], item1[2]):
                res[i] = True
                break

    with open("签到名单2.csv", 'w', encoding="utf-8") as fw:
        fw.write("判断," + content2[0])
        for k in range(1, len(content2)):
            if res[k]:
                fw.write("是," + content2[k])
            else:
                fw.write("否," + content2[k])
    print(res)

    # test1 = "徐敏,人事经理,公信贸易,201-500,非常有帮助,非常适用,10,完全理解,10,10,非常满意,薪酬与员工激励，涉疫劳动关系处理,"
    # test2 = "1,公信,徐老师,老板,确认,sofia,是,1,1"
    # print(_match(test1.strip().split(",")[2], test2.strip().split(",")[1]))