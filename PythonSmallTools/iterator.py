# -*-coding:utf-8 -*-

'''
@File       : iterator.py
@Author     : HW Shen
@Date       : 2020/8/28
@Desc       : 生成器
https://www.cnblogs.com/yxi-liu/p/8193472.html
'''


# 先看一道经典的生成式问题
def generator():
    def gen():
        for i in range(4):
            yield i
    base = gen()
    # print(list(base))  # [0, 1, 2, 3]
    for n in (2, 10):
        base = (i+n for i in base)  # for i in base中的 base =[0, 1, 2, 3]
        # print(list(base))
    print(list(base))


# 生成器表达式，会在程序执行的过程中运行for 后面的代码，并对for后面的代码进行赋值，而for之前的代码以及生成器函数并不会执行，只会进行编译。
# 尽管，生成器表达式代码更简洁，但在生成器初始化和生成器调用的效率上都表现出了与传统生成器函数的差距。
def test():
    a = 3
    b = (i for i in range(a))
    a = 5
    print(list(b))

    a0 = 3
    b0 = (a0 for i in range(3))
    a0 = 5
    print(list(b0))


import timeit

def b1():
    a1 = 999
    def c1():
        for i in range(a1):
            yield i
    list(c1())


def b2():
    a2 = 999
    c2 = (i for i in range(a2))
    list(c2)


if __name__ == '__main__':
    generator()
    # 答案：[20, 21, 22, 23]
    # 因为for循环了两次，并对base从新赋值了，所以可以简化为（i+n for i in (i+n for i in base)），
    # 而n全部引用了后赋值的10,最里面的base引用的是gen, base=[0, 1, 2, 3].

    # test()
    # [0, 1, 2]
    # [5, 5, 5]

    # print(timeit.timeit(stmt=b1, number=10000))
    # print(timeit.timeit(stmt=b2, number=10000))  # 生成器表达式确实要慢一些
    # 我们看到生成器表达式提供的便利的确是以效率的损耗作为代价的。
    # 进一步的验证表明：生成器表达式初始化的过程相比生成器函数需要花费更多的时间（接近2倍），但是由于初始化的时间过短，
    # 并不是形成差距的主要原因。函数模式的生成器会随着next()次数的增加在时间上逐步拉开与生成器表达式差距。
    # 调用效率的差距才是主要原因。