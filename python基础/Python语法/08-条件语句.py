# 电脑生成一个随机数，让人猜这个数（键盘输入），猜错了就提示：哈哈，你猜错了

# 导入产生随机数的库
import random

# 产生随机整数， 范围1到10
r = random.randint(1, 10)
# 输入你猜的数字
guess = int(input("请输入你猜的数字："))
# 用if语句判断是否猜中
if r != guess:
    print("哈哈，你猜错了, 实际是：",  r)



