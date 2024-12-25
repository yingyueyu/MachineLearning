# import hello
# 如果模块名字太长，可以用as关键字给它一个别名，代码中可以使用别名调用方法
import hello as h


# hello.sayHello()
# hello.sayGoodBye()

h.sayHello()
h.sayGoodBye()
# 也可访问模块的变量
print(h.num)


from hello import sayHello
from hello import sayGoodBye
from hello import num

sayHello()
sayGoodBye()
print(num)



# import test.aaa as aaa
from test import aaa
from test.sub import bbb

aaa.info()
bbb.info()

