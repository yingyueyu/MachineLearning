# 注册工具
# 作用: 用来告诉AI可以使用哪些工具
import inspect
from typing import Annotated, List, Dict, Optional
from types import GenericAlias

tool_meta_datas = {}


# 注册工具
def register_tool(func):
    # 获取工具名称
    name = func.__name__
    # 获取文档注释作为工具的描述信息
    description = inspect.getdoc(func)
    # 获取函数签名，并获取其中的参数
    params = inspect.signature(func).parameters
    param_list = []
    # 获取参数
    for k, v in params.items():
        # 获取参数类型
        typ = v.annotation.__origin__
        # 检查是否是泛型，并根据条件赋值参数类型
        typ = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        # 获取参数元数据并结构成描述信息和required
        description, required = v.annotation.__metadata__
        # 追加参数信息
        param_list.append({
            "name": k,
            "description": description,
            "type": typ,
            "required": required
        })
    # 创建并保存工具的元数据
    tool_meta_datas[name] = {
        "name": name,
        "description": description,
        "params": param_list
    }
    # 返回函数本身
    return func


if __name__ == '__main__':
    # 注册工具作为装饰器来调用
    # 函数add需要添加文档注释
    @register_tool
    def add(a: Annotated[float, '第一个加数', True], b: Annotated[float, '第二个加数', True]) -> float:
        """加法运算"""
        return a + b


    # 定义泛型列表
    def multi(ops: Annotated[List[float], '乘数集合', True]):
        return ops[0] * ops[1]


    # 定义泛型字典
    def div(ops: Annotated[Dict[str, float], '乘数集合', True]):
        return ops['a'] / ops['b']


    # Optional: 表达参数为可选参数
    def greeting(message: Annotated[Optional[str], '乘数集合', True] = None):
        if message is None:
            message = 'greeting'
        print(message)


    greeting()
    greeting('你好')
