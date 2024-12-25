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
        desc, required = v.annotation.__metadata__
        # 追加参数信息
        param_list.append({
            "name": k,
            "description": desc,
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


@register_tool
def goal_num_compare(
        num: Annotated[int, '用户参数', True],
        goal: Annotated[int, '目标数', True]
):
    """计算 num - goal"""
    return str(num - goal)
