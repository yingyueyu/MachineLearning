import sys

sys.path.append(r'D:\projects\学校\课程笔记\cq_ai_240701\大模型部署\chatglm.cpp\示例代码')

from typing import Annotated
from register_tool import register_tool


@register_tool
def add_fn(a: Annotated[float, '第一个加数', True], b: Annotated[float, '第二个加数', True]):
    """加法运算"""
    return a + b

