import sys
from typing import Annotated

sys.path.append(r'D:\projects\学校\课程笔记\cq_ai_240701\大模型部署\chatglm.cpp\示例代码')

import requests
import json
from register_tool import register_tool


@register_tool
def get_weather(city: Annotated[str, '城市名称', True]):
    """查询城市的实时天气"""
    # 请求网络
    response = requests.get(f'https://api.seniverse.com/v3/weather/now.json?key=SqXOlV6wp9XtP_zqZ&location={city}')
    # 判断是否请求成功
    if response.status_code == 200:
        result = json.loads(response.text)
        return result['results'][0]['now']
    else:
        return '未查到天气信息'
