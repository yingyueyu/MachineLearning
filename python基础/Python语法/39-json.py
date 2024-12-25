# import json

# json字符串转python对象
# json_str = '''
#     {
#       "name": "小明",
#       "age": 22,
#       "hobby": ["唱跳", "爬山", "游泳"],
#       "isstudent": true,
#       "money": null,
#       "adrr": {
#         "city": "cq",
#         "postcode": 510030
#       }
#     }
# '''
#
# py_obj = json.loads(json_str)
# print(py_obj)
# print(type(py_obj))
# print(py_obj.get('name'))
# print(type(py_obj.get("hobby")))

# python转JSON字符串
# dict1 = {}
# dict1['name'] = '小王'
# dict1['age'] = 21
# dict1['hobby'] = ('唱歌', '跳舞', '打篮球')
# dict1['isstudent'] = False
# dict1['money'] = None
# dict1['addr'] = {'city': 'gz', 'postcode': 400000}
#
# json_str = json.dumps(dict1)
# print(json_str)
#
# print(json.loads(json_str))


# 保存json文件
# dict1 = {}
# dict1['name'] = '小王'
# dict1['age'] = 21
# dict1['hobby'] = ('唱歌', '跳舞', '打篮球')
# dict1['isstudent'] = False
# dict1['money'] = None
# dict1['addr'] = {'city': 'gz', 'postcode': 400000}
#
# with open('student.json', 'w', encoding='utf-8') as f1:
#     # 把python对象转换为字符串，并且写入到文件
#     json.dump(dict1, f1)


# 读取Json文件
# with open('student.json', 'r', encoding='utf-8') as f2:
#     py_obj = json.load(f2)
#     print(py_obj)


# 课堂练习
import json

# Python对象
data = {
    'name': '人工智能数学',
    'author': '汤普森',
    'price': '85'
}

# 将Python对象转换为JSON格式的字符串
json_string = json.dumps(data)

# 将JSON字符串写入文件
with open('data.json', 'w', encoding='utf-8') as f1:
    json.dump(data, f1)

# 从文件中读取JSON字符串
with open('data.json', 'r', encoding='utf-8') as f2:
    json_data = json.load(f2)
    print(json_data)