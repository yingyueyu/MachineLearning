import json
json_string = '{"prd_no": 20240710001, "prd_name": "I7 CUP", "prd_type": "I7", "prd_price": 1000, "prd_provider":"intel"}'
data_dict = json.loads(json_string)
print(data_dict['prd_price'])