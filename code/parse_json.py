import json

# 打开JSON文件
with open('config.json', 'r') as file:
    # 读取文件内容
    data = file.read()

# 解析JSON字符串为Python对象
parsed_data = json.loads(data)

# 现在你可以使用parsed_data变量来访问JSON数据
print(parsed_data[0])