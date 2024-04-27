import json

# 创建一个Python字典
config = [
    {
        "action": "DeepSquat_up",
        "score": 8,
        "next": {
            "action": "DeepSquat_down",
            "score": 8,
            "next": None
        }
    },
    {
        "action": "HighKnees_prepare",
        "score": 8,
        "next": [
            {
                "action": "HighKnees_left",
                "score": 8,
                "next": None
            },
            {
                "action": "HighKnees_right",
                "score": 8,
                "next": None
            }
        ]
    }
]

# 将Python字典写入JSON文件
with open('config.json', 'w') as f:
    json.dump(config, f)
