import dashscope

# 如果环境变量配置无效请启用以下代码
dashscope.api_key = 'sk-5806dc5ecf1b4438bc35629dacbf0553'

# respose获得的为
messages = [{'role': 'user', 'content': '如何做炒西红柿鸡蛋？'}]

response = dashscope.Generation.call(dashscope.Generation.Models.qwen_turbo, messages=messages, result_format='message')
print(response)


# {"status_code": 200, "request_id": "6d53e094-e8bc-9d88-a84a-6085c9425ad8", "code": "", "message": "", "output": {"token_ids": [108386, 11319], "tokens": ["你好", "？"]}, "usage": {"input_tokens": 2}}
