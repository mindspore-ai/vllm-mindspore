import requests
import json
import base64

# 1. 将图片编码为Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_base64 = image_to_base64("path/to/your/image.jpg")

# 2. 构建请求数据
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload = {
    "model": "qwen-vl",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片中的场景。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 1024
}

# 3. 发送请求并处理结果
response = requests.post(url, headers=headers, data=json.dumps(payload))
answer = response.json()['choices'][0]['message']['content']
print(answer)