"""
Flask 后端：为手机端语音交互网页提供接口
启动: python3 app.py
手机访问: http://<电脑局域网IP>:5001  （电脑和手机需在同一网络下）
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from full_pipeline import capture_and_check

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/understand', methods=['POST'])
def understand():
    user_text = request.json.get('text', '')

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{
            "role": "user",
            "content": (
                f"用户说了这样一句话:「{user_text}」。"
                "请判断用户想要查找/确认的具体物品是什么,然后生成一句自然的确认问句"
                "(格式如'请问您是在寻找XX吗?')。只回复这句确认问句,不要有其他内容。"
                "如果完全无法判断用户想找什么,回复'抱歉,没有听清楚,能再说一遍吗?'"
            )
        }]
    )
    question = response.choices[0].message.content.strip()
    return jsonify({"question": question})


@app.route('/execute', methods=['POST'])
def execute():
    result = capture_and_check()
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
