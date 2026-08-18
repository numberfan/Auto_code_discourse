import os
print("1. 正在导入库...")

from openai import OpenAI
from google.auth import default
from google.auth.transport.requests import Request
import httpx

print("2. 正在获取 GCP 默认凭证与项目 ID...")
credentials, project_id = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
print(f"-> 成功获取！GCP Project ID: {project_id}")

print("3. 正在请求刷新 Google 访问凭证 (若卡在这里说明网络/代理未连通 Google 认证服务器)...")
try:
    credentials.refresh(Request())
    print("-> 凭证刷新成功！")
except Exception as e:
    print(f"-> 凭证刷新报错: {e}")
    exit(1)

region = "us-central1"
base_url = f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"

print("4. 正在初始化 OpenAI 客户端（已配置代理）...")
# 本地代理
custom_http_client = httpx.Client(
    proxy="http://127.0.0.1:7897",
    trust_env=False,
    timeout=30.0
)

client = OpenAI(
    base_url=base_url,
    api_key=credentials.token,
    http_client=custom_http_client
)

print("5. 正在向 Vertex AI 发送测试对话请求...")
try:
    response = client.chat.completions.create(
        model="google/gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hello, are you working?"}]
    )
    print("6. 调用成功！大模型返回结果：")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"-> API 调用报错: {e}")