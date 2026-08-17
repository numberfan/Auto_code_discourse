#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 13:52
# @Description  : LLM配置管理（默认vertex_ai, 备选deepseek)

import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
REGION = os.getenv("VERTEX_AI_REGION", "us-central1")

PROJECT_ID = None

# 提取本地项目 ID
def _get_gcp_project_id():
    from google.auth import default
    _, project_id = default()
    return project_id


PROJECT_ID = _get_gcp_project_id()

PROVIDERS = {
    "vertex_ai": {
        "api_key_env": None,
        "base_url": f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{{PROJECT_ID}}/locations/{REGION}/endpoints/openapi",
        "default_model": f"publishers/google/models/gemini-1.5-pro-001",
        "supports_json_mode": True,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "supports_json_mode": True,
    }
}

def get_client() -> OpenAI:

    """OpenAI SDK 统一调用"""
    provider = PROVIDERS.get(LLM_PROVIDER)
    if not provider:
        raise ValueError(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}")

    if LLM_PROVIDER == "vertex_ai":
        from google.auth import default
        from google.auth.transport.requests import Request
        # 获取 GCP 项目 ID
        global PROJECT_ID
        if PROJECT_ID is None:
            PROJECT_ID = _get_gcp_project_id()

        base_url = provider["base_url"].format(PROJECT_ID=PROJECT_ID)

        # 获取 Google 访问凭证
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = credentials.token

        # 如果使用代理，请确保代理支持 HTTPS；否则注释掉代理部分
        # custom_http_client = httpx.Client(proxy="socks5://127.0.0.1:7897", trust_env=False)
        custom_http_client = httpx.Client(trust_env=False)  # 完全不使用环境代理
        return OpenAI(base_url=base_url, api_key=token, http_client=custom_http_client)
    else:
        api_key = os.getenv(provider["api_key_env"])
        if not api_key:
            raise ValueError(f"缺少 {provider['api_key_env']} 环境变量")

        # DeepSeek 直连
        custom_http_client = httpx.Client(trust_env=False)
        return OpenAI(api_key=api_key, base_url=provider["base_url"], http_client=custom_http_client)


def get_model_name() -> str:
    provider = PROVIDERS.get(LLM_PROVIDER)
    return os.getenv("LLM_MODEL", provider["default_model"])


def supports_json_mode() -> bool:
    provider = PROVIDERS.get(LLM_PROVIDER)
    return provider["supports_json_mode"]