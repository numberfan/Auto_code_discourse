#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 13:52
# @Description  : LLM配置管理（默认vertex_ai, 备选deepseek)

import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex_ai").lower()
REGION = os.getenv("VERTEX_AI_REGION", "us-central1")

PROJECT_ID = None

# 提取本地项目 ID (google需要）
def _get_gcp_project_id():
    # 优先从环境变量读取
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project:
        return env_project

    from google.auth import default
    _, project_id = default()
    if not project_id:
        raise ValueError("未能自动获取 GCP 项目 ID")
    return project_id

PROVIDERS = {
    "vertex_ai": {
        "api_key_env": None,
        "default_model": "google/gemini-2.5-pro",
        "supports_json_mode": True,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "supports_json_mode": True,
    }
}

# config.py 中添加此函数
def print_status():
    model = get_model_name()
    print(f"\n{'='*40}")
    print(f"系统配置已加载")
    print(f"Provider: {LLM_PROVIDER.upper()}")
    print(f"Model:    {model}")
    print(f"{'='*40}\n")

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

        base_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi"

        # 获取 Google 访问凭证
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = credentials.token

        # 本地代理
        proxy_url = os.getenv("PROXY_URL", "http://127.0.0.1:7897")
        custom_http_client = httpx.Client(
            proxy=proxy_url,
            trust_env=False,
            timeout=60.0
        )

        # custom_http_client = httpx.Client(trust_env=False)
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