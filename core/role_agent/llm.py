"""LLM 配置解析与结构化调用。

优先使用设置页填写的 Base URL / Key / 模型；未填写 Key 时回退到项目根目录 .env 中的 OpenRouter Key。
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, TypeVar

import openai
from pydantic import BaseModel, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
ENV_KEY_NAMES = ("OPENROUTER_API_KEY", "OpenRouter_API_KEY", "OPENROUTER_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

T = TypeVar("T", bound=BaseModel)

# 鉴权、网络类错误不必再换结构化方式重试
FATAL_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.APIConnectionError,
    openai.RateLimitError,
)


class LLMConfigError(Exception):
    """LLM 配置缺失或无效"""


@dataclass
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: int
    source: str  # 'settings' | 'env'

    @property
    def is_openrouter(self) -> bool:
        return 'openrouter.ai' in self.base_url


def _read_env_key() -> str:
    for name in ENV_KEY_NAMES:
        value = os.environ.get(name, '').strip()
        if value:
            return value
    try:
        from dotenv import dotenv_values
    except ImportError:
        return ""
    for env_path in (PROJECT_ROOT / '.env', Path.cwd() / '.env'):
        if not env_path.exists():
            continue
        values = dotenv_values(env_path)
        for name in ENV_KEY_NAMES:
            value = (values.get(name) or '').strip()
            if value:
                return value
    return ""


def normalize_base_url(base_url: str) -> str:
    """把用户填写的地址规范成 OpenAI SDK 需要的 .../v1 形式。"""
    base = (base_url or '').strip().rstrip('/')
    if base.endswith('/chat/completions'):
        base = base[:-len('/chat/completions')]
    if re.search(r'/v\d+$', base):
        return base
    return f"{base}/v1"


def resolve_llm_settings(config_manager) -> LLMSettings:
    base_url = (config_manager.get('llm_base_url', '') or '').strip()
    api_key = (config_manager.get('llm_api_key', '') or '').strip()
    model = (config_manager.get('llm_model', '') or '').strip()
    timeout = int(config_manager.get('llm_timeout_sec', 60) or 60)
    source = 'settings'

    # 兼容旧逻辑：误把 URL 填进 Key 时视为未填写
    if api_key.lower().startswith('http'):
        api_key = ''

    if not api_key and (not base_url or 'openrouter.ai' in base_url):
        env_key = _read_env_key()
        if env_key:
            api_key = env_key
            base_url = base_url or OPENROUTER_BASE_URL
            model = model or OPENROUTER_DEFAULT_MODEL
            source = 'env'

    if not base_url:
        raise LLMConfigError("请先在设置页面填写 LLM Base URL，或在 .env 中配置 OPENROUTER_API_KEY")
    if not model:
        raise LLMConfigError("请先在设置页面填写 LLM 模型名称")

    return LLMSettings(
        base_url=normalize_base_url(base_url),
        api_key=api_key,
        model=model,
        timeout=timeout,
        source=source,
    )


def build_chat_model(settings: LLMSettings, temperature: float = 0.1):
    from langchain_openai import ChatOpenAI

    headers = {}
    if settings.is_openrouter:
        headers = {
            'HTTP-Referer': 'https://github.com/Moeary/CosyVoiceDesktop',
            'X-Title': 'CosyVoice Desktop',
        }
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        # 本地无鉴权服务也要求非空 key
        api_key=settings.api_key or 'sk-no-key',
        temperature=temperature,
        timeout=settings.timeout,
        max_retries=2,
        default_headers=headers or None,
    )


def _message_text(message) -> str:
    content = getattr(message, 'content', message)
    if isinstance(content, list):
        return ''.join(
            item.get('text', '') if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or '')


def _extract_json(text: str):
    content = (text or '').strip()
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', content, re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(content)
    object_match = re.search(r'(\{[\s\S]*\})', content)
    if object_match:
        candidates.append(object_match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("无法从模型输出中解析 JSON")


def invoke_structured(llm, schema: Type[T], messages: List, usage: Optional[dict] = None) -> T:
    """按 function_calling → json_schema → 纯文本 JSON 的顺序逐级降级，返回经 Pydantic 校验的对象。

    纯文本模式下解析失败会把错误回传给模型修复一次。
    """
    errors = []
    for method in ('function_calling', 'json_schema'):
        try:
            runnable = llm.with_structured_output(schema, method=method, include_raw=True)
            result = runnable.invoke(messages)
            accumulate_usage(usage, result.get('raw'))
            parsed = result.get('parsed')
            if parsed is not None:
                return parsed
            errors.append(f"{method}: {result.get('parsing_error')}")
        except FATAL_ERRORS:
            raise
        except Exception as e:
            errors.append(f"{method}: {e}")

    from langchain_core.messages import AIMessage, HumanMessage

    schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    prompt = list(messages) + [HumanMessage(
        content=f"请只输出一个符合以下 JSON Schema 的 JSON 对象，不要任何解释：\n{schema_hint}"
    )]
    for _ in range(2):
        response = llm.invoke(prompt)
        accumulate_usage(usage, response)
        text = _message_text(response)
        try:
            return schema.model_validate(_extract_json(text))
        except (ValueError, ValidationError) as e:
            errors.append(f"json_text: {e}")
            prompt = prompt + [AIMessage(content=text), HumanMessage(
                content=f"上面的输出无法通过校验：{str(e)[:500]}\n请修正后重新只输出 JSON。"
            )]

    raise RuntimeError("模型未能返回有效的结构化结果: " + " | ".join(str(e)[:200] for e in errors))


def accumulate_usage(usage: Optional[dict], message):
    if usage is None or message is None:
        return
    meta = getattr(message, 'usage_metadata', None) or {}
    usage['calls'] = usage.get('calls', 0) + 1
    usage['input_tokens'] = usage.get('input_tokens', 0) + int(meta.get('input_tokens', 0) or 0)
    usage['output_tokens'] = usage.get('output_tokens', 0) + int(meta.get('output_tokens', 0) or 0)

