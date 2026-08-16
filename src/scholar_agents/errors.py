"""Job 边界可理解的显式错误类型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any


class JobError(RuntimeError):
    """带有稳定重试语义的业务错误。"""

    retryable = True
    retry_after_seconds: float | None = None


class PermanentJobError(JobError):
    """配置、输入或实体缺失等重试无法恢复的错误。"""

    retryable = False


class JobDeadlineExceeded(JobError):
    """整条 job 超过 wall-clock deadline。"""


@dataclass(eq=False)
class ProviderError(JobError):
    """Provider adapter 归一化后的错误，不暴露 SDK 私有异常给 Worker。"""

    message: str
    provider: str
    status_code: int | None = None
    error_code: str | None = None
    retryable: bool = True
    retry_after_seconds: float | None = None

    def __str__(self) -> str:
        return self.message


_PERMANENT_PROVIDER_CODES = {
    "authentication_error",
    "billing_hard_limit_reached",
    "billing_not_active",
    "credit_balance_too_low",
    "insufficient_quota",
    "invalid_api_key",
    "model_not_found",
    "permission_denied",
}


def normalize_provider_error(provider: str, exc: BaseException) -> ProviderError:
    """从 OpenAI/Anthropic SDK 的结构化字段生成稳定错误语义。"""
    status = getattr(exc, "status_code", None)
    status_code = int(status) if isinstance(status, int) else None
    body = getattr(exc, "body", None)
    code = _error_code(body) or _string_attr(exc, "code")
    retryable = status_code is None or status_code in {408, 409, 425, 429} or status_code >= 500
    if status_code in {400, 401, 403, 404, 405, 422}:
        retryable = False
    if code and code.lower() in _PERMANENT_PROVIDER_CODES:
        retryable = False
    return ProviderError(
        message=str(exc),
        provider=provider,
        status_code=status_code,
        error_code=code,
        retryable=retryable,
        retry_after_seconds=_retry_after(exc),
    )


def _error_code(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    candidate: Any = body.get("error", body)
    if isinstance(candidate, dict):
        value = candidate.get("code") or candidate.get("type")
        return str(value) if value else None
    return None


def _string_attr(value: object, name: str) -> str | None:
    candidate = getattr(value, name, None)
    return str(candidate) if candidate else None


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(raw))
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
