"""OpenTelemetry 初始化、低基数指标和安全的手工埋点。"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram, UpDownCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

from scholar_agents.errors import ProviderError
from scholar_agents.job_context import JobContext, current_job

log = structlog.get_logger()
_tracer = trace.get_tracer("scholar-agents")
_meter = metrics.get_meter("scholar-agents")
_providers: tuple[TracerProvider, MeterProvider] | None = None

jobs_started: Counter
jobs_completed: Counter
jobs_failed: Counter
jobs_retried: Counter
jobs_dead_lettered: Counter
structured_retries: Counter
formatter_violations: Counter
items_inserted: Counter
duplicates: Counter
job_duration: Histogram
queue_wait: Histogram
llm_duration: Histogram
embedding_duration: Histogram
feed_fetch_duration: Histogram
page_fetch_duration: Histogram
worker_concurrency: UpDownCounter


def _create_instruments() -> None:
    global jobs_started, jobs_completed, jobs_failed, jobs_retried, jobs_dead_lettered
    global structured_retries, formatter_violations, items_inserted, duplicates, job_duration
    global queue_wait
    global llm_duration, embedding_duration, feed_fetch_duration, page_fetch_duration
    global worker_concurrency
    meter = metrics.get_meter("scholar-agents")
    jobs_started = meter.create_counter("scholar_agent_jobs_started_total")
    jobs_completed = meter.create_counter("scholar_agent_jobs_completed_total")
    jobs_failed = meter.create_counter("scholar_agent_jobs_failed_total")
    jobs_retried = meter.create_counter("scholar_agent_jobs_retried_total")
    jobs_dead_lettered = meter.create_counter("scholar_agent_jobs_dead_lettered_total")
    structured_retries = meter.create_counter("scholar_agent_structured_retries_total")
    formatter_violations = meter.create_counter("scholar_agent_formatter_violations_total")
    items_inserted = meter.create_counter("scholar_agent_items_inserted_total")
    duplicates = meter.create_counter("scholar_agent_duplicates_total")
    job_duration = meter.create_histogram("scholar_agent_job_duration_seconds")
    queue_wait = meter.create_histogram("scholar_agent_queue_wait_seconds")
    llm_duration = meter.create_histogram("scholar_agent_llm_duration_seconds")
    embedding_duration = meter.create_histogram("scholar_agent_embedding_duration_seconds")
    feed_fetch_duration = meter.create_histogram("scholar_agent_feed_fetch_duration_seconds")
    page_fetch_duration = meter.create_histogram("scholar_agent_page_fetch_duration_seconds")
    worker_concurrency = meter.create_up_down_counter("scholar_agent_worker_concurrency")


def init_telemetry() -> None:
    """没有 endpoint 时保持 no-op；初始化失败由调用方记录后继续业务。"""
    global _providers, _tracer
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        _create_instruments()
        return
    insecure = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    resource = Resource.create(
        {
            "service.name": "scholar-agents",
            "service.version": os.environ.get("SERVICE_VERSION", "dev"),
            "deployment.environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "local"),
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
    )
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=insecure), export_interval_millis=15000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    propagate.set_global_textmap(propagate.get_global_textmap())
    _providers = tracer_provider, meter_provider
    _tracer = trace.get_tracer("scholar-agents")
    _create_instruments()


def shutdown_telemetry() -> None:
    if _providers is None:
        return
    tracer_provider, meter_provider = _providers
    for provider in (meter_provider, tracer_provider):
        try:
            provider.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("telemetry_shutdown_failed", error=str(exc))


def extracted_parent(payload: dict[str, Any]) -> Any:
    raw = payload.get("_meta")
    if not isinstance(raw, dict):
        return None
    carrier = {
        key: str(raw[key]) for key in ("traceparent", "tracestate", "baggage") if raw.get(key)
    }
    return propagate.extract(carrier) if carrier else None


def _mark_error(current: Any, exc: BaseException) -> None:
    """只写稳定、安全的错误分类，禁止把可能含 URL/密钥的原始消息送入 Tempo。"""
    current.set_attribute("error.type", type(exc).__name__)
    if isinstance(exc, ProviderError):
        current.set_attribute("gen_ai.system", exc.provider)
        current.set_attribute("job.retryable", exc.retryable)
        if exc.status_code is not None:
            current.set_attribute("http.response.status_code", exc.status_code)
        if exc.error_code:
            current.set_attribute("gen_ai.error.code", exc.error_code)
    current.set_status(Status(StatusCode.ERROR))


@contextmanager
def job_span(context: JobContext, payload: dict[str, Any]) -> Iterator[Any]:
    parent = extracted_parent(payload)
    attrs: dict[str, Any] = {
        "messaging.system": "pgmq",
        "messaging.destination.name": context.queue,
        "messaging.message.id": context.msg_id,
        "messaging.message.receive_count": context.read_count,
        "job.id": str(context.job_id),
        "job.type": context.queue,
        "correlation.id": str(context.correlation_id),
        "job.attempt": context.read_count,
    }
    started = time.monotonic()
    jobs_started.add(1, {"queue": context.queue, "job_type": context.queue})
    queue_wait.record(
        max(0.0, time.time() - context.enqueued_at.timestamp()),
        {"queue": context.queue, "job_type": context.queue},
    )
    with _tracer.start_as_current_span(
        f"messaging.process {context.queue}",
        context=parent,
        kind=SpanKind.CONSUMER,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as exc:
            _mark_error(span, exc)
            raise
        finally:
            job_duration.record(
                time.monotonic() - started,
                {"queue": context.queue, "job_type": context.queue},
            )


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    context = current_job()
    inherited: dict[str, Any] = {}
    if context is not None:
        inherited = {
            "job.id": str(context.job_id),
            "job.type": context.queue,
            "correlation.id": str(context.correlation_id),
            "job.attempt": context.read_count,
        }
    with _tracer.start_as_current_span(
        name,
        attributes={**inherited, **attributes},
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except BaseException as exc:
            _mark_error(current, exc)
            raise


def record_job_outcome(context: JobContext, outcome: str, error_type: str | None = None) -> None:
    labels = {"queue": context.queue, "job_type": context.queue}
    if outcome == "completed":
        jobs_completed.add(1, {**labels, "status": "succeeded"})
    elif outcome == "retry":
        jobs_failed.add(1, {**labels, "error_type": error_type or "unknown"})
        jobs_retried.add(1, labels)
    elif outcome == "dead_letter":
        jobs_failed.add(1, {**labels, "error_type": error_type or "unknown"})
        jobs_dead_lettered.add(1, labels)


_create_instruments()
