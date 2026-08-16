from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from scholar_agents import telemetry


def test_span_does_not_export_raw_exception_message(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("test"))

    with pytest.raises(RuntimeError, match="sensitive-source-url"), telemetry.span(
        "privacy-check"
    ):
        raise RuntimeError("sensitive-source-url")

    (finished_span,) = exporter.get_finished_spans()
    assert finished_span.attributes is not None
    assert finished_span.attributes["error.type"] == "RuntimeError"
    assert finished_span.events == ()
