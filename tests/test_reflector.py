from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from scholar_agents.agents.reflector import build_calibration, insight_status, pearson
from scholar_agents.db_access import PerformanceSampleRecord


def _sample(percentile: float, topic_score: float, article_score: float) -> PerformanceSampleRecord:
    return PerformanceSampleRecord(
        publication_id=uuid4(),
        article_id=uuid4(),
        platform="xiaohongshu",
        snapshot_window="h24",
        captured_at=datetime.now(UTC),
        performance_raw=100 + percentile,
        performance_percentile=percentile,
        article_title=f"Article P{percentile}",
        article_content="content",
        article_score=article_score,
        article_dimensions={"hook": article_score / 10},
        topic_id=uuid4(),
        topic_title="Topic",
        topic_angle="Angle",
        topic_score=topic_score,
        topic_dimensions={"timeliness": topic_score / 10},
        metrics={"views": 1000, "likes": 100},
    )


def test_pearson_is_deterministic_and_requires_variance() -> None:
    assert pearson([(1, 10), (2, 20), (3, 30)]) == 1.0
    assert pearson([(1, 30), (2, 20), (3, 10)]) == -1.0
    assert pearson([(1, 10), (1, 20), (1, 30)]) is None
    assert pearson([(1, 10), (2, 20)]) is None


def test_build_calibration_separates_high_and_low_cases() -> None:
    samples = [_sample(0, 40, 45), _sample(50, 65, 70), _sample(100, 90, 95)]

    report = build_calibration(samples)

    assert report["coldStart"] is True
    assert [case["band"] for case in report["highCases"]] == ["high"]
    assert [case["band"] for case in report["lowCases"]] == ["low"]
    correlations = {item["key"]: item for item in report["correlations"]}
    assert correlations["topic.total"]["sampleSize"] == 3
    assert correlations["topic.total"]["coefficient"] == 1.0


def test_insight_only_becomes_active_with_five_distinct_publications() -> None:
    publications = [uuid4() for _ in range(5)]
    evidence = [
        {"articleIds": [], "publicationIds": [str(publication)], "note": "evidence"}
        for publication in publications
    ]

    assert insight_status(evidence[:4], 0.9) == "candidate"
    assert insight_status(evidence, 0.64) == "candidate"
    assert insight_status(evidence, 0.65) == "active"


def test_calibration_marks_thirty_samples_out_of_cold_start() -> None:
    start = datetime.now(UTC) - timedelta(days=1)
    samples = [_sample(float(index), float(index), float(index)) for index in range(30)]
    assert start < samples[0].captured_at

    report = build_calibration(samples)

    assert report["coldStart"] is False
