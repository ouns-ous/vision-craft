"""
Unit tests for VisionBenchmark module.
"""

from pathlib import Path
import pytest
import json

from vision_craft.benchmark import VisionBenchmark, BenchmarkScore


def test_benchmark_runner():
    scores = VisionBenchmark.run_benchmark(
        resolution="480p",
        iterations=3,
        target_ops=["Grayscale", "Canny"],
    )
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, BenchmarkScore)
        assert s.mean_ms > 0
        assert s.fps > 0
        assert s.megapixels_per_sec > 0


def test_benchmark_markdown_table():
    scores = VisionBenchmark.run_benchmark(
        resolution="480p",
        iterations=2,
        target_ops=["Grayscale"],
    )
    table = VisionBenchmark.format_markdown_table(scores)
    assert "| Operation |" in table
    assert "Grayscale" in table
