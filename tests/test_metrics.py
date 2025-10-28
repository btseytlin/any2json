import pytest
from any2json.benchmarks.benchmark import calculate_diff_metrics


class TestCalculateDiffMetrics:
    def test_identical_dicts(self):
        answer = {"a": 1, "b": 2}
        correct = {"a": 1, "b": 2}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 0
        assert result["diff_size_chars_added"] == 0
        assert result["diff_size_chars_missing"] == 0
        assert round(result["levenstein_similarity"], 4) == 1.0

    def test_added_field(self):
        answer = {"a": 1, "b": 2, "c": 3}
        correct = {"a": 1, "b": 2}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 2
        assert result["diff_size_chars_added"] == 6
        assert result["diff_size_chars_missing"] == 0
        assert round(result["levenstein_similarity"], 4) == 0.8125

    def test_removed_field(self):
        answer = {"a": 1}
        correct = {"a": 1, "b": 2}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 2
        assert result["diff_size_chars_added"] == 0
        assert result["diff_size_chars_missing"] == 6
        assert round(result["levenstein_similarity"], 4) == 0.7

    def test_modified_value(self):
        answer = {"a": 1, "b": 3}
        correct = {"a": 1, "b": 2}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 1
        assert result["diff_size_chars_added"] == 1
        assert result["diff_size_chars_missing"] == 1
        assert round(result["levenstein_similarity"], 4) == 0.9231

    def test_modified_key(self):
        answer = {"a": 1, "c": 2}
        correct = {"a": 1, "b": 2}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 1
        assert result["diff_size_chars_added"] == 1
        assert result["diff_size_chars_missing"] == 1
        assert round(result["levenstein_similarity"], 4) == 0.9231

    def test_empty_dicts(self):
        answer = {}
        correct = {}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 0
        assert result["diff_size_chars_added"] == 0
        assert result["diff_size_chars_missing"] == 0
        assert round(result["levenstein_similarity"], 4) == 1.0

    def test_nested_structure_modified_value(self):
        answer = {"a": {"nested": 1}}
        correct = {"a": {"nested": 2}}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 1
        assert result["diff_size_chars_added"] == 1
        assert result["diff_size_chars_missing"] == 1
        assert round(result["levenstein_similarity"], 4) == 0.9444

    def test_list_difference(self):
        answer = {"items": [1, 2, 3]}
        correct = {"items": [1, 2]}
        result = calculate_diff_metrics(answer, correct)

        assert result["diff_size_lines"] == 2
        assert result["diff_size_chars_added"] == 2
        assert result["diff_size_chars_missing"] == 0
        assert round(result["levenstein_similarity"], 4) == 0.9375
