import pytest

from any2json.utils import extract_from_markdown


class TestExtractJsonFromMarkdown:
    def test_no_json(self):
        markdown_text = (
            "Some text before\n```html\n<p>Some text inside</p>\n```\nSome text after"
        )
        expected_json = []
        assert extract_from_markdown(markdown_text) == expected_json

    def test_extract_single_json(self):
        markdown_text = 'Some text before\n```json\n{"name": "John", "age": 30}\n```\nSome text after'
        expected_json = [{"name": "John", "age": 30}]
        assert extract_from_markdown(markdown_text) == expected_json

    def test_broken_code_block(self):
        markdown_text = 'Some text before\n```json\n{"name": "John", "age": 30}\n'
        expected_json = []
        assert extract_from_markdown(markdown_text) == expected_json

    def test_extract_multiple_json(self):
        blocks = [
            "Some text before",
            '```json\n{"name": "John", "age": 30}\n```',
            "Some text after",
            '```json\n{"name": "Jane", "age": 25}\n```',
            "Some text after",
        ]
        markdown_text = "\n".join(blocks)
        expected_json = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        assert extract_from_markdown(markdown_text) == expected_json

    def test_invalid_json(self):
        blocks = [
            "Some text before",
            '```json\n{"name": "John", "age": 30}\n```',
            "Some text after",
            '```json\n{"name": "Jane", "age": 25, \'invalid\': no}\n```',
            "Some text after",
        ]
        markdown_text = "\n".join(blocks)
        expected_json = [{"name": "John", "age": 30}]
        assert extract_from_markdown(markdown_text) == expected_json

    def test_real_world_example(self):
        text = """

If you used another OSM extract, use appropriate coordinates.

Assess the response


Response JSON


```json
{ "trip": { "language": "en-US", "status": 0, "units": "miles", "status_message": "Found route between points", "legs": [ { "shape": "yx{xmA_lybd@oClBqWxRqWhRsFlEeKlHaChBiGbFqGtEkWxRyQbN", "summary": { "max_lon": 19.461329, "max_lat": 41.321014, "time": 28, "length": 0.178, "min_lat": 41.318813, "min_lon": 19.45956 }, "maneuvers": [ { "travel_mode": "drive", "begin_shape_index": 0, "length": 0.154, "time": 24, "type": 1, "end_shape_index": 9, "instruction": "Drive northwest on Rruga Stefan Kaçulini.", "verbal_pre_transition_instruction": "Drive northwest on Rruga Stefan Kaçulini for 2 tenths of a mile.", "travel_type": "car", "street_names": [ "Rruga Stefan Kaçulini" ] }, { "travel_type": "car", "travel_mode": "drive", "verbal_pre_transition_instruction": "Continue on Rruga Glaukia for 100 feet. Then You will arrive at your destination.", "verbal_transition_alert_instruction": "Continue on Rruga Glaukia.", "length": 0.024, "instruction": "Continue on Rruga Glaukia.", "end_shape_index": 10, "type": 8, "time": 4, "verbal_multi_cue": true, "street_names": [ "Rruga Glaukia" ], "begin_shape_index": 9 }, { "travel_type": "car", "travel_mode": "drive", "begin_shape_index": 10, "time": 0, "type": 4, "end_shape_index": 10, "instruction": "You have arrived at your destination.", "length": 0, "verbal_transition_alert_instruction": "You will arrive at your destination.", "verbal_pre_transition_instruction": "You have arrived at your destination." } ] } ], "summary": { "max_lon": 19.461329, "max_lat": 41.321014, "time": 28, "length": 0.178, "min_lat": 41.318813, "min_lon": 19.45956 }, "locations": [ { "original_index": 0, "lon": 19.461336, "lat": 41.318817, "type": "break" }, { "original_index": 1, "lon": 19.459599, "lat": 41.320999, "type": "break" } ] } } ```

In case you get a response looking like this:

"error_code": 171,
"error": "No suitable edges near location",
"status_code": 400,
"status": "Bad Request"
"""
        expected_json = [
            {
                "trip": {
                    "language": "en-US",
                    "status": 0,
                    "units": "miles",
                    "status_message": "Found route between points",
                    "legs": [
                        {
                            "shape": "yx{xmA_lybd@oClBqWxRqWhRsFlEeKlHaChBiGbFqGtEkWxRyQbN",
                            "summary": {
                                "max_lon": 19.461329,
                                "max_lat": 41.321014,
                                "time": 28,
                                "length": 0.178,
                                "min_lat": 41.318813,
                                "min_lon": 19.45956,
                            },
                            "maneuvers": [
                                {
                                    "travel_mode": "drive",
                                    "begin_shape_index": 0,
                                    "length": 0.154,
                                    "time": 24,
                                    "type": 1,
                                    "end_shape_index": 9,
                                    "instruction": "Drive northwest on Rruga Stefan Kaçulini.",
                                    "verbal_pre_transition_instruction": "Drive northwest on Rruga Stefan Kaçulini for 2 tenths of a mile.",
                                    "travel_type": "car",
                                    "street_names": ["Rruga Stefan Kaçulini"],
                                },
                                {
                                    "travel_type": "car",
                                    "travel_mode": "drive",
                                    "verbal_pre_transition_instruction": "Continue on Rruga Glaukia for 100 feet. Then You will arrive at your destination.",
                                    "verbal_transition_alert_instruction": "Continue on Rruga Glaukia.",
                                    "length": 0.024,
                                    "instruction": "Continue on Rruga Glaukia.",
                                    "end_shape_index": 10,
                                    "type": 8,
                                    "time": 4,
                                    "verbal_multi_cue": True,
                                    "street_names": ["Rruga Glaukia"],
                                    "begin_shape_index": 9,
                                },
                                {
                                    "travel_type": "car",
                                    "travel_mode": "drive",
                                    "begin_shape_index": 10,
                                    "time": 0,
                                    "type": 4,
                                    "end_shape_index": 10,
                                    "instruction": "You have arrived at your destination.",
                                    "length": 0,
                                    "verbal_transition_alert_instruction": "You will arrive at your destination.",
                                    "verbal_pre_transition_instruction": "You have arrived at your destination.",
                                },
                            ],
                        }
                    ],
                    "summary": {
                        "max_lon": 19.461329,
                        "max_lat": 41.321014,
                        "time": 28,
                        "length": 0.178,
                        "min_lat": 41.318813,
                        "min_lon": 19.45956,
                    },
                    "locations": [
                        {
                            "original_index": 0,
                            "lon": 19.461336,
                            "lat": 41.318817,
                            "type": "break",
                        },
                        {
                            "original_index": 1,
                            "lon": 19.459599,
                            "lat": 41.320999,
                            "type": "break",
                        },
                    ],
                }
            }
        ]
        assert extract_from_markdown(text) == expected_json

    def test_json_with_comments(self):
        text = """
```json
{"devDependencies": {
// import published v1.0 types with a version from NPM
// import beta types with a version from NPM
"@microsoft/microsoft-graph-types-beta": "^0.1.0-preview"
} } ``` ```

        """
        expected_json = [
            {
                "devDependencies": {
                    "@microsoft/microsoft-graph-types-beta": "^0.1.0-preview"
                }
            }
        ]
        assert extract_from_markdown(text) == expected_json
