import random
from typing import Any, Callable

import fastjsonschema
import toml
from any2json.containers import FromOtherFormatSample, Sample
import yaml

from any2json.containers import Sample
from any2json.data_engine.generators.base import SampleGenerator
from any2json.enums import ContentType


class Converter:
    format = None

    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        pass

    def get_state(self) -> dict:
        return {}

    def convert(self, data: dict | list) -> str:
        raise NotImplementedError


class ToYamlConverter(Converter):
    format = ContentType.YAML

    def __init__(self, sort_keys: bool | None = None, indent: int | None = None):
        self.sort_keys = sort_keys
        self.indent = indent

    def setup(self):
        self.sort_keys = random.random() < 0.5
        self.indent = random.randint(0, 4)

    def get_state(self) -> dict:
        return {
            "sort_keys": self.sort_keys,
            "indent": self.indent,
        }

    def convert(self, data: dict | list) -> str:
        yaml_str = yaml.dump(data, sort_keys=self.sort_keys, indent=self.indent)
        loaded_data = yaml.safe_load(yaml_str)
        assert loaded_data == data, "YAML conversion failed"
        return yaml_str


class ToTomlConverter(Converter):
    format = ContentType.TOML

    def convert(self, data: dict | list) -> str:
        if isinstance(data, list):
            data = {
                "root": data,
            }

        toml_str = toml.dumps(data)
        loaded_data = toml.loads(toml_str)

        if isinstance(data, dict):
            assert loaded_data == {
                k: v for k, v in data.items() if v is not None
            }, f"TOML conversion failed: {loaded_data} != {data}"
        elif isinstance(data, list):
            loaded_data = loaded_data["root"]
            for i, item in enumerate(data):
                assert (
                    loaded_data[i] == item
                ), f"TOML conversion failed: {loaded_data[i]} != {item}"
        else:
            assert loaded_data == data, "TOML conversion failed"

        return toml_str


class ToMarkdownTableConverter(Converter):
    format = ContentType.MARKDOWN

    def convert(self, data: dict | list) -> str:
        if isinstance(data, list):
            headers = "".join(f"| {key} |" for key in data[0].keys())
            rows = ""
            for row_dict in data:
                cells = "".join(f"| {val} |" for val in row_dict.values())
                rows += f"|{cells}|\n"
            return f"|{headers}|\n|{rows}|"
        else:
            headers = "".join(f"| {key} |" for key in data.keys())
            cells = "".join(f"| {val} |" for val in data.values())
            return f"|{headers}|\n|{cells}|"


class ToPythonStringConverter(Converter):
    format = ContentType.PYTHON_STRING

    def convert(self, data: dict | list) -> str:
        return str(data)


class ToHtmlTreeConverter(Converter):
    format = ContentType.HTML

    def convert(self, data: dict | list) -> str:
        general_block_tags = [
            "div",
            "p",
            "section",
            "article",
            "aside",
            "main",
            "nav",
            "header",
            "footer",
            "figure",
            "blockquote",
        ]
        list_container_tags = ["ul", "ol"]
        list_item_tag = "li"
        definition_list_tag = "dl"
        definition_term_tag = "dt"
        definition_description_tag = "dd"
        inline_tags = [
            "span",
            "b",
            "i",
            "strong",
            "em",
            "mark",
            "small",
            "a",
            "label",
            "time",
            "code",
            "samp",
            "kbd",
            "cite",
            "q",
            "data",
        ]
        key_display_tags = [
            "strong",
            "b",
            "label",
            "span",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "figcaption",
        ]

        memo: dict[int, Any] = {}

        def _format_primitive(val: Any) -> str:
            tag = random.choice(inline_tags)
            return f"<{tag}>{str(val)}</{tag}>"

        def _format_list(
            lst: list, depth: int, generate_recursive_func: Callable[[Any, int], str]
        ) -> str:
            if not lst:
                return ""

            items_html: list[str] = []
            list_style_choice = random.choice(["structured_list", "div_series"])

            if list_style_choice == "structured_list":
                parent_tag = random.choice(list_container_tags)
                for sub_item in lst:
                    item_content_html = generate_recursive_func(sub_item, depth + 1)
                    items_html.append(
                        f"<{list_item_tag}>{item_content_html}</{list_item_tag}>"
                    )
                return f"<{parent_tag}>{''.join(items_html)}</{parent_tag}>"
            else:  # div_series
                parent_tag = random.choice(general_block_tags)
                for sub_item in lst:
                    item_content_html = generate_recursive_func(sub_item, depth + 1)
                    item_wrapper_tag = random.choice(general_block_tags)
                    items_html.append(
                        f"<{item_wrapper_tag}>{item_content_html}</{item_wrapper_tag}>"
                    )
                return f"<{parent_tag}>{''.join(items_html)}</{parent_tag}>"

        def _format_dict(
            dct: dict, depth: int, generate_recursive_func: Callable[[Any, int], str]
        ) -> str:
            if not dct:
                return ""

            entries_html: list[str] = []
            dict_style_choice = random.choice(["definition_list", "general_structure"])

            if dict_style_choice == "definition_list":
                for key, value in dct.items():
                    key_html = str(key)
                    value_html_content = generate_recursive_func(value, depth + 1)
                    entries_html.append(
                        f"<{definition_term_tag}>{key_html}</{definition_term_tag}><{definition_description_tag}>{value_html_content}</{definition_description_tag}>"
                    )
                return f"<{definition_list_tag}>{''.join(entries_html)}</{definition_list_tag}>"
            else:  # general_structure
                parent_wrapper_tag = random.choice(general_block_tags)
                for key, value in dct.items():
                    key_str = str(key)
                    value_html_content = generate_recursive_func(value, depth + 1)

                    key_tag_1 = random.choice(key_display_tags)
                    html_style_1 = (
                        f"<{key_tag_1}>{key_str}</{key_tag_1}> {value_html_content}"
                    )

                    wrapper_tag_2 = random.choice(general_block_tags)
                    html_style_2 = f'<{wrapper_tag_2} data-key="{key_str}">{value_html_content}</{wrapper_tag_2}>'

                    key_simple_tag_3 = random.choice(inline_tags)
                    html_style_3 = f"<{key_simple_tag_3}>{key_str}</{key_simple_tag_3}> {value_html_content}"

                    chosen_kv_representation = random.choice(
                        [html_style_1, html_style_2, html_style_3]
                    )

                    pair_container_tag = random.choice(general_block_tags)
                    entries_html.append(
                        f"<{pair_container_tag}>{chosen_kv_representation}</{pair_container_tag}>"
                    )
                return f"<{parent_wrapper_tag}>{''.join(entries_html)}</{parent_wrapper_tag}>"

        def _generate_html_recursive_impl(item: Any, depth: int = 0) -> str:
            # Check for circular references using id() for mutable types (dict, list)
            if isinstance(item, (dict, list)):
                item_id = id(item)
                if item_id in memo:
                    return f"<{random.choice(inline_tags)}>recursive_ref</{random.choice(inline_tags)}>"
                memo[item_id] = item

            if depth > 10:
                return (
                    f"<{random.choice(inline_tags)}>...</{random.choice(inline_tags)}>"
                )

            result: str
            if isinstance(item, dict):
                result = _format_dict(item, depth, _generate_html_recursive_impl)
            elif isinstance(item, list):
                result = _format_list(item, depth, _generate_html_recursive_impl)
            else:
                result = _format_primitive(item)

            if isinstance(item, (dict, list)):
                # pyright incorrectly flags item_id as possibly unbound if item is not dict/list
                # but we only enter this block if it is.
                item_id_to_del = id(item)
                if item_id_to_del in memo:
                    del memo[item_id_to_del]
            return result

        return _generate_html_recursive_impl(data)
