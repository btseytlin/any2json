import random

import fastjsonschema
from any2json.containers import FromOtherFormatSample, Sample
import yaml

from any2json.containers import Sample
from any2json.data_engine.generators.base import SampleGenerator


class FormatConversionSampleGenerator(SampleGenerator):
    def generate_samples(
        self,
        source_data: dict | list,
        source_schema: dict,
        num_samples: int = 1,
    ) -> list[Sample]:
        raise NotImplementedError


class ToYamlSampleGenerator(FormatConversionSampleGenerator):
    format = "yaml"

    def convert(self, data: dict | list) -> str:
        sort_keys = random.random() < 0.5
        indent = random.randint(0, 4)

        return yaml.dump(data, sort_keys=sort_keys, indent=indent)

    def generate_sample(
        self,
        input_data: dict | list,
        input_schema: dict,
    ) -> Sample:
        try:
            formatted_str = self.convert(input_data)
        except Exception as e:
            raise ValueError(f"Failed to convert to yaml: {e}") from e

        return FromOtherFormatSample(
            input_data=formatted_str,
            schema=input_schema,
            output=input_data,
            input_format=self.format,
            output_format="json",
            chunk_id=None,
            generator=self.__class__.__name__,
        )

    def validate_source_data(self, source_data: dict, source_schema: dict) -> dict:
        assert source_data is not None, "Source data must be provided"
        assert source_schema is not None, "Source schema must be provided"
        fastjsonschema.validate(source_schema, source_data)

    def generate_samples(
        self,
        source_data: dict | list,
        source_schema: dict,
        num_samples: int = 1,
    ) -> list[Sample]:
        self.validate_source_data(source_data, source_schema)

        return [
            self.generate_sample(source_data, source_schema) for _ in range(num_samples)
        ]


# class ToTomlGenerator(FromOtherFormatGenerator):
#     def to_toml(self, data: dict | list) -> str:
#         if isinstance(data, list):
#             data = {
#                 "root": data,
#             }
#         return toml.dumps(data)

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_toml(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to toml: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="toml",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToMarkdownTableGenerator(FromOtherFormatGenerator):
#     def to_markdown_table(self, data: dict | list) -> str:
#         if isinstance(data, list):
#             headers = "".join(f"| {key} |" for key in data[0].keys())
#             rows = ""
#             for row_dict in data:
#                 cells = "".join(f"| {val} |" for val in row_dict.values())
#                 rows += f"|{cells}|\n"
#             return f"|{headers}|\n|{rows}|"
#         else:
#             headers = "".join(f"| {key} |" for key in data.keys())
#             cells = "".join(f"| {val} |" for val in data.values())
#             return f"|{headers}|\n|{cells}|"

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_markdown_table(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to markdown_table: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="markdown_table",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToPlainTextGenerator(FromOtherFormatGenerator):
#     def to_plain_text(self, data: dict | list | Any, depth: int = 0) -> str:
#         result_string = ""
#         indent_char = random.choice([" ", "\t"])
#         indent_size = random.randint(2, 4) if indent_char == " " else 1
#         indent = indent_char * indent_size * depth
#         if isinstance(data, list):
#             for item in data:
#                 result_string += self.to_plain_text(item, depth + 1)
#             return result_string
#         elif isinstance(data, dict):
#             for key, value in data.items():
#                 result_string += (
#                     f"{indent}{key}: {self.to_plain_text(value, depth + 1)}\n"
#                 )
#         else:
#             result_string += f"{indent}{str(data)}\n"
#         return result_string

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_plain_text(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to plain_text: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="plain_text",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToPythonStringGenerator(FromOtherFormatGenerator):
#     def to_python_dict_string(self, data: dict | list) -> str:
#         return str(data)

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_python_dict_string(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to python_dict_string: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="python_dict_string",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToXmlGenerator(FromOtherFormatGenerator):
#     def to_xml(self, data: dict | list) -> str:
#         do_add_root_element = random.random() < 0.5

#         def dict_to_xml(parent_element: ET.Element, d: dict) -> None:
#             for key, val in d.items():
#                 element = ET.SubElement(parent_element, key)
#                 if isinstance(val, dict):
#                     dict_to_xml(element, val)
#                 elif isinstance(val, list):
#                     for item in val:
#                         item_element = ET.SubElement(element, "item")
#                         if isinstance(item, dict):
#                             dict_to_xml(item_element, item)
#                         else:
#                             item_element.text = str(item)
#                 else:
#                     element.text = str(val)

#         root_element_name = random.choice(
#             ["root", "document", "data", "item", "entry", "record"]
#         )

#         root = ET.Element(root_element_name)
#         dict_to_xml(root, data)
#         if do_add_root_element:
#             return f"<{root_element_name}>{ET.tostring(root, encoding='unicode')}</{root_element_name}>"
#         else:
#             return ET.tostring(root, encoding="unicode")

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_xml(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to xml: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="xml",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToHtmlTableGenerator(FromOtherFormatGenerator):
#     def to_html_table(self, data: dict | list) -> str:
#         add_table_tag = random.random() < 0.5

#         if isinstance(data, list):
#             headers = "".join(f"<th>{key}</th>" for key in data[0].keys())
#             rows = ""
#             for row_dict in data:
#                 cells = "".join(f"<td>{val}</td>" for val in row_dict.values())
#                 rows += f"<tr>{cells}</tr>"
#             return (
#                 f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"
#             )
#         rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items())
#         if add_table_tag:
#             return f"<table><tbody>{rows}</tbody></table>"
#         else:
#             return rows

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_html_table(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to html_table: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="html_table",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToHtmlTreeGenerator(FromOtherFormatGenerator):
#     def to_html_tree(self, data: dict | list) -> str:
#         general_block_tags = [
#             "div",
#             "p",
#             "section",
#             "article",
#             "aside",
#             "main",
#             "nav",
#             "header",
#             "footer",
#             "figure",
#             "blockquote",
#         ]
#         list_container_tags = ["ul", "ol"]
#         list_item_tag = "li"
#         definition_list_tag = "dl"
#         definition_term_tag = "dt"
#         definition_description_tag = "dd"
#         inline_tags = [
#             "span",
#             "b",
#             "i",
#             "strong",
#             "em",
#             "mark",
#             "small",
#             "a",
#             "label",
#             "time",
#             "code",
#             "samp",
#             "kbd",
#             "cite",
#             "q",
#             "data",
#         ]
#         key_display_tags = [
#             "strong",
#             "b",
#             "label",
#             "span",
#             "h1",
#             "h2",
#             "h3",
#             "h4",
#             "h5",
#             "h6",
#             "figcaption",
#         ]

#         memo: dict[int, Any] = {}

#         def _format_primitive(val: Any) -> str:
#             tag = random.choice(inline_tags)
#             return f"<{tag}>{str(val)}</{tag}>"

#         def _format_list(
#             lst: list, depth: int, generate_recursive_func: Callable[[Any, int], str]
#         ) -> str:
#             if not lst:
#                 return ""

#             items_html: list[str] = []
#             list_style_choice = random.choice(["structured_list", "div_series"])

#             if list_style_choice == "structured_list":
#                 parent_tag = random.choice(list_container_tags)
#                 for sub_item in lst:
#                     item_content_html = generate_recursive_func(sub_item, depth + 1)
#                     items_html.append(
#                         f"<{list_item_tag}>{item_content_html}</{list_item_tag}>"
#                     )
#                 return f"<{parent_tag}>{''.join(items_html)}</{parent_tag}>"
#             else:  # div_series
#                 parent_tag = random.choice(general_block_tags)
#                 for sub_item in lst:
#                     item_content_html = generate_recursive_func(sub_item, depth + 1)
#                     item_wrapper_tag = random.choice(general_block_tags)
#                     items_html.append(
#                         f"<{item_wrapper_tag}>{item_content_html}</{item_wrapper_tag}>"
#                     )
#                 return f"<{parent_tag}>{''.join(items_html)}</{parent_tag}>"

#         def _format_dict(
#             dct: dict, depth: int, generate_recursive_func: Callable[[Any, int], str]
#         ) -> str:
#             if not dct:
#                 return ""

#             entries_html: list[str] = []
#             dict_style_choice = random.choice(["definition_list", "general_structure"])

#             if dict_style_choice == "definition_list":
#                 for key, value in dct.items():
#                     key_html = str(key)
#                     value_html_content = generate_recursive_func(value, depth + 1)
#                     entries_html.append(
#                         f"<{definition_term_tag}>{key_html}</{definition_term_tag}><{definition_description_tag}>{value_html_content}</{definition_description_tag}>"
#                     )
#                 return f"<{definition_list_tag}>{''.join(entries_html)}</{definition_list_tag}>"
#             else:  # general_structure
#                 parent_wrapper_tag = random.choice(general_block_tags)
#                 for key, value in dct.items():
#                     key_str = str(key)
#                     value_html_content = generate_recursive_func(value, depth + 1)

#                     key_tag_1 = random.choice(key_display_tags)
#                     html_style_1 = (
#                         f"<{key_tag_1}>{key_str}</{key_tag_1}> {value_html_content}"
#                     )

#                     wrapper_tag_2 = random.choice(general_block_tags)
#                     html_style_2 = f'<{wrapper_tag_2} data-key="{key_str}">{value_html_content}</{wrapper_tag_2}>'

#                     key_simple_tag_3 = random.choice(inline_tags)
#                     html_style_3 = f"<{key_simple_tag_3}>{key_str}</{key_simple_tag_3}> {value_html_content}"

#                     chosen_kv_representation = random.choice(
#                         [html_style_1, html_style_2, html_style_3]
#                     )

#                     pair_container_tag = random.choice(general_block_tags)
#                     entries_html.append(
#                         f"<{pair_container_tag}>{chosen_kv_representation}</{pair_container_tag}>"
#                     )
#                 return f"<{parent_wrapper_tag}>{''.join(entries_html)}</{parent_wrapper_tag}>"

#         def _generate_html_recursive_impl(item: Any, depth: int = 0) -> str:
#             # Check for circular references using id() for mutable types (dict, list)
#             if isinstance(item, (dict, list)):
#                 item_id = id(item)
#                 if item_id in memo:
#                     return f"<{random.choice(inline_tags)}>recursive_ref</{random.choice(inline_tags)}>"
#                 memo[item_id] = item

#             if depth > 10:
#                 return (
#                     f"<{random.choice(inline_tags)}>...</{random.choice(inline_tags)}>"
#                 )

#             result: str
#             if isinstance(item, dict):
#                 result = _format_dict(item, depth, _generate_html_recursive_impl)
#             elif isinstance(item, list):
#                 result = _format_list(item, depth, _generate_html_recursive_impl)
#             else:
#                 result = _format_primitive(item)

#             if isinstance(item, (dict, list)):
#                 # pyright incorrectly flags item_id as possibly unbound if item is not dict/list
#                 # but we only enter this block if it is.
#                 item_id_to_del = id(item)
#                 if item_id_to_del in memo:
#                     del memo[item_id_to_del]
#             return result

#         return _generate_html_recursive_impl(data)

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_html_tree(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to html_tree: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="html_tree",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )


# class ToCsvGenerator(FromOtherFormatGenerator):
#     def to_csv(self, data: dict | list) -> str:
#         separator = random.choice([",", "\t", ";", "|"])
#         do_write_header = random.random() < 0.5
#         output = io.StringIO()

#         if isinstance(data, list):
#             fieldnames = list(data[0].keys())
#         elif isinstance(data, dict):
#             fieldnames = list(data.keys())
#         else:
#             raise ValueError(
#                 "Input data must be a dictionary or a list of dictionaries for CSV conversion."
#             )

#         random.shuffle(fieldnames)
#         writer = csv.DictWriter(
#             output,
#             fieldnames=fieldnames,
#             delimiter=separator,
#         )
#         if do_write_header:
#             writer.writeheader()
#         writer.writerow(data)
#         return output.getvalue()

#     def generate_sample(
#         self,
#         input_data: dict | list,
#         input_schema: dict,
#     ) -> Sample:
#         try:
#             formatted_str = self.to_csv(input_data)
#         except Exception as e:
#             raise ValueError(f"Failed to convert to csv: {e}") from e

#         return FromOtherFormatSample(
#             input_data=formatted_str,
#             schema=input_schema,
#             output=input_data,
#             input_format="csv",
#             output_format="json",
#             chunk_id=None,
#             generator=self.__class__.__name__,
#         )
