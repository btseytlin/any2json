import random
import json
import re

import toml
import yaml
from bs4 import BeautifulSoup

from any2json.data_engine.utils import remove_none_kv
from any2json.enums import ContentType
from markdownify import markdownify as md
import markdown

from dicttoxml import dicttoxml

from any2json.utils import dictify_xml_string


class Converter:
    format = None

    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        pass

    def get_state(self) -> dict:
        return {}

    def _convert(self, data: dict | list) -> str:
        raise NotImplementedError

    def check_conversion(self, data: dict | list, converted_data: str) -> bool:
        loaded_data = self.load(converted_data)
        assert (
            loaded_data == data
        ), f"Conversion failed, expected: {data}, got: {loaded_data}"

    def convert(self, data: dict | list) -> str:
        assert data, "Empty data"
        converted_data = self._convert(data)
        self.check_conversion(data, converted_data)
        return converted_data

    def load(self, data: str) -> dict | list:
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

    def _convert(self, data: dict | list) -> str:
        yaml_str = yaml.dump(data, sort_keys=self.sort_keys, indent=self.indent)
        return yaml_str

    def load(self, data: str) -> dict | list:
        return yaml.safe_load(data)


class ToTomlConverter(Converter):
    format = ContentType.TOML

    def _convert(self, data: dict | list) -> str:
        if isinstance(data, list):
            data = {str(i): item for i, item in enumerate(data)}

        toml_str = toml.dumps(data)
        return toml_str

    def load(self, data: str) -> dict | list:
        loaded_data = toml.loads(data)
        all_keys_are_numbers = all(
            all(c.isdigit() for c in k) for k in loaded_data.keys()
        )
        if all_keys_are_numbers:
            loaded_data = [loaded_data[str(i)] for i in range(len(loaded_data))]
        return loaded_data

    def check_conversion(self, data: dict | list, converted_data: str) -> bool:
        loaded_data = self.load(converted_data)
        data_no_nulls = remove_none_kv(data)
        assert (
            loaded_data == data_no_nulls
        ), f"Conversion failed, expected: {data_no_nulls}, got: {loaded_data}"


class ToPythonStringConverter(Converter):
    format = ContentType.PYTHON_STRING

    def _convert(self, data: dict | list) -> str:
        string_data = str(data)
        return string_data

    def load(self, data: str) -> dict | list:
        return eval(data)


class ToHTMLTableConverter(Converter):
    format = ContentType.HTML

    def _convert(self, data: dict | list) -> str:
        if isinstance(data, list):
            return self._convert_list_of_dicts(data)
        return self._convert_dict(data)

    def _convert_list_of_dicts(self, data: list[dict]) -> str:
        soup = BeautifulSoup("", "html.parser")
        table = soup.new_tag("table")

        if not data:
            return str(table)

        keys = list(data[0].keys())

        thead = soup.new_tag("thead")
        header_row = soup.new_tag("tr")

        for key in keys:
            th = soup.new_tag("th")
            th.string = str(key)
            header_row.append(th)

        thead.append(header_row)
        table.append(thead)

        tbody = soup.new_tag("tbody")

        for item in data:
            row = soup.new_tag("tr")
            for key in keys:
                cell = soup.new_tag("td")
                cell.string = str(item[key])
                row.append(cell)
            tbody.append(row)

        table.append(tbody)
        return str(table)

    def _convert_dict(self, data: dict) -> str:
        soup = BeautifulSoup("", "html.parser")
        table = soup.new_tag("table")

        has_nested_dict = any(isinstance(v, dict) for v in data.values())

        if not has_nested_dict:
            thead = soup.new_tag("thead")
            header_row = soup.new_tag("tr")

            for key in data.keys():
                th = soup.new_tag("th")
                th.string = str(key)
                header_row.append(th)

            thead.append(header_row)
            table.append(thead)

            tbody = soup.new_tag("tbody")
            row = soup.new_tag("tr")

            for value in data.values():
                cell = soup.new_tag("td")
                cell.string = str(value)
                row.append(cell)

            tbody.append(row)
            table.append(tbody)
        else:
            has_simple_keys = any(not isinstance(v, dict) for v in data.values())

            if has_simple_keys:
                thead = soup.new_tag("thead")
                header_row = soup.new_tag("tr")

                for key in data.keys():
                    th = soup.new_tag("th")
                    th.string = str(key)
                    header_row.append(th)

                thead.append(header_row)
                table.append(thead)

            tbody = soup.new_tag("tbody")

            simple_values = {}
            nested_dicts = {}

            for key, value in data.items():
                if isinstance(value, dict):
                    nested_dicts[key] = value
                else:
                    simple_values[key] = value

            if simple_values and nested_dicts:
                for nested_key, nested_dict in nested_dicts.items():
                    nested_items = list(nested_dict.items())

                    for i, (nkey, nvalue) in enumerate(nested_items):
                        row = soup.new_tag("tr")

                        if i == 0:
                            for skey in data.keys():
                                if skey in simple_values:
                                    cell = soup.new_tag("td")
                                    cell.string = str(simple_values[skey])
                                    row.append(cell)
                                elif skey == nested_key:
                                    cell = soup.new_tag("td")
                                    cell.string = str(nkey)
                                    row.append(cell)
                                    cell = soup.new_tag("td")
                                    cell.string = str(nvalue)
                                    row.append(cell)
                                    break
                        else:
                            empty_cell = soup.new_tag("td")
                            empty_cell.string = ""
                            row.append(empty_cell)

                            cell = soup.new_tag("td")
                            cell.string = str(nkey)
                            row.append(cell)

                            cell = soup.new_tag("td")
                            cell.string = str(nvalue)
                            row.append(cell)

                        tbody.append(row)
            else:
                for key, value in data.items():
                    if isinstance(value, dict):
                        nested_items = list(value.items())
                        first_key, first_value = nested_items[0]

                        first_row = soup.new_tag("tr")
                        key_cell = soup.new_tag("td")
                        key_cell.string = str(key)
                        first_row.append(key_cell)

                        nested_key_cell = soup.new_tag("td")
                        nested_key_cell.string = str(first_key)
                        first_row.append(nested_key_cell)

                        nested_value_cell = soup.new_tag("td")
                        nested_value_cell.string = str(first_value)
                        first_row.append(nested_value_cell)

                        tbody.append(first_row)

                        for nested_key, nested_value in nested_items[1:]:
                            row = soup.new_tag("tr")

                            empty_cell = soup.new_tag("td")
                            empty_cell.string = ""
                            row.append(empty_cell)

                            nested_key_cell = soup.new_tag("td")
                            nested_key_cell.string = str(nested_key)
                            row.append(nested_key_cell)

                            nested_value_cell = soup.new_tag("td")
                            nested_value_cell.string = str(nested_value)
                            row.append(nested_value_cell)

                            tbody.append(row)

            table.append(tbody)

        table_str = str(table)
        return table_str

    def load(self, data: str) -> dict | list[dict]:
        soup = BeautifulSoup(data, "html.parser")
        table = soup.find("table")

        if not table:
            return {}

        thead = table.find("thead")
        tbody = table.find("tbody")

        if thead and tbody:
            header_row = thead.find("tr")
            headers = [th.get_text() for th in header_row.find_all("th")]
            body_rows = tbody.find_all("tr")

            if len(body_rows) == 1 and len(headers) == len(body_rows[0].find_all("td")):
                cells = body_rows[0].find_all("td")
                result = {}
                for i, header in enumerate(headers):
                    result[header] = self._parse_value(cells[i].get_text())
                return result
            elif len(body_rows) > 1 and any(
                len(row.find_all("td")) > len(headers) for row in body_rows
            ):
                return self._load_mixed_dict_with_thead(thead, tbody)
            else:
                return self._load_list_of_dicts_with_thead(thead, tbody)

        rows = table.find_all("tr")
        if not rows:
            return {}

        if len(rows) == 1:
            row = rows[0]
            cells = row.find_all("td")
            if len(cells) == 2:
                return {cells[0].get_text(): self._parse_value(cells[1].get_text())}

        return self._load_dict(rows)

    def _load_mixed_dict_with_thead(self, thead, tbody) -> dict:
        header_row = thead.find("tr")
        headers = [th.get_text() for th in header_row.find_all("th")]
        body_rows = tbody.find_all("tr")

        result = {}

        for row in body_rows:
            cells = row.find_all("td")

            if len(cells) == 3:
                first_cell = cells[0].get_text().strip()
                second_cell = cells[1].get_text().strip()
                third_cell = cells[2].get_text().strip()

                if first_cell:
                    for i, header in enumerate(headers):
                        if i == 0:
                            result[header] = self._parse_value(first_cell)
                        elif i == 1:
                            if header not in result:
                                result[header] = {}
                            result[header][second_cell] = self._parse_value(third_cell)
                            break
                else:
                    last_dict_key = None
                    for key, value in result.items():
                        if isinstance(value, dict):
                            last_dict_key = key
                            break
                    if last_dict_key:
                        result[last_dict_key][second_cell] = self._parse_value(
                            third_cell
                        )

        return result

    def _load_list_of_dicts_with_thead(self, thead, tbody) -> list[dict]:
        header_row = thead.find("tr")
        headers = [th.get_text() for th in header_row.find_all("th")]

        result = []
        for row in tbody.find_all("tr"):
            cells = row.find_all("td")
            item = {}
            for i, header in enumerate(headers):
                if i < len(cells):
                    item[header] = self._parse_value(cells[i].get_text())
            result.append(item)

        return result

    def _is_list_of_dicts_format(self, rows: list) -> bool:
        if len(rows) < 2:
            return False

        first_row_cells = len(rows[0].find_all("td"))
        return all(len(row.find_all("td")) == first_row_cells > 2 for row in rows)

    def _load_list_of_dicts(self, rows: list) -> list[dict]:
        keys = [cell.get_text() for cell in rows[0].find_all("td")[1:]]
        result = []

        for i, key in enumerate(keys):
            item = {}
            for row in rows:
                cells = row.find_all("td")
                field_name = cells[0].get_text()
                value = cells[i + 1].get_text()
                item[field_name] = self._parse_value(value)
            result.append(item)

        return result

    def _load_dict(self, rows: list) -> dict:
        result = {}

        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text()
                value = cells[1].get_text()
                result[key] = self._parse_value(value)
            elif len(cells) == 3:
                main_key = cells[0].get_text()
                nested_key = cells[1].get_text()
                nested_value = cells[2].get_text()

                if main_key:
                    if main_key not in result:
                        result[main_key] = {}
                    result[main_key][nested_key] = self._parse_value(nested_value)
                else:
                    last_main_key = list(result.keys())[-1]
                    result[last_main_key][nested_key] = self._parse_value(nested_value)

        return result

    def _parse_value(self, value: str):
        value = value.strip()
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


class ToMarkdownTableConverter(Converter):
    format = ContentType.MARKDOWN

    def __init__(self, markdownify_options: dict | None = None):
        self.markdownify_options = markdownify_options or {}
        self.html_converter = ToHTMLTableConverter()

    def _convert(self, data: dict | list) -> str:
        data_html = self.html_converter.convert(data)
        table_str = md(data_html)
        return table_str

    def load(self, data: str) -> dict | list[dict]:
        data_html = markdown.markdown(data, extensions=["tables"])
        return self.html_converter.load(data_html)


class ToCSVConverter(Converter):
    format = ContentType.CSV

    def __init__(self, sep: str = ","):
        self.sep = sep

    def _convert(self, data: dict | list) -> str:
        if isinstance(data, dict):
            flattened = self._flatten_dict(data)
            headers = list(flattened.keys())
            values = list(flattened.values())
            data_str = (
                self.sep.join(headers)
                + "\n"
                + self.sep.join(str(v) for v in values)
                + "\n"
            )
            return data_str

        elif isinstance(data, list):
            if not data:
                return ""

            flattened_rows = [self._flatten_dict(row) for row in data]
            headers = []
            for row in flattened_rows:
                for key in row.keys():
                    if key not in headers:
                        headers.append(key)

            rows = []
            rows.append(self.sep.join(headers))
            for row in flattened_rows:
                values = [str(row.get(header, "")) for header in headers]
                rows.append(self.sep.join(values))

            data_str = "\n".join(rows) + "\n"
            return data_str

    def _flatten_dict(self, data: dict, prefix: str = "") -> dict:
        result = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten_dict(value, full_key))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                assert False, "Nested lists of dicts not supported"
            else:
                result[full_key] = value
        return result

    def load(self, data: str) -> dict | list[dict]:
        if not data.strip():
            return []

        lines = data.strip().split("\n")
        if len(lines) < 2:
            return {}

        headers = lines[0].split(self.sep)
        rows = [line.split(self.sep) for line in lines[1:]]

        if len(rows) == 1:
            row_dict = {}
            for header, value in zip(headers, rows[0]):
                row_dict[header] = self._parse_value(value)
            return self._unflatten_dict(row_dict)
        else:
            result = []
            for row in rows:
                row_dict = {}
                for header, value in zip(headers, row):
                    if value == "":
                        continue
                    row_dict[header] = self._parse_value(value)
                result.append(self._unflatten_dict(row_dict))
            return result

    def _parse_value(self, value: str):
        if value == "True":
            return True
        elif value == "False":
            return False
        elif value == "None":
            return None
        elif value.isdigit():
            return int(value)
        elif self._is_float(value):
            return float(value)
        else:
            return value

    def _is_float(self, value: str) -> bool:
        try:
            float(value)
            return "." in value
        except ValueError:
            return False

    def _unflatten_dict(self, flattened: dict) -> dict:
        result = {}
        for key, value in flattened.items():
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result


class ToSQLInsertConverter(Converter):
    format = ContentType.SQL

    def _convert(self, data: dict | list) -> str:
        if isinstance(data, dict):
            data = [data]

        if not data:
            return ""

        keys = list(data[0].keys())
        if not keys:
            return ""

        values_list = []
        for item in data:
            values = []
            for key in keys:
                value = item[key]
                if isinstance(value, str):
                    values.append(f"'{value}'")
                elif isinstance(value, (dict, list)):
                    json_str = json.dumps(value, separators=(",", ":"))
                    values.append(f"'{json_str}'")
                elif value is None:
                    values.append("NULL")
                else:
                    values.append(str(value))
            values_list.append(f"({', '.join(values)})")

        columns = ", ".join(keys)
        values_clause = ", ".join(values_list)

        data_str = f"INSERT INTO users ({columns}) VALUES {values_clause};"
        return data_str

    def load(self, data: str) -> dict | list[dict]:
        if not data.strip():
            return []

        pattern = r"INSERT INTO users \(([^)]+)\) VALUES (.+);"
        match = re.match(pattern, data.strip())

        if not match:
            raise ValueError(f"Invalid SQL INSERT format: {data}")

        columns_str, values_str = match.groups()
        columns = [col.strip() for col in columns_str.split(",")]

        values_pattern = r"\(([^)]+)\)"
        value_groups = re.findall(values_pattern, values_str)

        result = []
        for value_group in value_groups:
            values = self._parse_values(value_group)
            row = {}
            for i, col in enumerate(columns):
                row[col] = values[i]
            result.append(row)

        return result[0] if len(result) == 1 else result

    def _parse_values(self, values_str: str) -> list:
        values = []
        current_value = ""
        in_quotes = False

        i = 0
        while i < len(values_str):
            char = values_str[i]

            if char == "'" and not in_quotes:
                in_quotes = True
                current_value += char
            elif char == "'" and in_quotes:
                in_quotes = False
                current_value += char
            elif char == "," and not in_quotes:
                values.append(self._convert_value(current_value.strip()))
                current_value = ""
            else:
                current_value += char

            i += 1

        if current_value.strip():
            values.append(self._convert_value(current_value.strip()))

        return values

    def _convert_value(self, value_str: str) -> any:
        value_str = value_str.strip()

        if value_str == "NULL":
            return None
        elif value_str.startswith("'") and value_str.endswith("'"):
            string_value = value_str[1:-1]
            if string_value.startswith("{") or string_value.startswith("["):
                try:
                    return json.loads(string_value)
                except json.JSONDecodeError:
                    return string_value
            return string_value
        elif value_str.isdigit() or (
            value_str.startswith("-") and value_str[1:].isdigit()
        ):
            return int(value_str)
        elif "." in value_str:
            try:
                return float(value_str)
            except ValueError:
                return value_str
        elif value_str.lower() == "true":
            return True
        elif value_str.lower() == "false":
            return False
        else:
            return value_str


class ToXMLConverter(Converter):
    format = ContentType.XML

    def postprocess_parsed_data(self, parsed_data: dict | list) -> dict | list:
        if isinstance(parsed_data, dict) and "@type" in parsed_data:
            if "#text" in parsed_data and parsed_data["@type"] in [
                "str",
                "int",
                "float",
                "bool",
            ]:
                python_type = eval(parsed_data["@type"])
                return python_type(parsed_data["#text"])
            elif parsed_data["@type"] == "null":
                return None
            elif parsed_data["@type"] in ["dict", "list"]:
                pass
            else:
                raise ValueError(f"Can't parse type: {parsed_data}")

        if isinstance(parsed_data, list):
            return [self.postprocess_parsed_data(item) for item in parsed_data]

        if isinstance(parsed_data, dict):
            for key, value in parsed_data.items():
                parsed_data[key] = self.postprocess_parsed_data(value)

        return parsed_data

    def _convert(self, data: dict | list) -> str:
        converted_data = dicttoxml(
            data,
            return_bytes=False,
            attr_type=True,
            root=True,
            xml_declaration=False,
        )
        return converted_data

    def load(self, data: str) -> dict | list:
        return dictify_xml_string(data)

    def check_conversion(self, data: dict | list, converted_data: str) -> bool:
        """Ensure all original values and keys are present in the converted data"""
        if isinstance(data, list):
            for item in data:
                self.check_conversion(item, converted_data)
            return

        for k, v in data.items():
            assert (
                k in converted_data
            ), f"Conversion failed, expected to find key: {k} in {converted_data}"
            if isinstance(v, bool):
                assert (
                    str(v).lower() in converted_data
                ), f"Conversion failed, expected to find value: {v} in {converted_data}"
            elif isinstance(v, type(None)):
                assert (
                    "null" in converted_data
                ), f"Conversion failed, expected to find value: {v} in {converted_data}"
            elif isinstance(v, (str, int, float)):
                assert (
                    str(v) in converted_data
                ), f"Conversion failed, expected to find value: {v} in {converted_data}"
