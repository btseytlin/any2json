import json
import logging
import fastjsonschema
import pandas as pd
from faker import Faker
import random

import yaml
from any2json.containers import FromOtherFormatSample, Sample
from any2json.data_engine.generators.base import SampleGenerator
from typing import Any, Callable, Dict, List, Literal, Union
from sqlalchemy.orm import Session

from any2json.schema_utils import to_supported_json_schema
from any2json.utils import logger


import re


def df_to_xml(df: pd.DataFrame) -> str:
    attr_cols = list(df.columns)[: random.randint(0, len(df.columns) - 1)]
    elem_cols = [c for c in df.columns if c not in attr_cols]
    root_name = Faker().word()
    index = random.choice([True, False])
    xml_declaration = random.choice([True, False])
    return df.to_xml(
        attr_cols=attr_cols,
        elem_cols=elem_cols,
        root_name=root_name,
        index=index,
        xml_declaration=xml_declaration,
    )


def df_to_insert_sql(df: pd.DataFrame, dest_table: str | None = None) -> str:
    if dest_table is None:
        dest_table = Faker().word()

    insert = """
    INSERT INTO `{dest_table}` (
        """.format(
        dest_table=dest_table
    )

    columns_string = str(list(df.columns))[1:-1]
    columns_string = re.sub(r" ", "\n        ", columns_string)
    columns_string = re.sub(r"\'", "", columns_string)

    values_string = ""

    for row in df.itertuples(index=False, name=None):
        values_string += re.sub(r"nan", "null", str(row))
        values_string += ",\n"

    return insert + columns_string + ")\n     VALUES\n" + values_string[:-2] + ";"


def df_to_json(df: pd.DataFrame, orient: str = "records") -> str:
    if orient in ("index", "columns", "split"):
        return df.to_json(orient=orient)
    return df.to_json(orient=orient, index=False)


def df_to_python_string(df: pd.DataFrame, orient: str = "records") -> str:
    string = df_to_json(df, orient=orient)
    return str(json.loads(string))


def df_to_yaml(df: pd.DataFrame, orient: str = "records") -> str:
    if orient in ("index", "columns", "split"):
        obj = json.loads(df.to_json(orient=orient))
    else:
        obj = df.to_dict(orient=orient)

    text = yaml.dump(obj, default_flow_style=None)
    assert yaml.full_load(text) is not None
    return text


def df_from_yaml(text: str) -> pd.DataFrame:
    return pd.DataFrame(yaml.full_load(text))


class PandasGenerator(SampleGenerator):
    def __init__(
        self,
        num_rows: int | None = None,
        num_cols: int | None = None,
        column_name_format: str | None = None,
        column_configs: Dict[str, Dict[str, Any]] | None = None,
        input_format: str | None = None,
        input_orient: str | None = None,
    ):
        super().__init__()
        self.fake = Faker()
        self.column_name_format: str | None = column_name_format
        self.num_rows: int | None = num_rows
        self.num_cols: int | None = num_cols
        self.column_configs: Dict[str, Dict[str, Any]] | None = column_configs
        self.input_format: str | None = input_format
        self.input_orient: str | None = input_orient

    def setup(self):
        self.column_name_format = random.choice(
            [
                "numbered",
                "random_words",
                "random_words_with_numbers",
            ]
        )
        self.num_rows = random.randint(1, 5)
        self.num_cols = random.randint(2, 10)

        self.csv_sep = random.choice([",", "\t", ";", "|"])
        self.column_sep = random.choice(["_", "-", " ", ""])

        conversion_options = [
            "yaml",
            "sql",
            "python_string",
            "xml",
            "csv",
            "markdown",
            "string",
            "html",
            "latex",
            "json",
        ]
        self.input_format = random.choice(conversion_options)
        self.input_orient = None
        if self.input_format in ("yaml",):
            self.input_orient = random.choice(
                ["dict", "list", "split", "tight", "index", "columns"]
            )
        if self.input_format in ("json", "python_string"):
            self.input_orient = random.choice(["split", "index", "columns"])

        if self.input_orient in ("values", "split"):
            self.column_name_format = "numbered"

        if self.input_format in ("xml", "sql"):
            self.column_name_format = "random_words"
            self.column_sep = "_"

        self.column_configs = self.get_random_column_configs()

        logger.debug(f"Input format: {self.input_format}")
        logger.debug(f"Input orient: {self.input_orient}")
        logger.debug(f"Column name format: {self.column_name_format}")
        logger.debug(f"Column sep: {self.column_sep}")
        logger.debug(f"Column configs: {self.column_configs}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "column_name_format": self.column_name_format,
            "column_sep": self.column_sep,
            "input_format": self.input_format,
            "input_orient": self.input_orient,
            "csv_sep": self.csv_sep,
        }

    def generate_colname_words(self) -> str:
        num_words = random.randint(1, 5)
        sep = self.column_sep
        is_camel_case = random.choice([True, False])
        if is_camel_case:
            return "".join(word.capitalize() for word in self.fake.words(num_words))
        else:
            return sep.join(self.fake.words(num_words))

    def generate_column_name(self, i: int) -> str:
        if self.column_name_format == "numbered":
            return f"{i+1}"
        elif self.column_name_format == "random_words":
            return self.generate_colname_words()
        elif self.column_name_format == "random_words_with_numbers":
            sep = self.column_sep
            return f"{self.generate_colname_words()}{sep}{i+1}"
        else:
            raise ValueError(f"Invalid column name format: {self.column_name_format}")

    def get_random_column_configs(self) -> Dict[str, Dict[str, Any]]:
        type_options: List[Dict[str, Any]] = [
            {"func": self.fake.name, "json_type": "string"},
            {"func": self.fake.paragraph, "json_type": "string"},
            {"func": self.fake.sentence, "json_type": "string"},
            {"func": self.fake.address, "json_type": "string"},
            {
                "func": lambda *args: self.fake.random_int(min=-1000000, max=1000000),
                "json_type": "integer",
            },
            {
                "func": lambda *args: self.fake.random_int(min=-100, max=100),
                "json_type": "integer",
            },
            {
                "func": lambda *args: str(self.fake.random_int(min=-100, max=100)),
                "json_type": "string",
            },
            {
                "func": lambda *args: float(
                    self.fake.pydecimal(
                        left_digits=random.randint(1, 3),
                        right_digits=random.randint(1, 3),
                    )
                ),
                "json_type": "number",
            },
            {
                "func": lambda *args: str(
                    self.fake.pydecimal(
                        left_digits=random.randint(1, 3),
                        right_digits=random.randint(1, 3),
                    )
                ),
                "json_type": "string",
            },
            {
                "func": self.fake.date,
                "json_type": "string",
                "format": "date",
            },
            {
                "func": self.fake.date_this_decade().isoformat,
                "json_type": "string",
                "format": "date",
            },
            {"func": self.fake.email, "json_type": "string", "format": "email"},
            {"func": self.fake.phone_number, "json_type": "string", "format": "phone"},
            {"func": self.fake.url, "json_type": "string", "format": "url"},
            {"func": self.fake.ipv4, "json_type": "string", "format": "ipv4"},
            {"func": self.fake.ipv6, "json_type": "string", "format": "ipv6"},
            {"func": self.fake.uuid4, "json_type": "string", "format": "uuid"},
            {"func": self.fake.boolean, "json_type": "boolean"},
            {"func": self.fake.color_name, "json_type": "string", "format": "color"},
            {
                "func": self.fake.currency_code,
                "json_type": "string",
                "format": "currency",
            },
            {
                "func": self.fake.currency_name,
                "json_type": "string",
                "format": "currency",
            },
            {"func": lambda *args: None, "json_type": "string"},
            {"func": lambda *args: None, "json_type": "number"},
            {"func": lambda *args: "", "json_type": "string"},
        ]
        chosen_configs: Dict[str, Dict[str, Any]] = {}
        for i in range(self.num_cols):
            col_name = self.generate_column_name(i)
            chosen_configs[col_name] = random.choice(type_options)
        return chosen_configs

    def generate_synthetic_dataframe(self) -> pd.DataFrame:
        data = []
        for _ in range(self.num_rows):
            row: Dict[str, Any] = {}
            for col_name, config in self.column_configs.items():
                row[col_name] = config["func"]()
            data.append(row)
        return pd.DataFrame(data)

    def infer_schema_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        for col_name, config in self.column_configs.items():
            prop = {"type": [config["json_type"], "null"]}
            properties[col_name] = prop

        if len(df) > 1:
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                },
            }
        else:
            return {
                "type": "object",
                "properties": properties,
            }

    def convert_dataframe_to_format(
        self, df: pd.DataFrame, format_choice: str, orient: str | None = None
    ) -> str:
        if format_choice == "csv":
            return df.to_csv(index=False, sep=self.csv_sep)
        elif format_choice == "latex":
            return df.to_latex(index=False)
        elif format_choice == "xml":
            return df_to_xml(df)
        elif format_choice == "sql":
            return df_to_insert_sql(df)
        elif format_choice == "markdown":
            return df.to_markdown(index=False)
        elif format_choice == "string":
            return df.to_string(index=False)
        elif format_choice == "html":
            return df.to_html(index=False)
        elif format_choice == "json":
            return df_to_json(df, orient=orient)
        elif format_choice == "python_string":
            return df_to_python_string(df, orient=orient)
        elif format_choice == "yaml":
            return df_to_yaml(df, orient=orient)
        else:
            raise ValueError(f"Unsupported output format: {format_choice}")

    def generate_triplet(
        self,
    ) -> tuple[str, dict, str]:
        df = self.generate_synthetic_dataframe()

        inferred_schema = self.infer_schema_from_dataframe(df)
        inferred_schema = to_supported_json_schema(inferred_schema)
        validate = fastjsonschema.compile(inferred_schema)

        if inferred_schema["type"][0] == "array":
            json_output_data = df.to_json(orient="records")
        elif inferred_schema["type"][0] == "object":
            json_output_data = df.iloc[0].to_json()
        else:
            raise ValueError(f"Unsupported schema type: {inferred_schema['type']}")

        formatted_str = self.convert_dataframe_to_format(
            df, self.input_format, self.input_orient
        )

        logger.debug(
            f"Generated data: {json.dumps(json.loads(json_output_data), indent=1)}"
        )
        logger.debug(f"Generated schema: {json.dumps(inferred_schema, indent=1)}")

        validate(json.loads(json_output_data))

        return formatted_str, inferred_schema, json_output_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    generator = PandasGenerator()
    generator.setup()
    input_data, schema, output_data = generator.generate_triplet()
    print("Input data:")
    print(input_data)
    print("Schema:")
    print(schema)
    print("Output data:")
    print(output_data)
