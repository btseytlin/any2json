import json
import fastjsonschema
import pandas as pd
from faker import Faker
import random
from any2json.containers import FromOtherFormatSample, Sample
from any2json.data_engine.generators.base import SampleGenerator
from typing import Any, Callable, Dict, List, Literal, Union
from sqlalchemy.orm import Session

from any2json.schema_utils import to_supported_json_schema
from any2json.utils import logger


class PandasGenerator(SampleGenerator):
    def __init__(
        self,
        num_rows: int | None = None,
        num_cols: int | None = None,
        column_name_format: str | None = None,
        column_configs: Dict[str, Dict[str, Any]] | None = None,
        input_format: str | None = None,
        format_name: str | None = None,
    ):
        super().__init__()
        self.fake = Faker()
        self.column_name_format: str | None = column_name_format
        self.num_rows: int | None = num_rows
        self.num_cols: int | None = num_cols
        self.column_configs: Dict[str, Dict[str, Any]] | None = column_configs
        self.input_format: str | None = input_format
        self.format_name: str | None = format_name

    def setup(self):
        self.column_name_format = random.choice(
            [
                "numbered",
                "random_words",
                "random_words_with_numbers",
            ]
        )
        self.num_rows = random.randint(1, 5)
        self.num_cols = random.randint(2, 20)

        conversion_options = [
            "csv",
            "markdown",
            "string",
            "html",
            "json_split",
            "json_index",
            "json_columns",
            "json_table",
            "json_values",
        ]
        self.input_format = random.choice(conversion_options)
        if self.input_format in ("json_values", "json_split"):
            self.column_name_format = "numbered"
        self.format_name = self.input_format.split("_")[0]

        self.column_configs = self.get_random_column_configs()

    def get_state(self) -> Dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "column_name_format": self.column_name_format,
            "input_format": self.input_format,
            "format_name": self.format_name,
        }

    def generate_colname_words(self) -> str:
        num_words = random.randint(1, 5)
        sep = random.choice(["_", "-", " "])
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
            sep = random.choice(["_", "-", "", " "])
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
                "func": lambda *args: self.fake.pydecimal(
                    left_digits=random.randint(1, 3),
                    right_digits=random.randint(1, 3),
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

    def convert_dataframe_to_format(self, df: pd.DataFrame, format_choice: str) -> str:
        if format_choice == "csv":
            return df.to_csv(index=False)
        elif format_choice == "markdown":
            return df.to_markdown(index=False)
        elif format_choice == "string":
            return df.to_string(index=False)
        elif format_choice == "html":
            return df.to_html(index=False)
        elif format_choice.startswith("json_"):
            orient = format_choice.replace("json_", "")
            if orient in ("index", "columns", "split"):
                return df.to_json(orient=orient)
            return df.to_json(orient=orient, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format_choice}")

    def generate_triplet(
        self,
    ) -> tuple[str, dict, str]:
        df = self.generate_synthetic_dataframe()

        inferred_schema = self.infer_schema_from_dataframe(df)

        if inferred_schema["type"] == "array":
            json_output_data = df.to_json(orient="records")
        elif inferred_schema["type"] == "object":
            json_output_data = df.iloc[0].to_json()
        else:
            raise ValueError(f"Unsupported schema type: {inferred_schema['type']}")

        formatted_str = self.convert_dataframe_to_format(df, self.input_format)

        logger.debug(
            f"Generated data: {json.dumps(json.loads(json_output_data), indent=1)}"
        )
        logger.debug(f"Generated schema: {json.dumps(inferred_schema, indent=1)}")

        validate = fastjsonschema.compile(inferred_schema)
        validate(json.loads(json_output_data))

        inferred_schema = to_supported_json_schema(inferred_schema)

        logger.debug(f"Simplified schema: {json.dumps(inferred_schema, indent=1)}")

        return formatted_str, inferred_schema, json_output_data
