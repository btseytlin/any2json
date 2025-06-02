import pandas as pd
from faker import Faker
import random
from any2json.containers import FromOtherFormatSample, Sample
from any2json.data_engine.generators.base import SampleGenerator
from typing import Any, Callable, Dict, List, Literal, Union


class PandasGenerator(SampleGenerator):
    def __init__(self):
        super().__init__()
        self.fake = Faker()
        self.num_rows: int
        self.num_cols: int
        self.column_configs: Dict[str, Dict[str, Any]]
        self.output_format_choice: str
        self.format_name: str = ""

    def setup(self):
        self.num_rows = random.randint(3, 10)
        self.num_cols = random.randint(2, 10)
        self.column_configs = self.get_random_column_configs()

        conversion_options = [
            "csv",
            "markdown",
            "string",
            "html",
            "json_records",
            "json_split",
            "json_index",
            "json_columns",
            "json_values",
            "json_table",
        ]
        self.output_format_choice = random.choice(conversion_options)
        self.format_name = f"pandas_dataframe_to_{self.output_format_choice}"

    def get_random_column_configs(self) -> Dict[str, Dict[str, Any]]:
        type_options: List[Dict[str, Any]] = [
            {"func": self.fake.name, "json_type": "string"},
            {"func": self.fake.address, "json_type": "string"},
            {"func": self.fake.random_int, "json_type": "integer"},
            {"func": self.fake.random_float, "json_type": "number"},
            {
                "func": self.fake.date_this_decade,
                "json_type": "string",
                "format": "date",
            },
            {"func": self.fake.email, "json_type": "string", "format": "email"},
        ]
        chosen_configs: Dict[str, Dict[str, Any]] = {}
        for i in range(self.num_cols):
            col_name = f"column_{i}"
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
            prop = {"type": config["json_type"]}
            if "format" in config:
                prop["format"] = config["format"]
            properties[col_name] = prop

        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": [],
            },
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
            if orient == "index":
                return df.to_json(orient=orient)
            return df.to_json(orient=orient, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format_choice}")

    def generate_sample(
        self,
    ) -> Sample:
        df = self.generate_synthetic_dataframe()

        json_output_data = df.to_dict(orient="records")
        inferred_schema = self.infer_schema_from_dataframe(df)

        formatted_str = self.convert_dataframe_to_format(df, self.output_format_choice)

        return FromOtherFormatSample(
            input_data=formatted_str,
            schema=inferred_schema,
            output=json_output_data,
            input_format=self.format_name,
            output_format="json",
            chunk_id=None,
            generator=self.__class__.__name__,
        )
