from enum import Enum


class ContentType(Enum):
    JSON = "JSON"
    CSV = "CSV"
    MARKDOWN = "MARKDOWN"
    STRING = "STRING"
    XML = "XML"
    HTML = "HTML"
    TEXT = "TEXT"
    OTHER = "OTHER"
    YAML = "YAML"
    TOML = "TOML"
    PYTHON_STRING = "PYTHON_STRING"
    SQL = "SQL"
    LATEX = "LATEX"
