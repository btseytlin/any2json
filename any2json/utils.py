from copy import deepcopy
import json
import logging
import re
import time
from typing import Any

from bs4 import BeautifulSoup
import httpx
import xml
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET
import toml
import yaml

from copy import copy
from dicttoxml import dicttoxml

from any2json.enums import ContentType

logger = logging.getLogger("any2json")


def json_dump_safe(*args, **kwargs) -> None:
    return json.dump(
        *args,
        **kwargs,
        default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
    )


def json_dumps_minified(*args, **kwargs) -> None:
    return json.dumps(
        *args,
        **kwargs,
        separators=(",", ":"),
        indent=None,
    )


def try_json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as e:
        return text


def try_minify_json_string(text: str) -> Any:
    try:
        return json_dumps_minified(json.loads(text))
    except Exception as e:
        logger.warning(f"Error minifying JSON: {e}\ntext: {text}")
        return text


def configure_loggers(level: str = "WARNING", basic_level: str = "WARNING"):
    global logger
    logging.basicConfig(level=basic_level, force=True)

    bm25s_logger = logging.getLogger("bm25s")
    bm25s_logger.setLevel(basic_level)
    bm25s_logger.propagate = True

    any2json_logger = logging.getLogger("any2json")
    any2json_logger.setLevel(level)
    any2json_logger.propagate = True

    logger = any2json_logger
    logger.info(f"Configured any2json logger with level {level}")
    return logger


def post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict,
    timeout: int = 10,
    max_retries: int = 3,
) -> dict:
    for i in range(max_retries):
        response = client.post(url, json=payload, timeout=timeout)

        if response.status_code == 429:
            time.sleep(i**2)
            continue

        response.raise_for_status()
        return response

    raise Exception(f"Failed to POST after {max_retries} retries")


def remove_comments(text: str) -> str:
    """
    Remove comments from the text. That is find all text between // and a newline, remove it.
    """
    return re.sub(r"//.*?\n", "", text, flags=re.DOTALL)


def parse_string(source_str: str, format: str) -> Any:
    source_str = source_str.strip()
    match format:
        case "json":
            return json.loads(source_str)
        case "xml":
            return xml.etree.ElementTree.fromstring(source_str)
        case "csv":
            return pd.read_csv(StringIO(source_str))
        case "html":
            assert (
                source_str[0] == "<" and source_str[-1] == ">"
            ), "HTML string must start and end with < and >"
            return BeautifulSoup(source_str, "html.parser")
        case "yaml":
            return yaml.full_load(source_str)
        case "toml":
            return toml.loads(source_str)
        case _:
            raise ValueError(f"Unsupported format: {format}")


def stringify_content(content: Any, format: ContentType | str) -> str:
    if isinstance(format, str):
        format = ContentType(format)
    match format:
        case ContentType.JSON:
            return json.dumps(content, indent=1, ensure_ascii=False)
        case ContentType.XML:
            return xml.etree.ElementTree.tostring(content, encoding="utf-8").decode(
                "utf-8"
            )
        case ContentType.CSV:
            return content.to_csv(index=False)
        case ContentType.HTML:
            # beautifulsoup to string
            return str(content)
        case ContentType.YAML:
            return yaml.dump(content)
        case ContentType.TOML:
            return toml.dumps(content)
        case ContentType.TEXT:
            return content
        case ContentType.MARKDOWN:
            return content
        case ContentType.LATEX:
            return content
        case ContentType.SQL:
            return content
        case ContentType.PYTHON_STRING:
            return content
        case ContentType.OTHER:
            return content
        case ContentType.STRING:
            return content
        case _:
            raise ValueError(f"Unsupported format: {format}")


def extract_from_markdown(
    markdown_text: str, max_checks: int = 10, format: str = "json"
) -> list[dict]:
    chunks = []
    pattern = f"```{format}" + r"\s*([^(```)]+)\s*```"
    markdown_text = markdown_text.strip().replace("“", '"').replace("”", '"')
    i = 0
    for match in re.finditer(pattern, markdown_text):
        source_str = match.group(1).strip()
        source_str = remove_comments(source_str)
        logger.debug(f"{source_str=}")

        try:
            data = parse_string(source_str, format)
        except Exception as e:
            logger.warning(f"Error parsing {format} string: {e}")
            continue

        chunks.append(data)

        i += 1
        if i >= max_checks:
            break

    return chunks


def dictify_xml(r, root=True):
    if root:
        return {r.tag: dictify_xml(r, False)}
    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify_xml(x, False))
    return d


def dictify_xml_string(xml_string: str) -> dict:
    root = ET.fromstring(xml_string)
    return dictify_xml(root)
