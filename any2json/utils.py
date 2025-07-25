from copy import deepcopy
import json
import logging
import re
import time

import httpx

logger = logging.getLogger("any2json")


def configure_loggers(level: str = "WARNING", basic_level: str = "WARNING"):
    global logger
    logging.basicConfig(level=basic_level, force=True)
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


def extract_json_from_markdown(markdown_text: str, max_checks: int = 10) -> list[dict]:
    json_chunks = []
    pattern = r"```json\s*([^(```)]+)\s*```"
    markdown_text = markdown_text.strip().replace("“", '"').replace("”", '"')
    i = 0
    for match in re.finditer(pattern, markdown_text):
        json_str = match.group(1).strip()
        json_str = remove_comments(json_str)
        logger.debug(f"{json_str=}")

        try:
            data = json.loads(json_str)
            json_chunks.append(data)
        except json.JSONDecodeError:
            continue

        i += 1
        if i >= max_checks:
            break

    return json_chunks
