from datetime import datetime
import asyncio
import json
import random
from any2json.utils import extract_json_from_markdown, logger
import httpx
import time
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


def process_document(doc_data: dict, rank: int) -> dict | None:
    document_content = "\n".join(span[0] for span in doc_data["spans"])

    # logger.debug(f"Extracting JSON chunks from document:\n{document_content}")
    json_chunks = extract_json_from_markdown(document_content)

    logger.debug(f"Extracted {len(json_chunks)} JSON chunks from document {rank}")

    if not json_chunks:
        return None

    document_meta = {
        "doc_ix": doc_data.get("doc_ix"),
        "doc_len": doc_data.get("doc_len"),
        "disp_len": doc_data.get("disp_len"),
        "rank": rank,
    }
    return {
        "document": document_content,
        "json_chunks": json_chunks,
        "document_meta": document_meta,
    }


class InfiniGramAPI:
    api_url = "https://api.infini-gram.io/"

    def __init__(
        self,
        index: str = "v4_olmo-2-1124-13b-instruct_llama",
        max_clause_freq: int = 500000,
        max_diff_tokens: int = 1000,
        timeout: int = 5,
        max_retries: int = 3,
        max_concurrent_requests: int = 25,
    ):
        self.index = index
        self.max_clause_freq = max_clause_freq
        self.max_diff_tokens = max_diff_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests

    async def _query_async(
        self,
        client: httpx.AsyncClient,
        payload: dict,
    ) -> int | Exception:
        logger.debug(f"Querying Infinigram API with payload: {payload}")

        start_time = time.time()
        last_error = None

        for retry in range(self.max_retries):
            try:
                response = await client.post(
                    self.api_url,
                    json=payload,
                )
                response.raise_for_status()
                # logger.debug(f"Infinigram API response: {response.json()}")
                logger.debug(
                    f"Infinigram API query took {round(time.time() - start_time, 2)} seconds, {retry} retries"
                )
                return response
            except Exception as e:
                logger.warning(f"Infinigram API error {type(e)}: {e}")
                last_error = e

                sleep_time = (1 + retry + random.random()) ** 2
                if isinstance(e, httpx.HTTPStatusError):
                    if retry_after := e.response.headers.get("Retry-After"):
                        try:
                            sleep_time = int(retry_after)
                        except ValueError:
                            logger.warning(
                                f"Could not parse Retry-After header: {retry_after}"
                            )

                await asyncio.sleep(sleep_time)

        logger.error(
            f"Infinigram API query failed after {self.max_retries} retries in {round(time.time() - start_time, 2)} seconds"
        )
        if last_error:
            return last_error
        return Exception("Infinigram API query failed without a specific exception")

    async def batch_query_async(
        self,
        payloads: list[dict],
    ) -> list[httpx.Response | Exception]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def query_with_semaphore(
            client, payload: dict
        ) -> httpx.Response | Exception:
            async with semaphore:
                return await self._query_async(client, payload)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_concurrent_requests,
                max_keepalive_connections=self.max_concurrent_requests,
            ),
            verify=False,
        ) as client:
            tasks = [
                asyncio.create_task(query_with_semaphore(client, payload))
                for payload in payloads
            ]
            return await tqdm_asyncio.gather(*tasks)

    async def find_documents(
        self,
        query: str,
    ) -> dict:
        payload = {"index": self.index, "query_type": "find", "query": query}
        response = await self.batch_query_async([payload])
        return response[0].json()

    def fetch_and_process_documents(
        self,
        num_chunks: int,
        segment_by_shard: list[tuple[int, int]],
    ) -> list[dict]:
        results = []
        collected_chunks = 0

        query_payloads = []
        for shard_index, (shard_start, shard_end) in enumerate(segment_by_shard):
            for rank in range(shard_start, shard_end):
                payload = {
                    "index": self.index,
                    "query_type": "get_doc_by_rank",
                    "query": "```json",
                    "rank": rank,
                    "max_disp_len": 10000,
                    "s": shard_index,
                }
                query_payloads.append(payload)

        query_payloads = query_payloads[:num_chunks]

        doc_datas = asyncio.run(self.batch_query_async(query_payloads))
        doc_datas = [
            response.json()
            for response in doc_datas
            if not isinstance(response, Exception)
        ]

        seen_documents = set()
        seen_chunks = set()

        for payload, doc_data in zip(query_payloads, doc_datas):
            rank = payload["rank"]
            doc_id = json.loads(doc_data.get("metadata")).get("metadata", {}).get("id")
            if doc_id in seen_documents:
                continue
            seen_documents.add(doc_id)

            processed_doc = process_document(doc_data, rank)
            if processed_doc:
                for chunk in processed_doc["json_chunks"]:
                    chunk_str = json.dumps(chunk, ensure_ascii=False)
                    if chunk_str in seen_chunks:
                        continue  # if we saw this chunk before, skip the whole document
                    seen_chunks.add(chunk_str)
                results.append(processed_doc)
                collected_chunks += len(processed_doc["json_chunks"])
            if collected_chunks >= num_chunks:
                break
        return results
