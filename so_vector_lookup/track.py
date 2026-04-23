import logging
from collections import defaultdict
from threading import Lock
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_K = 10
DEFAULT_NUM_CANDIDATES = 100
DEFAULT_LOOKUP_POOL_SIZE = 1000
DEFAULT_VECTOR_FIELD = "titleVector"


class LookupIdPool:
    _ids_by_index: Dict[str, List[str]] = {}
    _position_by_index: Dict[str, int] = defaultdict(int)
    _lock = Lock()

    @classmethod
    async def _load_ids(cls, es, index: str, size: int):
        logger.info("Loading lookup id pool for index '%s' with size=%d", index, size)
        response = await es.search(
            index=index,
            size=size,
            request_cache=True,
            body={
                "query": {"match_all": {}},
                "sort": [{"questionId": "asc"}],
                "_source": False,
            },
        )
        ids = [hit["_id"] for hit in response["hits"]["hits"]]
        if not ids:
            raise ValueError(f"No ids found for lookup pool in index '{index}'.")
        with cls._lock:
            cls._ids_by_index[index] = ids
            cls._position_by_index[index] = 0

    @classmethod
    async def next_id(cls, es, index: str, size: int) -> str:
        if index not in cls._ids_by_index:
            await cls._load_ids(es, index, size)

        with cls._lock:
            ids = cls._ids_by_index[index]
            if not ids:
                raise ValueError(f"Lookup id pool for index '{index}' is empty.")
            position = cls._position_by_index[index]
            doc_id = ids[position]
            cls._position_by_index[index] = (position + 1) % len(ids)
            return doc_id


class GetThenKnnRunner:
    async def __call__(self, es, params):
        index = params["index"]
        vector_field = params.get("field", DEFAULT_VECTOR_FIELD)
        k = params.get("k", DEFAULT_K)
        num_candidates = params.get("num_candidates", DEFAULT_NUM_CANDIDATES)
        lookup_pool_size = params.get("lookup_pool_size", DEFAULT_LOOKUP_POOL_SIZE)
        request_cache = params.get("cache", False)

        doc_id = await LookupIdPool.next_id(es, index, lookup_pool_size)
        source_response = await es.get(index=index, id=doc_id, source_includes=[vector_field])
        query_vector = source_response["_source"][vector_field]

        search_response = await es.search(
            index=index,
            size=k,
            request_cache=request_cache,
            body={
                "knn": {
                    "field": vector_field,
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": num_candidates,
                }
            },
        )
        return {
            "weight": 1,
            "unit": "ops",
            "hits": len(search_response["hits"]["hits"]),
        }

    def __repr__(self, *args, **kwargs):
        return "get-then-knn"


class LookupKnnRunner:
    async def __call__(self, es, params):
        index = params["index"]
        lookup_index = params.get("lookup_index", index)
        vector_field = params.get("field", DEFAULT_VECTOR_FIELD)
        k = params.get("k", DEFAULT_K)
        num_candidates = params.get("num_candidates", DEFAULT_NUM_CANDIDATES)
        lookup_pool_size = params.get("lookup_pool_size", DEFAULT_LOOKUP_POOL_SIZE)
        request_cache = params.get("cache", False)

        doc_id = await LookupIdPool.next_id(es, index, lookup_pool_size)

        lookup = {
            "index": lookup_index,
            "id": doc_id,
            "path": vector_field,
        }
        if "routing" in params:
            lookup["routing"] = params["routing"]

        search_response = await es.search(
            index=index,
            size=k,
            request_cache=request_cache,
            body={
                "knn": {
                    "field": vector_field,
                    "k": k,
                    "num_candidates": num_candidates,
                    "query_vector_builder": {
                        "lookup": lookup,
                    },
                }
            },
        )
        return {
            "weight": 1,
            "unit": "ops",
            "hits": len(search_response["hits"]["hits"]),
        }

    def __repr__(self, *args, **kwargs):
        return "lookup-knn"


def register(registry):
    registry.register_runner("get-then-knn", GetThenKnnRunner(), async_runner=True)
    registry.register_runner("lookup-knn", LookupKnnRunner(), async_runner=True)
