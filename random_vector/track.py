import random

from esrally.track.params import ParamSource


class RandomBulkParamSource(ParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self._bulk_size = params.get("bulk-size", 1000)
        self._index_name = params.get("index", track.indices[0].name)
        self._dims = params.get("dims", 128)
        self._partitions = params.get("partitions", 1000)

    def params(self):
        import numpy as np

        bulk_data = []
        for _ in range(self._bulk_size):
            # generate a random byte array of length `dims`
            vec = np.random.bytes(self._dims).hex()
            bulk_data.append({"index": {"_index": self._index_name}})
            bulk_data.append({"emb": vec})

        return {
            "body": bulk_data,
            "bulk-size": self._bulk_size,
            "action-metadata-present": True,
            "unit": "docs",
            "index": self._index_name,
            "type": "",
        }


def generate_knn_query(query_vector, partition_id, k):
    return {
        "knn": {
            "field": "emb",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": k,
            "filter": {"term": {"partition_id": partition_id}},
        }
    }


def generate_script_query(query_vector, partition_id):
    return {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {"source": "(1024.0 - hamming(params.query_vector, 'emb'))/1024", "params": {"query_vector": query_vector}},
            }
        }
    }


class RandomSearchParamSource:
    def __init__(self, track, params, **kwargs):
        # choose a suitable index: if there is only one defined for this track
        # choose that one, but let the user always override index
        if len(track.indices) == 1:
            default_index = track.indices[0].name
        else:
            default_index = "_all"

        self._index_name = params.get("index", default_index)
        self._cache = params.get("cache", False)
        self._partitions = params.get("partitions", 1000)
        self._dims = params.get("dims", 128)
        self._top_k = params.get("k", 10)
        self._script = params.get("script", True)
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        import numpy as np

        partition_id = random.randint(0, self._partitions)
        # Get unsigned bytes array of length `dims`
        query_vec = list(np.random.bytes(self._dims))
        query_vec = [x - 128 for x in query_vec]
        if self._script:
            query = generate_script_query(query_vec, partition_id)
        else:
            query = generate_knn_query(query_vec, partition_id, self._topk)
        return {"index": self._index_name, "cache": self._cache, "size": self._top_k, "_source_excludes": ["emb"], "body": query}


def register(registry):
    registry.register_param_source("random-bulk-param-source", RandomBulkParamSource)
    registry.register_param_source("knn-param-source", RandomSearchParamSource)
