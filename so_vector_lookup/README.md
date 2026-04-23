## StackOverflow vector lookup track

This track benchmarks the performance difference between two ways of supplying
query vectors for kNN search:

- `GET` a document by `_id` and pass `titleVector` as `query_vector`
- use `query_vector_builder.lookup` to fetch `titleVector` server-side

Both search paths run with fixed parameters:

- `k = 10`
- `num_candidates = 100`

The track keeps indexing behavior aligned with `so_vector` and runs an
`index-and-search` challenge.

## ID pool behavior

After indexing, search operations gather the first 1000 document `_id` values
(ordered by `questionId` ascending) and store them in memory in the runner.
Each iteration uses the next ID from this in-memory pool in round-robin order.

This track is intended for latency and throughput comparison only. It does not
measure recall.

## Parameters

This track accepts the same indexing and challenge parameters as `so_vector`,
including:

- `bulk_size` (default: 500)
- `bulk_indexing_clients` (default: 1)
- `ingest_percentage` (default: 100)
- `vector_index_type` (default: `bbq_hnsw`)
- `corpora` (default: `so_vector_float`)
- `warmup_iterations` (default: 100)
- `iterations` (default: 100)
- `search_clients` (default: 8)
- `include_non_serverless_index_settings`
- `include_force_merge`

## License

We use the same data license as the source StackOverflow dump:
[CC-SA-4.0](http://creativecommons.org/licenses/by-sa/4.0/).
