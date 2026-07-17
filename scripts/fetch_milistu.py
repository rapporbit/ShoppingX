"""Fetch milistu/AMAZON-Products-2023 WITHOUT the precomputed `embeddings` column.

The HF dataset ships ~1.7GB of arrow that is dominated by a precomputed
text-embedding-3-small `embeddings` column we cannot reuse (we re-embed with
BGE-M3). We read the auto-converted parquet over HTTP and column-prune the
embedding column via range requests, so only the useful metadata is downloaded.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = "data/rag/milistu-amazon-products-2023-slim.parquet"
BASE = "datasets/milistu/AMAZON-Products-2023@refs/convert/parquet/default/train"

fs = HfFileSystem()
files = sorted(fs.glob(BASE + "/*.parquet"))
print("parquet shards:", files)

with fs.open(files[0]) as f:
    schema = pq.ParquetFile(f).schema_arrow
keep = [n for n in schema.names if "embed" not in n.lower()]
dropped = [n for n in schema.names if n not in keep]
print("dropping:", dropped)
print("keeping :", keep)

tables = []
for fp in files:
    with fs.open(fp) as f:
        tables.append(pq.read_table(f, columns=keep))
    print("read", fp)

table = pa.concat_tables(tables)
print("rows:", table.num_rows, "cols:", table.num_columns)
pq.write_table(table, OUT, compression="zstd")
print("wrote", OUT)
