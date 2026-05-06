import json
from qdrant_client import QdrantClient

# Check Qdrant total count
client = QdrantClient(url="http://localhost:6333")
info = client.get_collection("nic_medsearch")
print(f"Total vectors in Qdrant: {info.points_count}")

# Check technical specs content
print("\n── Sample from technical_specs.jsonl ──\n")
with open("processed/technical_specs.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        rec = json.loads(line)
        print(f"doc_type : {rec.get('doc_type')}")
        print(f"source   : {rec.get('source_file')}")
        print(f"content  : {rec.get('content', '')[:200]}")
        print()

# Check manuals content
print("\n── Sample from manuals.jsonl ──\n")
with open("processed/manuals.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        rec = json.loads(line)
        print(f"doc_type : {rec.get('doc_type')}")
        print(f"source   : {rec.get('source_file')}")
        print(f"content  : {rec.get('content', '')[:200]}")
        print()

client = QdrantClient(url="http://localhost:6333")

# Search specifically for tech spec records
results = client.scroll(
    collection_name="nic_medsearch",
    scroll_filter=None,
    limit=5,
    with_payload=True,
    with_vectors=False
)

print("Sample records in Qdrant:\n")
for point in results[0]:
    print(f"doc_type : {point.payload.get('doc_type')}")
    print(f"source   : {point.payload.get('source_file')}")
    print(f"content  : {str(point.payload.get('content', ''))[:150]}")
    print()

# Count by doc_type in Qdrant
from collections import Counter
counts = Counter()
offset = None
while True:
    results, offset = client.scroll(
        collection_name="nic_medsearch",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    for point in results:
        counts[point.payload.get('doc_type', 'unknown')] += 1
    if offset is None:
        break

print("\nRecords in Qdrant by doc_type:")
for doc_type, count in counts.items():
    print(f"  {doc_type}: {count}")