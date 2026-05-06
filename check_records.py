import json

with open("processed/repair_records.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        rec = json.loads(line)
        print(f"\n--- Record {i+1} ---")
        for key, val in rec.items():
            print(f"{key:20s}: {str(val)[:100]}")