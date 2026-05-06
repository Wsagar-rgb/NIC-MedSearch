# NIC MedSearch — Local RAG Pipeline

## Project Structure
```
NIC_MedSearch/
├── raw/                     ← PUT YOUR CSV FILES HERE
├── processed/               ← Auto-generated cleaned outputs
├── logs/                    ← Processing logs
├── config.py                ← All settings (paths, model names)
├── utils.py                 ← Shared cleaning functions
├── stream1_csv.py           ← CSV cleaner
├── embed_and_index.py       ← Qdrant embedding + indexing
├── query.py                 ← RAG search interface
├── run_all.py               ← Master runner
└── requirements.txt
```

## CSV Format (your actual columns)
| Column | Description |
|---|---|
| Equipment Description | Name of the equipment |
| Manufacturer | Equipment manufacturer |
| Model | Model number |
| Problem | Fault/problem description |
| Initial Diagnosis | First assessment |
| Work Done | How it was fixed |

## Running the Pipeline

### Full pipeline (clean + embed + index):
```bash
python run_all.py
```

### Clean only (no embedding):
```bash
python run_all.py --skip-embed
```

### Steps individually:
```bash
python stream1_csv.py      # Step 1: Clean CSV data
python embed_and_index.py  # Step 2: Embed + index in Qdrant
python query.py            # Step 3: Search
```

## Prerequisites
- Docker running with Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Ollama running with llama3.2:3b: `ollama run llama3.2:3b`
