# scienceqa-9444

## ScienceQA RAG Pipeline - Quick Start

### 1. Download and Filter Dataset
```bash
python src/data_preparation.py
```

### 2. Generate Text Embeddings
```bash
python src/text_vectorization.py
```

### 3. Build FAISS Vector Index
```bash
python src/build_faiss_index.py
```

### 4. Run RAG QA Pipeline (Interactive)
```bash
python src/rag_pipeline.py
```

---

- Make sure to install all dependencies before running the scripts:
```bash
pip install -r requirement.txt
```
- All data and intermediate files will be saved in the `data/` directory.