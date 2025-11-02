# Set up local dense retriever

```bash
uv venv

source .venv/bin/activate

uv pip install torch faiss-cpu==1.12.0 uvicorn fastapi huggingface_hub datasets transformers 

python download.py --save_path ./corpus_and_index
cd corpus_and_index
cat part_* > e5_Flat.index
rm part_*
gzip -d wiki-18.jsonl.gz


python retrieval_server.py \
    --index_path corpus_and_index/e5_Flat.index \
    --corpus_path corpus_and_index/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 

```
