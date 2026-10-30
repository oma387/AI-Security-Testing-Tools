1. `vector_db_auditor.py`
Read-only inventory for Qdrant, ChromaDB, Milvus, and Weaviate. It lists collection/class metadata and counts where supported. It does not export stored documents, payload values, or vectors.
Examples:
```bash
python vector_db_auditor.py --self-test
python vector_db_auditor.py --db qdrant --host 127.0.0.1
python vector_db_auditor.py --db chroma --host 127.0.0.1
python vector_db_auditor.py --db milvus --host 127.0.0.1
python vector_db_auditor.py --db weaviate --host 127.0.0.1
```
Default ports: Qdrant 6333, Chroma 8000, Milvus 19530, Weaviate 8080. Use `--port` to override. Add `--tls` for HTTPS/TLS where supported. Set `VECTOR_DB_TOKEN` or `WEAVIATE_API_KEY` for the selected service's configured auth. Authentication formats vary by deployment; adapt the client setup to your service's actual configuration.
2. `embedding_similarity_lab.py`
Generates embeddings with OpenAI, Sentence-Transformers/Hugging Face, or Ollama and ranks supplied candidate phrases against a reference vector. It is a similarity experiment, not an inversion tool. Candidate text and the target vector must use the exact same model and compatible settings.
Examples:
```bash
python embedding_similarity_lab.py --self-test
python embedding_similarity_lab.py --provider huggingface --model sentence-transformers/all-MiniLM-L6-v2 --target-text "Synthetic Northstar test policy" --candidates-file candidate_phrases.txt
python embedding_similarity_lab.py --provider openai --model text-embedding-3-small --target-text "Synthetic Northstar test policy" --candidates-file candidate_phrases.txt
python embedding_similarity_lab.py --provider ollama --model nomic-embed-text --target-text "Synthetic Northstar test policy" --candidates-file candidate_phrases.txt
```
For OpenAI set `OPENAI_API_KEY`. Ollama requires a locally running server and a model already pulled. Hugging Face may download the model on first use.
To compare against a vector file, provide `--target-vector-file target_vector.json`; it may contain a JSON float array or `{"embedding": [...]}`. Only use vectors you own or are authorized to assess.
3. `ai_prompt_test_runner.py`
Runs a small JSONL prompt test set against an authorized chat endpoint and records bounded response excerpts, status codes, and latency. It is rate-limited to at least one second between requests and at most 20 requests per run. It does not automatically score safety or claim that a response is vulnerable.
Example:
```bash
python ai_prompt_test_runner.py --self-test
export AI_TEST_BEARER_TOKEN="YOUR_TEST_TOKEN"
python ai_prompt_test_runner.py --authorized --url http://127.0.0.1:8000/api/v1/chat/query --cases prompt_test_cases.jsonl --template ai_payload_template.json --output results.jsonl
```
Change the payload template to match the API's documented request schema. Use only a test endpoint, test account, and synthetic test data. Review every result manually.
Installation
Create a virtual environment and install only the adapter/provider you plan to use:
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-qdrant.txt       # or requirements-chroma.txt / requirements-milvus.txt
python -m pip install -r requirements-openai.txt       # or requirements-huggingface.txt
```
Weaviate schema inventory and the prompt runner use the core `requests` dependency. Ollama also uses the local HTTP API through `requests`.
Limitations
There is no universal vector-database wire protocol, and vendor SDKs differ. The inventory script uses the selected vendor's client/API and may need a small connection/authentication adjustment for a particular deployment or SDK version.
There is no universal algorithm that reconstructs original text from arbitrary embeddings. Similarity scores are not proof of plaintext recovery. The similarity script intentionally tests only supplied candidate phrases with the same embedding model.
The included self-tests validate local helper logic. They do not replace integration testing against a real Qdrant, ChromaDB, Milvus, Weaviate, OpenAI, Hugging Face, or Ollama deployment.
