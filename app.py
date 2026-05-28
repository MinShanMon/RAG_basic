from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
import sys
import stat
import shutil
import logging
import re
import socket

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------- CONFIG ----------
EXCLUDE = [".obsidian", "copilot", "99-templates", "06-AI System"]

# Define vaults: key = internal id, label = shown in UI, folder = markdown source
VAULTS = {
    "mine": {
        "label": "DevOps",
        "folder": "./DevOps",
        "persist_dir": "./db/mine",
        "collection": "vault_mine",
    },
    "yuriko": {
        "label": "Yuriko",
        "folder": "./Yuriko/Yuriko",
        "persist_dir": "./db/yuriko",
        "collection": "vault_yuriko",
    },
}

# ---------- MODELS ----------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.1:8b")


# ---------- DOCS ----------
def load_docs(folder: str):
    loader = DirectoryLoader(
        folder,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
        silent_errors=True,
    )
    docs = loader.load()
    docs = [d for d in docs if not any(ex in d.metadata.get("source", "") for ex in EXCLUDE)]
    log.info(f"Loaded {len(docs)} documents after filtering from {folder}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)


# ---------- DB ----------
def _newest_md_mtime(folder: str) -> float:
    """Return the modification time of the most recently changed .md file in folder."""
    newest = 0.0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".md"):
                try:
                    mtime = os.path.getmtime(os.path.join(root, f))
                    if mtime > newest:
                        newest = mtime
                except OSError:
                    pass
    return newest


def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _find_available_port(start_port: int, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_tries - 1}")


def build_or_load_db(vault_cfg: dict, force_rebuild: bool = False):
    persist_dir = vault_cfg["persist_dir"]
    collection = vault_cfg["collection"]
    folder = vault_cfg["folder"]
    last_indexed_file = os.path.join(persist_dir, ".last_indexed")

    if force_rebuild and os.path.exists(persist_dir):
        log.info(f"Force rebuild: deleting {persist_dir}...")
        shutil.rmtree(persist_dir, onexc=_remove_readonly)

    # Auto-rebuild if any .md file is newer than the last index timestamp
    # (We use a dedicated .last_indexed file — NOT chroma.sqlite3 — because
    #  Chroma updates sqlite3 on every read, making mtime comparisons unreliable.)
    if not force_rebuild and os.path.exists(last_indexed_file):
        try:
            last_indexed = float(open(last_indexed_file).read().strip())
            newest_note = _newest_md_mtime(folder)
            if newest_note > last_indexed:
                log.info(f"[{collection}] Notes are newer than last index — triggering auto-rebuild...")
                shutil.rmtree(persist_dir, onexc=_remove_readonly)
        except (ValueError, OSError):
            pass

    db = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    count = db._collection.count()
    log.info(f"[{collection}] Existing vector count: {count}")

    if count == 0:
        log.info(f"[{collection}] Building DB from {folder}...")
        docs = load_docs(folder)
        if not docs:
            log.warning(f"[{collection}] No markdown documents found under {folder}")
            return db
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        final_count = db._collection.count()
        log.info(f"[{collection}] Created DB with {final_count} chunks from {len(set(d.metadata.get('source') for d in docs))} files")
        # Record when we last indexed so auto-rebuild comparisons are reliable
        with open(last_indexed_file, "w") as f:
            f.write(str(os.path.getmtime(os.path.join(persist_dir, "chroma.sqlite3"))))

    return db


# Load all vault DBs at startup — skip if rebuild is pending (files must not be open when deleted)
_REBUILD_REQUESTED = "--rebuild" in sys.argv or "rebuild" in sys.argv
if not _REBUILD_REQUESTED:
    vault_dbs = {key: build_or_load_db(cfg, force_rebuild=False) for key, cfg in VAULTS.items()}
else:
    vault_dbs = {}


# ---------- QUERY ----------
SCORE_THRESHOLD = 1.2   # L2 distance — below this = relevant match (lower is better)
                        # nomic-embed-text normalized embeddings: 0=identical, ~1.41=opposite
                        # typical relevant matches score 0.8–1.1

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "how", "i", "in", "is", "it", "my", "of", "on", "or", "please", "show", "tell",
    "that", "the", "this", "to", "what", "when", "where", "which", "who", "why", "with",
    "explain",
}


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _query_terms(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def _extract_snippet(content: str, terms: list[str]) -> str:
    lines = content.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(t in low for t in terms):
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            return "\n".join(lines[start:end]).strip()[:1200]
    return "\n".join(lines[:6])[:350]


def _keyword_fallback_chunks(folder: str, question: str, limit: int = 4) -> list[dict]:
    terms = _query_terms(question)
    if not terms:
        return []

    phrase = " ".join(terms)
    normalized_phrase = _normalize_text(phrase)
    candidates = []
    for root, _, files in os.walk(folder):
        if any(ex in root for ex in EXCLUDE):
            continue
        for name in files:
            if not name.endswith(".md"):
                continue

            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue

            haystack = f"{path}\n{content}".lower()
            score = sum(1 for t in terms if t in haystack)
            if phrase and phrase in haystack:
                score += 5

            # Strongly prefer files whose names/paths directly match query intent.
            rel_path = os.path.relpath(path).replace("/", "\\")
            normalized_rel_path = _normalize_text(rel_path)
            normalized_base_name = _normalize_text(os.path.splitext(name)[0])
            if normalized_phrase and normalized_phrase in normalized_base_name:
                score += 20
            elif all(t in normalized_base_name for t in terms):
                score += 12

            if normalized_phrase and normalized_phrase in normalized_rel_path:
                score += 8

            if score == 0:
                continue

            snippet = _extract_snippet(content, terms)
            candidates.append({
                "source": rel_path,
                "score": score,
                "content": snippet,
                "full": content[:4000],
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]

def query_ai(question: str, vault_key: str) -> dict:
    db = vault_dbs[vault_key]
    vault_folder = VAULTS[vault_key]["folder"]
    query_terms = _query_terms(question)

    docs_mmr = db.max_marginal_relevance_search(question, k=8, fetch_k=150)
    log.info(f"MMR retrieved {len(docs_mmr)} chunks: {[d.metadata.get('source','?') for d in docs_mmr]}")

    results = db.similarity_search_with_score(question, k=8)
    semantic_sources = []
    best_score = float("inf")
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        semantic_sources.append({"source": src, "score": round(score, 4)})
        log.info(f"  score={score:.4f} source={src}")
        if score < best_score:
            best_score = score

    # Require a majority of query terms to match — not just any one term.
    # A single common word like "iam" or "policy" in an irrelevant chunk is not enough
    # to suppress the keyword fallback.
    _min_term_hits = max(2, len(query_terms) // 2)
    semantic_hits_query_terms = any(
        sum(1 for t in query_terms if t in f"{d.metadata.get('source', '')} {d.page_content}".lower()) >= _min_term_hits
        for d in docs_mmr
    )

    # Always run keyword fallback — semantic search can retrieve off-topic chunks
    # (e.g. CodeDeploy notes that mention "iam" and "policy") and bury the real answer.
    # Keyword fallback ensures the most lexically-relevant file is always in context.
    fallback_chunks = _keyword_fallback_chunks(vault_folder, question, limit=4)
    if fallback_chunks:
        log.info(f"Keyword fallback retrieved {len(fallback_chunks)} files: {[c['source'] for c in fallback_chunks]}")
    strong_fallback = [c for c in fallback_chunks if c["score"] >= 2][:3]
    fallback_sources = [{"source": c["source"], "score": f"kw:{c['score']}"} for c in strong_fallback]

    # If keyword fallback has a strong direct match, run a focused prompt.
    # This fires whether or not semantic retrieval looked on-topic — semantic can
    # pick up noise from other services that share words like "iam" / "policy".
    if fallback_chunks and fallback_chunks[0]["score"] >= max(3, len(query_terms) - 1):
        top = fallback_chunks[0]
        focused_chunks = strong_fallback if strong_fallback else [top]
        focused_context = "\n\n".join([f"[From {c['source']}]\n{c.get('full', c['content'])}" for c in focused_chunks])
        focused_prompt = f"""You are a helpful assistant. Answer the question using ONLY the notes below.
Rules:
- Answer the question DIRECTLY and CONCISELY. Start with the direct answer, then add supporting detail.
- If the notes describe a service/feature/type that uses or requires something, that IS the answer — state it.
- Do NOT add, infer, or speculate about services not mentioned in the notes.
- Do NOT apply concepts to other services not explicitly mentioned in the notes.
- If you see image/embed syntax like ![[...]], ignore the markup and use the surrounding text.
- Only say NOT FOUND IN MY NOTES if the notes contain absolutely no information relevant to the question.

Notes:
{focused_context}

Question: {question}
Answer:"""
        focused_answer = llm.invoke(focused_prompt)

        if isinstance(focused_answer, str) and "NOT FOUND IN MY NOTES" not in focused_answer.upper():
            return {
                "answer": focused_answer,
                "sources": fallback_sources,
            }

        # LLM said NOT FOUND despite a strong keyword hit. Retry once with a
        # more lenient extraction prompt using the full note content.
        retry_context = "\n\n".join([f"[From {c['source']}]\n{c.get('full', c['content'])}" for c in focused_chunks])
        retry_prompt = f"""You are a helpful assistant. The user is looking for an answer inside the notes below.
Even if the notes do not use the exact same words as the question, extract any relevant information that helps answer it.
Do NOT speculate about services not mentioned.

Notes:
{retry_context}

Question: {question}
Answer:"""
        retry_answer = llm.invoke(retry_prompt)
        if isinstance(retry_answer, str) and "NOT FOUND IN MY NOTES" not in retry_answer.upper():
            return {"answer": retry_answer, "sources": fallback_sources}

        # Final safety net: return the full note content so the user can read it themselves.
        return {
            "answer": f"Based on your notes, this is covered in {top['source']}:\n\n{top.get('full', top['content'])}",
            "sources": fallback_sources,
        }

    if best_score > SCORE_THRESHOLD and not fallback_chunks:
        log.info(f"Best score {best_score:.4f} exceeds threshold {SCORE_THRESHOLD} and keyword fallback found nothing — returning NOT FOUND")
        return {"answer": "NOT FOUND IN MY NOTES", "sources": semantic_sources}

    # Build context: strong keyword matches go FIRST so the LLM sees the most
    # lexically-relevant note before any noisy semantic chunks.
    seen_content = set()
    context_parts = []
    for c in strong_fallback:
        text = f"[From {c['source']}]\n{c.get('full', c['content'])}"
        if text not in seen_content:
            seen_content.add(text)
            context_parts.append(text)
    for doc, _score in results:
        text = doc.page_content
        if text not in seen_content:
            seen_content.add(text)
            context_parts.append(text)
    # Add any MMR chunks not already included (for breadth)
    for doc in docs_mmr:
        text = doc.page_content
        if text not in seen_content:
            seen_content.add(text)
            context_parts.append(text)
    # Append remaining weak fallback chunks
    for c in fallback_chunks:
        text = f"[From {c['source']}]\n{c['content']}"
        if text not in seen_content:
            seen_content.add(text)
            context_parts.append(text)
    context = "\n\n".join(context_parts)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the notes below.
Rules:
- Answer the question DIRECTLY and CONCISELY. Start with the direct answer, then add supporting detail.
- If the notes describe a service/feature/type that uses or requires something, that IS the answer — state it.
- Do NOT add, infer, or speculate about services not mentioned in the notes.
- Do NOT apply concepts to other services not explicitly mentioned in the notes.
- Do NOT say things like "the same concept can apply to X" unless X is stated in the notes.
- Only say NOT FOUND IN MY NOTES if the notes contain absolutely no information relevant to the question.

Notes:
{context}

Question: {question}
Answer:"""

    answer = llm.invoke(prompt)

    # Prefer concise relevant source list: fallback sources first (if any), then semantic sources.
    sources = fallback_sources + semantic_sources

    # Guardrail: if LLM says NOT FOUND but keyword fallback found a strong direct match,
    # return an extractive answer from the matched note instead of a false negative.
    if (
        isinstance(answer, str)
        and "NOT FOUND IN MY NOTES" in answer.upper()
        and fallback_chunks
        and fallback_chunks[0]["score"] >= max(3, len(query_terms) - 1)
    ):
        top = fallback_chunks[0]
        answer = (
            f"Based on your notes, this is covered in {top['source']}:\n\n"
            f"{top['content']}"
        )

    return {"answer": answer, "sources": sources}


# ---------- FASTAPI ----------
app = FastAPI(title="RAG API")

# -- IP Allowlist: add your Tailscale IPs here --
ALLOWED_IPS = {
    "127.0.0.1",         # localhost (for local testing)
    "100.100.212.63",    # this PC (Tailscale)
    "100.122.119.32",    # Pixel 7a (Tailscale)
    "100.88.254.119",    # iPhone 15 Pro (Tailscale)
    "100.101.121.63",    # tss20230915 (Tailscale)
}

@app.middleware("http")
async def ip_allowlist(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        log.warning(f"Blocked request from {client_ip}")
        # Temporarily allow /myip from anyone to diagnose phone IP
        if request.url.path == "/myip":
            return await call_next(request)
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)


class AskRequest(BaseModel):
    question: str
    vault: str = "mine"


class AskResponse(BaseModel):
    answer: str
    sources: list


@app.get("/", response_class=HTMLResponse)
def ui():
    vault_options = "\n".join(
        f'<option value="{k}">{v["label"]}</option>' for k, v in VAULTS.items()
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG</title>
  <style>
    body {{ font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; background: #1e1e2e; color: #cdd6f4; }}
    h1 {{ color: #cba6f7; }}
    select, textarea {{ width: 100%; padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #45475a; background: #313244; color: #cdd6f4; box-sizing: border-box; }}
    select {{ margin-bottom: 10px; }}
    button {{ margin-top: 10px; padding: 10px 24px; font-size: 16px; background: #cba6f7; color: #1e1e2e; border: none; border-radius: 8px; cursor: pointer; }}
    button:disabled {{ opacity: 0.5; }}
    #answer {{ margin-top: 24px; background: #313244; padding: 16px; border-radius: 8px; white-space: pre-wrap; }}
    #sources {{ margin-top: 12px; font-size: 13px; color: #a6adc8; }}
    label {{ font-size: 13px; color: #a6adc8; display: block; margin-bottom: 4px; }}
  </style>
</head>
<body>
  <h1>RAG</h1>
  <label>Notes vault</label>
  <select id="vault">
    {vault_options}
  </select>
  <textarea id="q" rows="3" placeholder="Ask a question..."></textarea><br>
  <button id="btn" onclick="ask()">Ask</button>
  <div id="answer"></div>
  <div id="sources"></div>
  <script>
    async function ask() {{
      const q = document.getElementById('q').value.trim();
      const vault = document.getElementById('vault').value;
      if (!q) return;
      const btn = document.getElementById('btn');
      btn.disabled = true;
      document.getElementById('answer').innerText = 'Thinking...';
      document.getElementById('sources').innerText = '';
      try {{
        const res = await fetch('/ask', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{question: q, vault: vault}})
        }});
        const data = await res.json();
        document.getElementById('answer').innerText = data.answer;
        document.getElementById('sources').innerText = data.sources.map(s => s.source + ' (' + s.score + ')').join('\\n');
      }} catch(e) {{
        document.getElementById('answer').innerText = 'Error: ' + e;
      }}
      btn.disabled = false;
    }}
    document.getElementById('q').addEventListener('keydown', e => {{ if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); ask(); }}}});
  </script>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok", "vaults": {k: vault_dbs[k]._collection.count() for k in vault_dbs}}

@app.get("/myip")
def myip(request: Request):
    return {"your_ip": request.client.host}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if req.vault not in vault_dbs:
        raise HTTPException(status_code=400, detail=f"unknown vault: {req.vault}")
    log.info(f"Question [{req.vault}]: {req.question}")
    result = query_ai(req.question, req.vault)
    return result


# ---------- CLI fallback ----------
if __name__ == "__main__":
    import uvicorn
    if _REBUILD_REQUESTED:
        log.info("Rebuild requested — rebuilding all vaults...")
        for key, cfg in VAULTS.items():
            build_or_load_db(cfg, force_rebuild=True)
        log.info("Rebuild complete. Reloading vault DBs...")
        vault_dbs.update({key: build_or_load_db(cfg, force_rebuild=False) for key, cfg in VAULTS.items()})
        for flag in ("--rebuild", "rebuild"):
            if flag in sys.argv:
                sys.argv.remove(flag)  # prevent re-trigger on uvicorn module import
    preferred_port = int(os.getenv("PORT", "8000"))
    port = _find_available_port(preferred_port)
    if port != preferred_port:
        log.warning(f"Port {preferred_port} is in use. Starting on {port} instead.")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")