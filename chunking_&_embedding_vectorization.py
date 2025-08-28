
# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU langchain-text-splitters

#!pip install faiss-cpu
#!pip install cohere

#!pip install --upgrade pip
#!pip install requests pandas unstructured langchain openpyxl python-docx tqdm
#!pip install -U langchain-community

import os
import re
import time
import json
import tempfile
from typing import List, Tuple, Dict, Optional, Set

import requests
import pandas as pd
from urllib.parse import urlparse, urljoin
from langchain.schema import Document
try:
    from unstructured.partition.auto import partition
except Exception:
    partition = None

import os
os.environ["COHERE_API_KEY"] = "aa-dG91mhIw7EajpJofVphAH5nUFsvYPrb0lCxRL7zSELzU6TWs"


class Settings:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    EBOO_TOKEN      = os.getenv("EBOO_TOKEN", "TPSHJFdOWcqR2I5wRgJkFloiBqmQBxwl1")
    JINA_API_KEY    = os.getenv("JINA_API_KEY", "jina_71a9a199a9fd4109826a1f417ef04ef9Z6jA3xRAka0d4sa66V8S6q_0SIOf")

    # Embedding settings
    EMBED_MODEL     = os.getenv("EMBED_MODEL", "embed-multilingual-v3.0")
    EMBED_INPUT_TYPE= os.getenv("EMBED_INPUT_TYPE", "clustering")
    COHERE_BATCH_SIZE = int(os.getenv("COHERE_BATCH_SIZE", 64))
    EMBED_RETRY       = int(os.getenv("EMBED_RETRY", 3))

    # FAISS / paths
    FAISS_DIR = os.getenv("FAISS_DIR", "faiss_db")

# نمونه‌سازی
settings = Settings()

class EbooClient:
    def __init__(self, token: Optional[str] = None, api_url: str = "https://www.eboo.ir/api/ocr/getway"):
        self.token = token or settings.EBOO_TOKEN
        self.api_url = api_url

    def upload_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            files = {"filehandle": (file_path, f)}
            payload = {"token": self.token, "command": "addfile"}
            resp = requests.post(self.api_url, data=payload, files=files)
        resp.raise_for_status()
        return resp.json().get("FileToken")

    def convert(self, filetoken: str) -> str:
        payload = {"token": self.token, "command": "convert", "filetoken": filetoken,
                   "method": 4, "output": "keeplayout"}
        resp = requests.post(self.api_url, data=payload)
        resp.raise_for_status()
        return resp.json().get("FileToDownload")

    def download_to_file(self, download_url: str, out_path: str):
        r = requests.get(download_url)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path

def extract_text_from_pdf_or_image(path: str) -> List[Tuple[str, Dict]]:
        client = EbooClient()
        token = client.upload_file(path)
        download_url = client.convert(token)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        client.download_to_file(download_url, out.name)
        from unstructured.partition.auto import partition
        elements = partition(out.name)
        text = " ".join([el.text for el in elements if hasattr(el, "text")])
        os.unlink(out.name)
        return [(text, {"source": path})]

class JinaCrawler:
    def __init__(self, api_key: str, api_url: str = "https://r.jina.ai",
                 max_depth: int = 1, delay: float = 0.5):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.max_depth = max_depth
        self.delay = delay
        self.visited = set()
        self.pages: List[Dict] = []

    def scrape_page(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.post(self.api_url, json={"url": url}, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                print(f"Failed {url}: {resp.status_code}")
                return None

            md_text = resp.text
            title_match = re.search(r'^Title:\s*(.+)', md_text, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else ""
            links = [urljoin(url, href) for _, href in re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', md_text)]

            content = re.sub(r'^Title:.*\n', '', md_text, flags=re.MULTILINE)
            content = re.sub(r'^URL Source:.*\n', '', content, flags=re.MULTILINE).strip()
            if not content:
                content = title or f"URL: {url}"


            return {"url": url, "title": title, "content": content, "links": links}
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def crawl(self, start_url: str, depth: int = 0):
        if start_url in self.visited or depth > self.max_depth:
            return
        print(f"Crawling {start_url} at depth {depth}")
        self.visited.add(start_url)

        page_data = self.scrape_page(start_url)
        if page_data:
            self.pages.append(page_data)
            domain = urlparse(start_url).netloc
            for link in page_data["links"]:
                if urlparse(link).netloc == domain:
                    self.crawl(link, depth + 1)
        time.sleep(self.delay)



def _split_json_by_size(json_data, max_chunk_size: int, min_chunk_size: int, convert_lists: bool = True) -> List[str]:

    text = json.dumps(json_data, ensure_ascii=False)
    texts = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        texts.append(text[start:end])
        start = end
    return texts

def chunk_file(file_path: str, max_chunk_size: int = 500, min_chunk_size: int = 100) -> List[Document]:

    docs: List[Document] = []
    path_lower = file_path.lower()

    # ---------- Excel ----------
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(Document(page_content=content, metadata={"source": file_path}))

    # ---------- CSV ----------
    elif path_lower.endswith(".csv"):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(Document(page_content=content, metadata={"source": file_path}))

    # ---------- JSON ----------
    elif path_lower.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_str = json.dumps(data, ensure_ascii=False)
        if len(json_str) <= max_chunk_size:
            docs.append(Document(page_content=json_str, metadata={"source": file_path}))
        else:
            try:

                from langchain_text_splitters import RecursiveJsonSplitter

                splitter = RecursiveJsonSplitter(
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size
                )
                texts = splitter.split_text(json_data=data, convert_lists=True)
            except Exception:

                texts = _split_json_by_size(data, max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size)

            for t in texts:
                docs.append(Document(page_content=t, metadata={"source": file_path}))
    # ---------- DOCX ----------
    elif path_lower.endswith(".docx"):
        if partition is None:
            raise RuntimeError("unstructured.partition.auto.partition is required to parse DOCX files.")
        elements = partition(file_path)
        content = " ".join([el.text for el in elements if hasattr(el, "text")])
        words = content.split()
        start = 0
        chunk_idx = 1
        while start < len(words):
            end = min(start + max_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            docs.append(Document(page_content=chunk_text, metadata={"source": file_path, "chunk_index": chunk_idx}))
            start += max((end - start) - min_chunk_size, 1)
            chunk_idx += 1

    else:
        raise ValueError("File must be .xlsx, .csv, .json, or .docx")

    return docs

def chunk_text_to_docs(text: str, meta: Dict = None,
                       min_words: int = 100, max_words: int = 500,
                       overlap: int = 50) -> List[Document]:
    meta = meta or {}
    words = text.split()
    docs = []
    start = 0
    chunk_idx = 1

    while start < len(words):
        end = min(start + max_words, len(words))
        if end - start < min_words and len(words) - start >= min_words:
            end = start + min_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunk_meta = meta.copy()
        chunk_meta.update({"chunk_index": chunk_idx})
        docs.append(Document(page_content=chunk_text, metadata=chunk_meta))
        start += max((end - start) - overlap, 1)
        chunk_idx += 1

    return docs

import cohere
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings



# ---------- کمکی برای id سازی ----------
def make_doc_id(source: str) -> str:

    h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"doc_{h}"

def make_chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_chunk{idx:03d}"

# ---------- Cohere client + batching embeddings ----------
class CohereClient:
    def __init__(self, api_key: str, model: str = None, input_type: str | None = None, batch_size: int = None, retries: int = None):
        if not api_key:
            raise ValueError("COHERE_API_KEY is empty. Set COHERE_API_KEY environment variable or pass key.")
        self.client = cohere.Client(api_key)
        self.model = model or settings.EMBED_MODEL
        self.input_type = input_type or settings.EMBED_INPUT_TYPE
        self.batch_size = batch_size or settings.COHERE_BATCH_SIZE
        self.retries = retries or settings.EMBED_RETRY

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embedding با batching. بازگشت لیست بردارها به ترتیب texts."""
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            attempt = 0
            while True:
                try:
                    resp = self.client.embed(
                        model=self.model,
                        texts=batch,
                        input_type=self.input_type
                    )
                    # resp.embeddings: list[list[float]]
                    all_embs.extend(resp.embeddings)
                    break
                except Exception as e:
                    attempt += 1
                    logger.warning(f"Cohere embed error (attempt {attempt}/{self.retries}): {e}")
                    if attempt >= self.retries:
                        raise
                    time.sleep(1 + attempt * 1.5)
        return all_embs

# ---------- wrapper مناسب برای LangChain (برای FAISS.from_documents) ----------
class CohereEmbeddingWrapper(Embeddings):
    def __init__(self, cohere_client: CohereClient):
        self.cohere_client = cohere_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.cohere_client.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

from __future__ import annotations
import os
import time
import hashlib
import logging
from typing import List, Dict, Optional, Iterable
from dataclasses import dataclass, field

# ---------- توابع تبدیل و ذخیره (فایل/کرال صفحات) ----------
def docs_for_file(file_path: str) -> List[Document]:

    try:
        docs = chunk_file(file_path)
    except NameError:
        raise RuntimeError("chunk_file function not found. باید تابع chunk_file را در همان محیط تعریف کرده باشی.")
    doc_id = make_doc_id(os.path.basename(file_path))
    out_docs: List[Document] = []
    for idx, doc in enumerate(docs, start=1):
        meta = dict(doc.metadata or {})
        meta.setdefault("doc_id", doc_id)
        meta.setdefault("id", make_chunk_id(doc_id, idx))
        out_docs.append(Document(page_content=doc.page_content, metadata=meta))
    return out_docs

def docs_for_crawled_pages(crawled_pages: Iterable[Dict]) -> List[Document]:

    out_docs: List[Document] = []
    for page in crawled_pages:
        source = page.get("url") or page.get("source") or page.get("title", "unknown")
        doc_id = make_doc_id(source)
        meta = {
            "doc_id": doc_id,
            "id": make_chunk_id(doc_id, 1),
            "url": page.get("url"),
            "title": page.get("title")
        }
        out_docs.append(Document(page_content=page.get("content", ""), metadata=meta))
    return out_docs

# ---------- ساخت و ذخیره FAISS ----------
def build_and_save_faiss(documents: List[Document], cohere_client: CohereClient, db_path: str = None, overwrite: bool = True) -> FAISS:

    db_path = db_path or settings.FAISS_DIR
    logger.info("Preparing embedding wrapper...")
    embedding_wrapper = CohereEmbeddingWrapper(cohere_client)

    missing = [d for d in documents if not (d.metadata and "id" in d.metadata and "doc_id" in d.metadata)]
    if missing:
        raise ValueError("بعضی Document ها metadata id/doc_id ندارند. قبل از ادامه آنها را تنظیم کن.")

    logger.info("Building FAISS index from documents (this will call Cohere to embed)...")
    vectorstore = FAISS.from_documents(documents, embedding_wrapper)

    # ذخیره محلی
    if overwrite and os.path.isdir(db_path):
        logger.info("Overwriting existing FAISS folder: %s", db_path)
    vectorstore.save_local(db_path)
    logger.info("FAISS saved to %s", db_path)
    return vectorstore

# ---------- بارگذاری FAISS ----------
def load_faiss(db_path: str = None, cohere_client: Optional[CohereClient] = None, allow_dangerous_deserialization: bool = False) -> FAISS:
    db_path = db_path or settings.FAISS_DIR
    if cohere_client is None:
        cohere_client = CohereClient(settings.COHERE_API_KEY)
    embedding_wrapper = CohereEmbeddingWrapper(cohere_client)
    if not os.path.isdir(db_path):
        raise FileNotFoundError(f"FAISS folder not found: {db_path}")
    logger.info("Loading FAISS from %s (allow_dangerous_deserialization=%s)", db_path, allow_dangerous_deserialization)
    vectorstore = FAISS.load_local(db_path, embedding_wrapper, allow_dangerous_deserialization=allow_dangerous_deserialization)
    return vectorstore

# ---------- جستجو و حذف ----------
def query_vectorstore(vectorstore: FAISS, query: str, k: int = 5):
    results = vectorstore.similarity_search(query, k=k)
    return results

def delete_by_doc_id(vectorstore: FAISS, doc_id: str):

    ids_to_remove = []
    for internal_id, doc in vectorstore.docstore._dict.items():
        md = doc.metadata or {}
        if md.get("doc_id") == doc_id:
            ids_to_remove.append(internal_id)
    if not ids_to_remove:
        logger.info("No documents found for doc_id=%s", doc_id)
        return 0
    vectorstore.remove_ids(ids_to_remove)
    logger.info("Removed %d vectors for doc_id=%s", len(ids_to_remove), doc_id)
    return len(ids_to_remove)

# ---------- ابزارهای کمکی برای نمایش ----------
def sample_vectors_info(vectorstore: FAISS, n: int = 3):

    keys = list(vectorstore.docstore._dict.keys())
    out = []
    for i, k in enumerate(keys[:n]):
        doc = vectorstore.docstore._dict[k]
        try:
            vec = vectorstore.index.reconstruct(int(k))  # در بعضی پیاده‌سازی‌ها key ها int هستند
        except Exception:
            try:
                vec = vectorstore.index.reconstruct(i)
            except Exception:
                vec = None
        out.append({"internal_id": k, "metadata": doc.metadata, "vector_head": vec[:10] if vec is not None else None})
    return out

# ---------- مثال استفاده (تابع‌های سطح بالا) ----------
def process_file_to_faiss_safe(file_path: str, db_path: Optional[str] = None):
    co_client = CohereClient(settings.COHERE_API_KEY, model=settings.EMBED_MODEL, input_type=settings.EMBED_INPUT_TYPE)
    docs = docs_for_file(file_path)
    return build_and_save_faiss(docs, co_client, db_path=db_path)

def save_crawled_pages_to_faiss_safe(crawled_pages: List[Dict], db_path: Optional[str] = None):
    co_client = CohereClient(settings.COHERE_API_KEY, model=settings.EMBED_MODEL, input_type=settings.EMBED_INPUT_TYPE)
    docs = docs_for_crawled_pages(crawled_pages)
    return build_and_save_faiss(docs, co_client, db_path=db_path)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("vector_pipeline")

# crawler = JinaCrawler(api_key=settings.JINA_API_KEY, max_depth=1, delay=0.5)
# start_url = "https://iiid.tech/"   # آدرس تستت رو بذار
# crawler.crawl(start_url)

# print("Pages crawled:", len(crawler.pages))
# if crawler.pages:
#     print("First page title:", crawler.pages[0].get("title"))
#     print("First page url:", crawler.pages[0].get("url"))
#     print("First page content (preview):", crawler.pages[0].get("content")[:300])

# vs = save_crawled_pages_to_faiss_safe(crawler.pages, db_path="faiss_db")
# print("Saved to FAISS.")

