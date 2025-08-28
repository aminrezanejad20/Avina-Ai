
#%pip install -qU langchain-text-splitters

#!pip install --upgrade pip
#!pip install requests pandas unstructured langchain openpyxl python-docx tqdm

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

class Settings:
    EBOO_TOKEN = os.getenv("EBOO_TOKEN", "TPSHJFdOWcqR2I5wRgJkFloiBqmQBxwl1")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_71a9a199a9fd4109826a1f417ef04ef9Z6jA3xRAka0d4sa66V8S6q_0SIOf")
    MIN_WORDS = 100
    MAX_WORDS = 500
    CHUNK_OVERLAP = 50

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

#crawler = JinaCrawler(api_key=settings.JINA_API_KEY, max_depth=1)
#crawler.crawl("https://iiid.tech/")

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





