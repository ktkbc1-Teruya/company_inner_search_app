# initialize.py
from __future__ import annotations

import os
import logging
import unicodedata
import streamlit as st
from dotenv import load_dotenv

# LangChain / Chroma
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,      # PDF
    Docx2txtLoader,     # DOCX
    TextLoader,         # TXT
)
from langchain_community.document_loaders.csv_loader import CSVLoader

import constants as ct

# ── .env の読み込み ───────────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ── サポート拡張子 ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv":  lambda path: CSVLoader(path, encoding="utf-8"),
    ".txt":  lambda path: TextLoader(path, encoding="utf-8"),
}

# ── 文字列整形（日本語の文字化け・不可視文字ケア） ─────────────
def adjust_string(s):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    try:
        # Windows環境由来の文字を可能な範囲で丸める
        s = s.encode("cp932", "ignore").decode("cp932")
    except Exception:
        # 念のためUTF-8にフォールバック
        s = s.encode("utf-8", "ignore").decode("utf-8")
    return s.strip()

# ── データ読み込み ───────────────────────────────────────────────
def load_data_sources() -> list:
    """./data 配下の pdf/docx/csv/txt を再帰的に読み込み、整形して返す"""
    docs = []
    for root, _dirs, files in os.walk("./data"):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                loader_factory = SUPPORTED_EXTENSIONS[ext]
                file_path = os.path.join(root, file)
                loader = loader_factory(file_path) if callable(loader_factory) else loader_factory(file_path)

                loaded = loader.load()  # -> List[Document]
                # 本文/メタデータをここで整形（読み込み直後にやると安全）
                for d in loaded:
                    d.page_content = adjust_string(d.page_content)
                    # source などのパスも正規化
                    if "source" in d.metadata:
                        d.metadata["source"] = adjust_string(os.path.normpath(d.metadata["source"]))
                    for k in list(d.metadata.keys()):
                        d.metadata[k] = adjust_string(d.metadata[k])
                docs.extend(loaded)
    return docs

# ── Retriever 初期化（ここに全ロジックを閉じ込める！） ───────────
def initialize_retriever():
    """画面読み込み時に RAG の Retriever を作成"""
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでに作成済みなら何もしない
    if "retriever" in st.session_state:
        return

    # 1) ドキュメント読み込み
    docs_all = load_data_sources()

    # 2) チャンク分割 & 埋め込み器
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # 3) 新しい Chroma クライアント（PersistentClient）
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # 4) LangChain の Chroma（新クライアントを渡す）
    db = Chroma(
        client=client,
        collection_name="your_collection_name",
        embedding_function=embeddings,
    )

    # 5) 初回のみ投入・永続化
    try:
        need_init = (db._collection.count() == 0)  # noqa: SLF001（内部属性）
    except Exception:
        need_init = True

    if need_init:
        if splitted_docs:
            db.add_documents(splitted_docs)
            try:
                db.persist()
            except Exception:
                # 新実装では不要だが、後方互換のために best-effort
                pass
            logger.info("✅ 初回起動: ドキュメントを追加して保存しました")
        else:
            logger.warning("⚠️ 追加するドキュメントがありません")
    else:
        logger.info("✅ 既存の Chroma DB をロードしました")

    # 6) Retriever を Session に格納
    st.session_state.retriever = db.as_retriever(
        search_kwargs={"k": ct.RETRIEVER_TOP_K}
    )

# ── main.py から呼ぶエントリ ────────────────────────────────────
def initialize():
    """アプリ起動時の初期化処理"""
    initialize_retriever()
