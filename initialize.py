"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import WebBaseLoader
import constants as ct


# Streamlit Cloud の Secrets からキーを取得
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini"
)

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_session_id():
    """
    セッションIDを生成して st.session_state に保存
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_retriever():
    """
    画面読み込み時にRAGのRetrieverを作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    docs_all = load_data_sources()

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    splitted_docs = text_splitter.split_documents(docs_all)

    # ✅ DuckDB バックエンドを明示的に設定
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None
    )

    db = Chroma(
        embedding_function=embeddings,
        client_settings=client_settings
    )

    db.add_documents(splitted_docs)

    st.session_state.retriever = db.as_retriever(
        search_kwargs={"k": ct.RETRIEVER_TOP_K}
    )


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み
    """
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        web_docs_all.extend(web_docs)

    docs_all.extend(web_docs_all)
    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイルの拡張子に応じて適切なローダーでファイルを読み込み、docs_allに追加する
    """
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()

    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    """
    if type(s) is not str:
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    return s
