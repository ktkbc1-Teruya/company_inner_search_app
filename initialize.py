from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import logging, os, streamlit as st
import constants as ct

# 仮の load_data_sources 関数（実装に応じて修正してください）
def load_data_sources():
    # ここでデータソースをロードして Document オブジェクトのリストを返す
    # 例: return [Document(page_content="...", metadata={...}), ...]
    return []

# 仮の adjust_string 関数（必要に応じて実装を変更してください）
def adjust_string(s):
    # ここで文字列の調整処理を行う（例: 改行コードの統一など）
    return s

def initialize_retriever():
    """
    画面読み込み時にRAGのRetrieverを作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みなら何もしない
    if "retriever" in st.session_state:
        return
    
    # データソースをロード
    docs_all = load_data_sources()

    # 文字列調整（Windows対策）
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 埋め込みモデルとチャンク分割器
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # ✅ 永続化ディレクトリを設定
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir
    )

    # ✅ Chroma DB の作成または読み込み
    db = Chroma(
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=persist_dir
    )

    # ✅ 初回のみドキュメント追加して保存
    if db._collection.count() == 0:
        db.add_documents(splitted_docs)
        db.persist()
        logger.info("✅ 初回起動: ドキュメントを追加して保存しました")
    else:
        logger.info("✅ 既存のChroma DBをロードしました")

    # Retriever化
    st.session_state.retriever = db.as_retriever(
        search_kwargs={"k": ct.RETRIEVER_TOP_K}
    )
