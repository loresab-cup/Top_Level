import streamlit as st
import os
import chromadb
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="CodeLens - Поиск по коду", layout="wide")

st.title("CodeLens - Поиск по коду")

if 'db' not in st.session_state:
    st.session_state.db = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'answer' not in st.session_state:
    st.session_state.answer = None

@st.cache_resource
def load_db():
    try:
        if not os.path.exists("./chroma_db"):
            return []
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("code_snippets")
        data = collection.get(include=["embeddings", "metadatas"])
        if not data['ids']:
            return []
        records = []
        for i in range(len(data['ids'])):
            record = data['metadatas'][i].copy()
            record['embedding'] = data['embeddings'][i]
            records.append(record)
        return records
    except:
        return []

with st.sidebar:
    if st.button("Загрузить базу"):
        st.session_state.db = load_db()
        if st.session_state.db:
            st.success(f"Загружено {len(st.session_state.db)} фрагментов")
    
    if st.session_state.db is not None:
        st.metric("Фрагментов", len(st.session_state.db))
    
    use_llm = st.checkbox("Использовать ИИ", value=True)

user_query = st.text_input("Вопрос о коде:", placeholder="Например: как сделать авторизацию?")

if st.button("Найти") and user_query:
    if not st.session_state.db:
        st.warning("Загрузите базу данных")
        st.stop()
    
    with st.spinner("Поиск..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)
        
        if st.session_state.results:
            if use_llm:
                with st.spinner("Генерация ответа..."):
                    st.session_state.answer = generate_llm_answer(user_query, st.session_state.results)
            else:
                st.session_state.answer = None
        else:
            st.warning("Ничего не найдено")

if st.session_state.results:
    results = st.session_state.results
    
    if st.session_state.answer:
        st.divider()
        st.subheader("Ответ ИИ")
        st.markdown(st.session_state.answer)
    
    st.divider()
    st.subheader("Найденные фрагменты")
    
    for i, result in enumerate(results):
        with st.expander(f"{i+1}. {result['name']} ({result['score']}%)"):
            st.markdown(f"**Файл:** `{result['file_path']}`")
            st.markdown(f"**Тип:** {result['type']}")
            st.markdown(f"**Строки:** {result['lines']}")
            if result.get('docstring'):
                st.markdown(f"**Документация:** {result['docstring']}")
            st.code(result['code'], language="python")
