import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import os
import chromadb
import socket
import subprocess
import time
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_ollama_is_running():
    ollama_port = 11434
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", ollama_port))
        s.close()
    except (socket.error, socket.timeout):
        st.info("Локальный сервис Ollama не активен. Выполняется фоновый запуск...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3.5)
        except FileNotFoundError:
            st.error("Команда 'ollama' не найдена в системе.")


ensure_ollama_is_running()

st.title("CodeLens")
st.caption("Система семантического поиска и интеллектуального анализа исходного кода")


def fetch_records_from_storage():
    with st.spinner("Выполняется чтение и загрузка базы данных..."):
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("code_snippets")
            data = collection.get(include=["embeddings", "metadatas", "documents"])
            if not data or not data['ids']: return []
            records = []
            for i in range(len(data['ids'])):
                record = data['metadatas'][i].copy()
                record['embedding'] = data['embeddings'][i]
                record['code'] = data['documents'][i]
                records.append(record)
            return records
        except Exception as e:
            st.error(f"Ошибка чтения БД: {e}")
            return []


@st.cache_resource
def manage_database_state():
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        with st.spinner("Локальная база данных не обнаружена. Индексация исходного кода..."):
            try:
                subprocess.run(["python", "index.py", "gymhero"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                st.error(f"Не удалось запустить скрипт индексации: {e}")
                return []
    return fetch_records_from_storage()


if 'db' not in st.session_state or st.session_state.db is None:
    st.session_state.db = manage_database_state()

if 'results' not in st.session_state: st.session_state.results = None
if 'answer' not in st.session_state: st.session_state.answer = None

with st.sidebar:
    st.markdown("### Статус системы")
    if st.session_state.db:
        st.info(f"Подключение: Активно\n\nПроиндексировано фрагментов: {len(st.session_state.db)}")
    else:
        st.warning("Подключение: База данных пуста.")
    st.divider()
    use_llm = st.checkbox("Включить генерацию ответа ИИ", value=True)

user_query = st.text_input("Введите технический вопрос или ключевые слова:",
                           placeholder="Например: как в проекте создаётся токен доступа?")

if st.button("Выполнить поиск", type="primary") and user_query:
    if not st.session_state.db:
        st.error("База данных не инициализирована.")
        st.stop()
    st.session_state.results = None
    st.session_state.answer = None

    with st.spinner("Выполняется анализ кодовой базы..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)

if st.session_state.results:
    st.divider()
    st.markdown("### Результаты поиска (Top-5)")
    for i, result in enumerate(st.session_state.results):
        with st.expander(f"[{i + 1}] {result['name']} | Путь: {result['file_path']} ({result['score']}%)."):
            st.code(result['code'], language="python")
            if result.get('docstring'): st.markdown(f"**Документация:** {result['docstring']}")

    if use_llm and st.session_state.answer is None:
        with st.spinner("Генерация аналитического ответа ИИ..."):
            st.session_state.answer = generate_llm_answer(user_query, st.session_state.results)
            st.rerun()

    if st.session_state.answer:
        st.divider()
        st.markdown("### Аналитический ответ ассистента")
        st.markdown(st.session_state.answer)