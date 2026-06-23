import streamlit as st
import os
import chromadb
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer
import socket
import subprocess
import time


# Автозапуск Ollama
def ensure_ollama_is_running():
    """
    Проверяет, открыт ли порт Ollama (11434).
    Если порт закрыт, автоматически запускает процесс Ollama в фоновом режиме.
    """
    ollama_port = 11434
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)

    try:
        # Пробуем подключиться к локальному порту Ollama
        s.connect(("127.0.0.1", ollama_port))
        s.close()
    except (socket.error, socket.timeout):
        # Если подключиться не удалось, значит сервис выключен
        st.info("Локальный сервис Ollama не активен. Запускаю автоматически в фоновом режиме...")
        try:
            # Запускаем фоновый процесс 'ollama serve'
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Даем модели 3-4 секунды, чтобы инициализировать сокет и считать конфиги
            time.sleep(3.5)
            st.success("Ollama успешно запущена!")
        except FileNotFoundError:
            st.error(
                "Ошибка: Команда 'ollama' не найдена в системе. Убедитесь, что Ollama установлена и добавлена в PATH.")


# Запускаем проверку при каждом старте или обновлении страницы Streamlit
ensure_ollama_is_running()


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

        # ОБЯЗАТЕЛЬНО запрашиваем и документы (documents) тоже!
        data = collection.get(include=["embeddings", "metadatas", "documents"])

        if not data['ids']:
            return []
        records = []
        for i in range(len(data['ids'])):
            record = data['metadatas'][i].copy()
            record['embedding'] = data['embeddings'][i]
            record['code'] = data['documents'][i]  # СРОЧНО ДОБАВЛЯЕМ САМ КОД КУСКА!
            records.append(record)
        return records
    except Exception as e:
        st.error(f"Ошибка загрузки БД: {e}")
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
