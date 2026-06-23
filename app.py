import streamlit as st
import os
import chromadb
import socket
import subprocess
import time
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

# Настройка окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Сервис Ollama (Автозапуск) ---
def ensure_ollama_is_running():
    ollama_port = 11434
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", ollama_port))
        s.close()
    except (socket.error, socket.timeout):
        st.info("Информационное сообщение: Локальный сервис Ollama не активен. Выполняется фоновый запуск...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3.5)
            st.success("Сервис Ollama успешно запущен.")
        except FileNotFoundError:
            st.error("Критическая ошибка: Команда 'ollama' не найдена в системе. Проверьте переменную PATH.")


ensure_ollama_is_running()

# Конфигурация страницы
st.set_page_config(page_title="CodeLens - Поиск по коду", layout="wide")
st.title("CodeLens")
st.caption("Система семантического поиска и анализа исходного кода")


# --- Автоматическая загрузка базы данных ---
@st.cache_resource
def load_db_records():
    if not os.path.exists("./chroma_db"):
        return []
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("code_snippets")
        data = collection.get(include=["embeddings", "metadatas", "documents"])

        if not data or not data['ids']:
            return []

        records = []
        for i in range(len(data['ids'])):
            record = data['metadatas'][i].copy()
            record['embedding'] = data['embeddings'][i]
            record['code'] = data['documents'][i]
            records.append(record)
        return records
    except Exception as e:
        st.error(f"Ошибка при чтении базы данных: {e}")
        return []


# Проверка состояния сессии и загрузка данных на старте
if 'db' not in st.session_state or st.session_state.db is None:
    with st.spinner("Выполняется автоматическое подключение к базе данных..."):
        st.session_state.db = load_db_records()

if 'results' not in st.session_state:
    st.session_state.results = None
if 'answer' not in st.session_state:
    st.session_state.answer = None

# --- Боковая панель управления (Sidebar) ---
with st.sidebar:
    st.markdown("### Управление системой")

    # Кнопка обновления данных
    if st.button("Обновить базу данных", use_container_width=True):
        st.cache_resource.clear()
        with st.spinner("Перезагрузка данных..."):
            st.session_state.db = load_db_records()
        st.success("Данные успешно обновлены.")

    st.divider()

    # Индикатор статуса подключения
    if st.session_state.db:
        st.info(f"Статус: Подключено\n\nПроиндексировано фрагментов: {len(st.session_state.db)}")
    else:
        st.warning("Статус: База данных не найдена. Запустите скрипт index.py")

    st.divider()
    use_llm = st.checkbox("Включить генерацию ответа ИИ", value=True)

# --- Интерфейс поисковых запросов ---
user_query = st.text_input(
    "Введите технический вопрос или ключевые слова:",
    placeholder="Например: как в проекте создаётся токен доступа?"
)

if st.button("Выполнить поиск", type="primary") and user_query:
    if not st.session_state.db:
        st.error("Поиск отклонен: база данных не загружена.")
        st.stop()

    with st.spinner("Выполняется анализ кодовой базы..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)

        if st.session_state.results:
            if use_llm:
                with st.spinner("Генерация аналитического ответа..."):
                    st.session_state.answer = generate_llm_answer(user_query, st.session_state.results)
            else:
                st.session_state.answer = None
        else:
            st.warning("По вашему запросу совпадений не обнаружено. Измените формулировку.")

# --- Отображение результатов анализа ---
if st.session_state.results:
    results = st.session_state.results

    # Блок ответа локальной языковой модели
    if st.session_state.answer:
        st.divider()
        st.markdown("### Аналитический ответ ассистента")
        st.markdown(st.session_state.answer)

    # Блок вывода найденных фрагментов кода
    st.divider()
    st.markdown("### Результаты поиска (Top-5)")

    for i, result in enumerate(results):
        # Название функции, путь и процент релевантности в строгом текстовом формате
        header_text = f"[{i + 1}] {result['name']} | Путь: {result['file_path']} (Релевантность: {result['score']}%). "

        with st.expander(header_text):
            st.code(result['code'], language="python")
            if result.get('docstring'):
                st.markdown(f"**Документация:** {result['docstring']}")