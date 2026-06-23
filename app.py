import streamlit as st
import os
import chromadb
import socket
import subprocess
import time
import sys
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

# Настройка переменных окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Автозапуск Ollama (без лишних сообщений) ---
def ensure_ollama_is_running():
    ollama_port = 11434
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", ollama_port))
        s.close()
    except (socket.error, socket.timeout):
        with st.spinner("Запуск Ollama..."):
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(3)
            except FileNotFoundError:
                st.error("Ollama не найден. Установите его или отключите LLM-режим.")

ensure_ollama_is_running()

st.set_page_config(page_title="CodeLens - Поиск по коду", layout="wide")
st.title("CodeLens")
st.caption("Система семантического поиска и анализа исходного кода")

# --- Работа с базой данных ---
def fetch_records_from_storage():
    """Загружает записи из ChromaDB, если коллекция существует."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection("code_snippets")
        except Exception:
            return None  # коллекция не создана

        data = collection.get(include=["embeddings", "metadatas", "documents"])
        if not data or not data['ids']:
            return None

        records = []
        for i in range(len(data['ids'])):
            record = data['metadatas'][i].copy()
            record['embedding'] = data['embeddings'][i]
            record['code'] = data['documents'][i]
            records.append(record)
        return records
    except Exception as e:
        st.error(f"Ошибка чтения БД: {e}")
        return None

def reindex_codebase(directory: str):
    """Запускает индексацию и возвращает True при успехе."""
    if not os.path.isdir(directory):
        st.error(f"Директория '{directory}' не существует.")
        return False

    with st.spinner(f"Индексация {directory} ..."):
        try:
            proc = subprocess.run(
                [sys.executable, "index.py", directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300
            )
            if proc.returncode != 0:
                st.error(f"Ошибка индексации:\n{proc.stderr}")
                return False
            st.success("Индексация завершена.")
            return True
        except subprocess.TimeoutExpired:
            st.error("Индексация заняла слишком много времени.")
            return False
        except Exception as e:
            st.error(f"Не удалось запустить index.py: {e}")
            return False

def evaluate_precision():
    """Запускает generate_results.py и score.py, возвращает текст отчёта."""
    if not os.path.exists("eval_questions.json"):
        return "Файл eval_questions.json не найден."

    # Генерация results.json
    with st.spinner("Генерация результатов..."):
        gen = subprocess.run(
            [sys.executable, "generate_results.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if gen.returncode != 0:
            return f"Ошибка generate_results: {gen.stderr}"

    # Вычисление Precision@5
    with st.spinner("Расчёт Precision@5..."):
        score = subprocess.run(
            [sys.executable, "score.py", "--predictions", "results.json", "--questions", "eval_questions.json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if score.returncode != 0:
            return f"Ошибка score.py: {score.stderr}"

    return score.stdout

# --- Инициализация состояния ---
if 'db' not in st.session_state:
    st.session_state.db = fetch_records_from_storage()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'answer' not in st.session_state:
    st.session_state.answer = None
# ИЗМЕНЕНИЕ: директория по умолчанию теперь ./parsing_folder
if 'directory' not in st.session_state:
    st.session_state.directory = "./parsing_folder"
if 'precision_report' not in st.session_state:
    st.session_state.precision_report = None

# --- Боковая панель ---
with st.sidebar:
    st.markdown("### Управление системой")

    # Выбор директории
    new_dir = st.text_input(
        "Путь к папке с .py файлами",
        value=st.session_state.directory,
        help="Относительный или абсолютный путь. Пример: ./gymhero"
    )
    if new_dir != st.session_state.directory:
        st.session_state.directory = new_dir

    # Кнопка индексации
    if st.button("Индексировать / Переиндексировать", use_container_width=True):
        if reindex_codebase(st.session_state.directory):
            st.session_state.db = fetch_records_from_storage()
            st.session_state.results = None
            st.session_state.answer = None
            st.session_state.precision_report = None
            st.rerun()

    st.divider()

    # Кнопка оценки точности
    if st.button("Оценить Precision@5", use_container_width=True):
        report = evaluate_precision()
        st.session_state.precision_report = report
        st.rerun()

    st.divider()

    # Статус БД
    db_status = "Не подключено"
    if st.session_state.db is None:
        db_status = "База не создана"
    elif isinstance(st.session_state.db, list) and len(st.session_state.db) == 0:
        db_status = "База пуста"
    elif st.session_state.db:
        db_status = f"Подключено ({len(st.session_state.db)} фрагментов)"

    if "не создана" in db_status or "пуста" in db_status:
        st.warning(f"Статус: {db_status}\n\nНажмите «Индексировать» для создания.")
    else:
        st.info(f"Статус: {db_status}")

    st.divider()
    use_llm = st.checkbox("Включить генерацию ответа ИИ", value=True)

# --- Отображение отчёта о точности (с кнопкой закрытия) ---
if st.session_state.precision_report:
    st.divider()
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### Отчёт Precision@5")
    with col2:
        if st.button("✕ Закрыть", key="close_report", use_container_width=True):
            st.session_state.precision_report = None
            st.rerun()
    st.text(st.session_state.precision_report)

# --- Поисковый интерфейс ---
user_query = st.text_input(
    "Введите технический вопрос или ключевые слова:",
    placeholder="Например: как в проекте создаётся токен доступа?"
)

if st.button("Выполнить поиск", type="primary") and user_query:
    if st.session_state.db is None or len(st.session_state.db) == 0:
        st.error("База данных не готова. Сначала выполните индексацию.")
        st.stop()

    # Очищаем предыдущий отчёт о метрике при новом поиске
    st.session_state.precision_report = None
    st.session_state.results = None
    st.session_state.answer = None

    with st.spinner("Поиск..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)

    if not st.session_state.results:
        st.warning("Ничего не найдено. Попробуйте другой запрос.")

# --- Вывод результатов поиска ---
if st.session_state.results:
    st.divider()
    st.markdown("### Результаты поиска (Top-5)")

    for i, res in enumerate(st.session_state.results):
        header = f"[{i+1}] {res['name']} | {res['file_path']} (Релевантность: {res['score']}%)"
        with st.expander(header):
            st.code(res['code'], language="python")
            if res.get('docstring'):
                st.markdown(f"**Документация:** {res['docstring']}")

    # Генерация LLM-ответа (если включено)
    if use_llm and st.session_state.answer is None:
        with st.spinner("Генерация ответа ИИ..."):
            st.session_state.answer = generate_llm_answer(user_query, st.session_state.results)
            st.rerun()

    if st.session_state.answer:
        st.divider()
        st.markdown("### Аналитический ответ ассистента")
        st.markdown(st.session_state.answer)