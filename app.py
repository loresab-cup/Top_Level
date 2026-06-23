import streamlit as st
import os
import chromadb
import socket
import subprocess
import time
import json
import sys
from pathlib import Path
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

# Настройка переменных окружения для подавления технических предупреждений
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Автозапуск локального сервиса Ollama ---
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

# Конфигурация интерфейса
st.set_page_config(page_title="CodeLens - Поиск по коду", layout="wide")
st.title("CodeLens")
st.caption("Система семантического поиска и интеллектуального анализа исходного кода")

# --- Низкоуровневое чтение данных из ChromaDB ---
def fetch_records_from_storage():
    """Загружает записи из ChromaDB, если коллекция существует."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        # Проверяем, есть ли коллекция
        try:
            collection = client.get_collection("code_snippets")
        except Exception:
            # Коллекция не существует – база ещё не создана
            return None  # вернём None, чтобы сигнализировать об отсутствии базы

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
        st.error(f"Ошибка при чтении базы данных: {e}")
        return None

# --- Функция переиндексации с заданной директорией ---
def reindex_codebase(directory: str):
    with st.spinner(f"Индексация директории {directory}..."):
        try:
            process = subprocess.run(
                ["python", "index.py", directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if process.returncode != 0:
                st.error(f"Ошибка индексации: {process.stderr}")
                return False
            st.success("Индексация завершена успешно.")
            return True
        except Exception as e:
            st.error(f"Не удалось запустить скрипт индексации: {e}")
            return False

# --- Функция оценки Precision@5 ---
def evaluate_precision():
    """Запускает generate_results.py и score.py, возвращает строку с результатами."""
    # Проверяем наличие eval_questions.json
    if not os.path.exists("eval_questions.json"):
        return "Ошибка: файл eval_questions.json не найден в корне проекта."

    # Сначала генерируем результаты
    with st.spinner("Генерация результатов для тестовых вопросов..."):
        gen_proc = subprocess.run(
            [sys.executable, "generate_results.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if gen_proc.returncode != 0:
            return f"Ошибка при генерации результатов: {gen_proc.stderr}"

    # Затем запускаем score.py
    with st.spinner("Вычисление Precision@5..."):
        score_proc = subprocess.run(
            [sys.executable, "score.py", "--predictions", "results.json", "--questions", "eval_questions.json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if score_proc.returncode != 0:
            return f"Ошибка при вычислении метрики: {score_proc.stderr}"

    # Возвращаем полный вывод score.py (он уже содержит таблицы и итоговую оценку)
    return score_proc.stdout

# --- Инициализация состояния сессии ---
if 'db' not in st.session_state:
    st.session_state.db = fetch_records_from_storage()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'directory' not in st.session_state:
    st.session_state.directory = "parsing_folder"   # значение по умолчанию
if 'precision_report' not in st.session_state:
    st.session_state.precision_report = None

# --- Боковая панель управления ---
with st.sidebar:
    st.markdown("### Управление системой")

    # Поле ввода пути к директории с кодом
    new_dir = st.text_input(
        "Путь к директории с Python-кодом",
        value=st.session_state.directory,
        help="Укажите относительный или абсолютный путь к папке, содержащей .py файлы."
    )
    if new_dir != st.session_state.directory:
        st.session_state.directory = new_dir

    # Кнопка переиндексации
    if st.button("Переиндексировать код", use_container_width=True):
        if os.path.isdir(st.session_state.directory):
            if reindex_codebase(st.session_state.directory):
                st.session_state.db = fetch_records_from_storage()
                st.session_state.results = None
                st.session_state.answer = None
                st.session_state.precision_report = None
                st.rerun()
        else:
            st.error(f"Директория '{st.session_state.directory}' не существует.")

    st.divider()

    # Кнопка оценки точности
    if st.button("Оценить Precision@5", use_container_width=True):
        report = evaluate_precision()
        st.session_state.precision_report = report
        st.rerun()

    st.divider()

    # Индикатор статуса базы данных
    if st.session_state.db:
        st.info(f"Статус: Подключено\n\nПроиндексировано фрагментов: {len(st.session_state.db)}")
    else:
        st.warning("Статус: База данных не загружена или пуста.")

    st.divider()
    use_llm = st.checkbox("Включить генерацию ответа ИИ", value=True)

# --- Основной интерфейс: отображение отчёта по метрике (если есть) ---
if st.session_state.precision_report:
    st.divider()
    st.markdown("### Отчёт о точности (Precision@5)")
    st.text(st.session_state.precision_report)

# --- Поисковый интерфейс ---
user_query = st.text_input(
    "Введите технический вопрос или ключевые слова:",
    placeholder="Например: как в проекте создаётся токен доступа?"
)

if st.button("Выполнить поиск", type="primary") and user_query:
    if not st.session_state.db:
        st.error("Поиск отклонен: база данных не инициализирована.")
        st.stop()

    st.session_state.results = None
    st.session_state.answer = None

    with st.spinner("Выполняется анализ кодовой базы..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)

    if not st.session_state.results:
        st.warning("По вашему запросу совпадений не обнаружено. Измените формулировку.")

# --- Отображение результатов поиска и генерация ИИ ---
if st.session_state.results:
    st.divider()
    st.markdown("### Результаты поиска (Top-5)")

    for i, result in enumerate(st.session_state.results):
        header_text = f"[{i + 1}] {result['name']} | Путь: {result['file_path']} (Релевантность: {result['score']}%). "
        with st.expander(header_text):
            st.code(result['code'], language="python")
            if result.get('docstring'):
                st.markdown(f"**Документация:** {result['docstring']}")

    if use_llm and st.session_state.answer is None:
        with st.spinner("Генерация аналитического ответа ИИ..."):
            st.session_state.answer = generate_llm_answer(user_query, st.session_state.results)
            st.rerun()

    if st.session_state.answer:
        st.divider()
        st.markdown("### Аналитический ответ ассистента")
        st.markdown(st.session_state.answer)