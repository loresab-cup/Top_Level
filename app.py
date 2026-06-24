import streamlit as st
import os
import chromadb
import socket
import subprocess
import time
import sys
from pathlib import Path
from search_engine import search_top_code_snippets
from llm_assistant import generate_llm_answer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Чтобы не зависеть от того, из какой папки запустили
PROJECT_DIR = Path(__file__).parent.resolve()
CHROMA_PATH = PROJECT_DIR / "chroma_db"


def ensure_ollama_is_running():
    # Проверяем, запущен ли Ollama на стандартном порту
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", 11434))
        s.close()
    except (socket.error, socket.timeout):
        with st.spinner("Запуск Ollama..."):
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(3)  # Даём время запуститься
            except FileNotFoundError:
                st.error("Ollama не найден. Установите его или отключите LLM-режим.")


ensure_ollama_is_running()

st.set_page_config(page_title="CodeLens - Поиск по коду", layout="wide")
st.title("CodeLens")
st.caption("Система семантического поиска и анализа исходного кода")


def fetch_records_from_storage():
    # Загружаем всё из ChromaDB, если она есть
    if not CHROMA_PATH.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            collection = client.get_collection("code_snippets")
        except Exception:
            return None

        data = collection.get(include=["embeddings", "metadatas", "documents"])
        if not data or not data["ids"]:
            return None

        records = []
        for i in range(len(data["ids"])):
            record = data["metadatas"][i].copy()
            record["embedding"] = data["embeddings"][i]
            record["code"] = data["documents"][i]
            records.append(record)
        return records if records else None
    except Exception as e:
        st.error(f"Ошибка чтения БД: {e}")
        return None


def resolve_directory(raw_path: str) -> Path:
    # Превращаем любой путь в абсолютный
    p = Path(raw_path.strip()).expanduser()
    if not p.is_absolute():
        p = (PROJECT_DIR / p).resolve()
    return p.resolve()


def reindex_codebase(raw_path: str) -> bool:
    target = resolve_directory(raw_path)

    if not target.is_dir():
        st.error(
            f"Директория не найдена: `{target}`\n\n"
            "Проверьте путь и попробуйте снова."
        )
        return False

    with st.spinner(f"Индексация: {target} ..."):
        try:
            proc = subprocess.run(
                [sys.executable, str(PROJECT_DIR / "index.py"), str(target)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 минут, обычно хватает
                cwd=str(PROJECT_DIR),
            )
            if proc.returncode != 0:
                st.error(f"Ошибка индексации:\n```\n{proc.stderr}\n```")
                return False

            st.success("Индексация завершена!")
            if proc.stdout:
                st.text(proc.stdout[-3000:])
            return True
        except subprocess.TimeoutExpired:
            st.error("Индексация заняла больше 5 минут. Попробуйте меньшую папку.")
            return False
        except Exception as e:
            st.error(f"Не удалось запустить index.py: {e}")
            return False


def evaluate_precision() -> str:
    # Прогоняем тестовые вопросы и считаем метрику
    if not (PROJECT_DIR / "eval_questions.json").exists():
        return "Файл eval_questions.json не найден в папке проекта."

    with st.spinner("Генерация results.json..."):
        gen = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "generate_results.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        if gen.returncode != 0:
            return f"Ошибка generate_results.py:\n{gen.stderr}"

    with st.spinner("Расчёт Precision@5..."):
        score_proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "score.py"),
                "--predictions", str(PROJECT_DIR / "results.json"),
                "--questions", str(PROJECT_DIR / "eval_questions.json"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        if score_proc.returncode != 0:
            return f"Ошибка score.py:\n{score_proc.stderr}"

    return score_proc.stdout


# Инициализация состояния сессии
# Не загружаем базу при старте, пусть пользователь сам нажмёт кнопку
if "db" not in st.session_state:
    st.session_state.db = None
if "results" not in st.session_state:
    st.session_state.results = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "directory" not in st.session_state:
    st.session_state.directory = ""
if "precision_report" not in st.session_state:
    st.session_state.precision_report = None

# Боковая панель
with st.sidebar:
    st.markdown("### Управление системой")

    new_dir = st.text_input(
        "Путь к папке с .py файлами",
        value=st.session_state.directory,
        placeholder="Например: C:\\projects\\gymhero или ./gymhero",
        help=(
            "Абсолютный или относительный путь к папке с исходным кодом.\n"
            "Примеры:\n"
            "• C:\\1All\\UNIK\\dataset_case\\gymhero\n"
            "• /home/user/projects/gymhero\n"
            "• ./gymhero"
        ),
    )
    if new_dir != st.session_state.directory:
        st.session_state.directory = new_dir

    if st.button("Индексировать / Переиндексировать", use_container_width=True):
        if not st.session_state.directory.strip():
            st.error("Введите путь к папке с кодом.")
        else:
            # Очищаем всё перед переиндексацией
            st.session_state.db = None
            st.session_state.results = None
            st.session_state.answer = None
            st.session_state.precision_report = None

            if reindex_codebase(st.session_state.directory):
                st.session_state.db = fetch_records_from_storage()
                st.rerun()

    st.divider()

    if st.button("Оценить Precision@5", use_container_width=True):
        if not st.session_state.db:
            st.error("Сначала выполните индексацию.")
        else:
            report = evaluate_precision()
            st.session_state.precision_report = report
            st.rerun()

    st.divider()

    # Показываем статус
    if not st.session_state.db:
        st.warning("Статус: база не создана\n\nВведите путь и нажмите «Индексировать».")
    else:
        st.info(f"Статус: подключено ({len(st.session_state.db)} фрагментов)")

    st.divider()
    use_llm = st.checkbox("Включить генерацию ответа ИИ", value=True)

# --- Отчёт Precision@5 ---
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

# Поиск
user_query = st.text_input(
    "Введите технический вопрос или ключевые слова:",
    placeholder="Например: как в проекте создаётся токен доступа?",
)

if st.button("Выполнить поиск", type="primary") and user_query:
    if not st.session_state.db:
        st.error("База данных не готова. Сначала выполните индексацию.")
        st.stop()

    st.session_state.precision_report = None
    st.session_state.results = None
    st.session_state.answer = None

    with st.spinner("Поиск..."):
        st.session_state.results = search_top_code_snippets(user_query, st.session_state.db)

    if not st.session_state.results:
        st.warning("Ничего не найдено. Попробуйте другой запрос.")

# Результаты
if st.session_state.results:
    st.divider()
    st.markdown("### Результаты поиска (Top-5)")

    for i, res in enumerate(st.session_state.results):
        header = (
            f"[{i+1}] {res['name']} | {res['file_path']} "
            f"(Релевантность: {res['score']}%)"
        )
        with st.expander(header):
            st.code(res["code"], language="python")
            if res.get("docstring"):
                st.markdown(f"**Документация:** {res['docstring']}")

    if use_llm and st.session_state.answer is None:
        with st.spinner("Генерация ответа ИИ..."):
            st.session_state.answer = generate_llm_answer(
                user_query, st.session_state.results
            )
            st.rerun()

    if st.session_state.answer:
        st.divider()
        st.markdown("### Аналитический ответ ассистента")
        st.markdown(st.session_state.answer)