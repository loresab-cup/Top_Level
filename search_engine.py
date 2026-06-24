import os
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

print("[Поиск] Загрузка модели эмбеддингов...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("[Готово] Модель загружена.")


def get_text_embedding(text: str) -> list:
    if not text or not text.strip():
        return []
    return embedding_model.encode(text, convert_to_numpy=True).tolist()


def calculate_cosine_similarity(vector_a: list, vector_b: list) -> float:
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Нулевой вектор — значит что-то пошло не так при индексации
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def search_top_code_snippets(user_query: str, database_records: list) -> list:
    """
    Гибридный поиск: косинусное сходство + текстовые бонусы за ключевые слова.
    Возвращает топ-5 результатов.
    """
    query_vector = get_text_embedding(user_query)
    if not query_vector:
        return []

    # Короткие слова (≤3 символа) как "как", "где", "что" не несут смысловой нагрузки
    query_words = [w for w in user_query.lower().split() if len(w) > 3]
    compiled_results = []

    for record in database_records:
        code_vector = record.get("embedding", None)

        # ChromaDB может вернуть list, numpy array или None — обрабатываем все случаи
        if code_vector is None:
            continue
        if isinstance(code_vector, (list, np.ndarray)) and len(code_vector) == 0:
            continue

        code_text = record.get("code", "").lower()
        func_name = record.get("name", "").lower()
        docstring = record.get("docstring", "").lower()

        semantic_score = calculate_cosine_similarity(query_vector, code_vector)

        keyword_bonus = 0.0
        for word in query_words:
            if word in func_name:
                # Если слово из запроса есть прямо в названии — весомый сигнал
                keyword_bonus += 0.15
            elif word in docstring:
                keyword_bonus += 0.08
            elif word in code_text:
                keyword_bonus += 0.05

        # Ограничиваем сверху, чтобы не выходить за [0, 1]
        final_score = min(semantic_score + keyword_bonus, 1.0)

        compiled_results.append({
            "file_path": record.get("file_path", "").replace("\\", "/"),
            "type": record.get("type", "unknown"),
            "name": record.get("name", ""),
            "lines": record.get("lines", "0-0"),
            "docstring": record.get("docstring", ""),
            "code": record.get("code", ""),
            "score": round(final_score * 100, 2)
        })

    compiled_results.sort(key=lambda x: x["score"], reverse=True)
    return compiled_results[:5]
