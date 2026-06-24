import json
import os
import chromadb
from search_engine import search_top_code_snippets


def main():
    # Все пути считаем относительно папки со скриптом, а не рабочей директории
    script_dir = os.path.dirname(os.path.abspath(__file__))

    questions_file = os.path.join(script_dir, "eval_questions.json")
    if not os.path.exists(questions_file):
        print("Ошибка: положите файл eval_questions.json в папку проекта!")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    chroma_path = os.path.join(script_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection("code_snippets")
    except Exception:
        print("Ошибка: коллекция code_snippets не найдена. Запустите индексацию.")
        return

    data = collection.get(include=["embeddings", "metadatas", "documents"])

    database_records = []
    for i in range(len(data["ids"])):
        record = data["metadatas"][i].copy()
        record["embedding"] = data["embeddings"][i]
        record["code"] = data["documents"][i]
        database_records.append(record)

    output_results = []

    print(f"Генерация результатов для {len(eval_data)} вопросов...")
    for item in eval_data:
        q_id = item["question_id"]
        query_text = item.get("query") or item.get("text") or item.get("question")

        top_snippets = search_top_code_snippets(query_text, database_records)

        predicted_ids = []
        for snippet in top_snippets:
            # file_path уже хранится как относительный (gymhero/security.py),
            # поэтому никаких префиксов убирать не нужно
            file_path = snippet["file_path"].replace("\\", "/")
            start_line = snippet["lines"].split("-")[0]
            snippet_id = f"{file_path}:{snippet['name']}:{start_line}"
            predicted_ids.append(snippet_id)

        output_results.append({
            "question_id": q_id,
            "top_5_chunks": predicted_ids,
        })

    results_file = os.path.join(script_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

    print(f"Готово! {results_file} создан.")
    print("Запускайте: python score.py --predictions results.json --questions eval_questions.json")


if __name__ == "__main__":
    main()
