import json
import os
import chromadb
from search_engine import search_top_code_snippets


def main():
    # 1. Загружаем вопросы от организаторов
    if not os.path.exists("eval_questions.json"):
        print("Ошибка: Положите файл eval_questions.json в корень проекта!")
        return

    with open("eval_questions.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 2. Подключаемся к нашей готовой базе данных
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("code_snippets")
    data = collection.get(include=["embeddings", "metadatas", "documents"])

    database_records = []
    for i in range(len(data['ids'])):
        record = data['metadatas'][i].copy()
        record['embedding'] = data['embeddings'][i]
        record['code'] = data['documents'][i]
        database_records.append(record)

    output_results = []

    print("Генерация результатов для eval_questions.json...")
    for item in eval_data:
        q_id = item["question_id"]
        query_text = item.get("query") or item.get("text") or item.get("question")

        top_snippets = search_top_code_snippets(query_text, database_records)

        predicted_ids = []
        for snippet in top_snippets:
            # Жесткая нормализация пути для Windows и любых форматов индексации
            file_path = snippet['file_path'].replace("\\", "/")

            # Убираем точки и лишние слэши в начале, если они есть
            if file_path.startswith("./"):
                file_path = file_path[2:]

            # Если папка gymhero продублировалась — срезаем дубль
            if file_path.startswith("gymhero/gymhero/"):
                file_path = file_path.replace("gymhero/gymhero/", "gymhero/", 1)

            # Если вдруг путь начинается не с gymhero (например, забыли при индексации)
            if not file_path.startswith("gymhero/"):
                file_path = "gymhero/" + file_path

            start_line = snippet['lines'].split('-')[0]
            snippet_id = f"{file_path}:{snippet['name']}:{start_line}"
            predicted_ids.append(snippet_id)

        output_results.append({
            "question_id": q_id,
            "top_5_chunks": predicted_ids
        })

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)
    print("Успешно сохранено в results.json")


if __name__ == "__main__":
    main()