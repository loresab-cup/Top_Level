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

    # Собираем записи базы данных в список словарей
    database_records = []
    for i in range(len(data['ids'])):
        record = data['metadatas'][i].copy()
        record['embedding'] = data['embeddings'][i]
        record['code'] = data['documents'][i]
        database_records.append(record)

    output_results = []

    # 3. Прогоняем каждый вопрос через твой поиск
    print("Генерация результатов для eval_questions.json...")
    for item in eval_data:
        q_id = item["question_id"]
        query_text = item.get("query") or item.get("text") or item.get("question")

        # Вызываем твой гибридный поиск (он вернет ТОП-5)
        top_snippets = search_top_code_snippets(query_text, database_records)

        # Собираем ID найденных кусков кода
        predicted_ids = []
        for snippet in top_snippets:
            # Исправляем дублирование пути: приводим к прямым слэшам и убираем лишнюю папку
            file_path = snippet['file_path'].replace("\\", "/")
            if file_path.startswith("gymhero/gymhero/"):
                file_path = file_path.replace("gymhero/gymhero/", "gymhero/", 1)

            # Берем первую строчку фрагмента кода
            start_line = snippet['lines'].split('-')[0]

            # Собираем итоговую строку по формату ТЗ и добавляем в список
            snippet_id = f"{file_path}:{snippet['name']}:{start_line}"
            predicted_ids.append(snippet_id)

        output_results.append({
            "question_id": q_id,
            "top_5_chunks": predicted_ids
        })

    # 4. Сохраняем итоговый файл результатов для жюри
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

    print("Успешно! Файл results.json создан. Теперь можно запускать python score.py")


if __name__ == "__main__":
    main()