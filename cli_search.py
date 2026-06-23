import sys
import os
import chromadb
from search_engine import search_top_code_snippets


def load_database():
    """Загружает записи из ChromaDB в список словарей, как ожидает search_engine."""
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print(f"Ошибка: база данных не найдена по пути {db_path}. Запустите сначала python index.py")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("code_snippets")
        data = collection.get(include=["embeddings", "metadatas", "documents"])
    except Exception as e:
        print(f"Ошибка при загрузке базы данных: {e}")
        sys.exit(1)

    if not data or not data['ids']:
        print("База данных пуста. Проиндексируйте код с помощью index.py.")
        sys.exit(1)

    records = []
    for i in range(len(data['ids'])):
        record = data['metadatas'][i].copy()
        record['embedding'] = data['embeddings'][i]
        record['code'] = data['documents'][i]
        records.append(record)

    print(f"База данных загружена, записей: {len(records)}")
    return records


def print_results(results):
    """Выводит результаты поиска в читаемом формате."""
    if not results:
        print("Совпадений не найдено.")
        return

    print("\n=== РЕЗУЛЬТАТЫ ПОИСКА (Top-5) ===\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}  |  Релевантность: {r['score']}%")
        print(f"   Файл: {r['file_path']}")
        print(f"   Строки: {r['lines']}")
        if r.get('docstring'):
            print(f"   Описание: {r['docstring'][:150]}...")
        # Показываем первые 5 строк кода, можно расширить
        code_lines = r['code'].split('\n')
        preview = '\n'.join(code_lines[:5])
        if len(code_lines) > 5:
            preview += "\n   ... (показаны первые 5 строк)"
        print(f"   Код:\n{preview}")
        print("-" * 50)


def main():
    db_records = load_database()

    # Если передан аргумент командной строки – выполняем разовый поиск
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Запрос: {query}")
        results = search_top_code_snippets(query, db_records)
        print_results(results)
        return

    # Интерактивный режим
    print("\n=== CodeLens Консольный поиск ===")
    print("Введите запрос на русском или английском. Для выхода введите /exit или /quit.\n")

    while True:
        try:
            query = input("Введите запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query:
            continue

        if query.lower() in ("/exit", "/quit"):
            print("Выход.")
            break
        if query.lower() == "/help":
            print("Доступные команды: /exit, /quit, /help")
            continue

        results = search_top_code_snippets(query, db_records)
        print_results(results)


if __name__ == "__main__":
    main()
