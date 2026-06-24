import os
import ast
import chromadb
from search_engine import get_text_embedding

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# chroma_db всегда создаётся рядом с этим скриптом
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_db")


def extract_code_with_metadata(file_path: str, base_directory: str) -> list:
    """
    Парсит Python-файл и извлекает функции и классы.
    Методы класса получают имя ClassName.method_name —
    именно такой формат ожидает eval_questions.json.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        print(f"[Пропуск] {file_path} — синтаксическая ошибка")
        return []

    # Относительный путь от корня датасета — именно он идёт в chunk_id
    rel_path = os.path.relpath(file_path, base_directory).replace("\\", "/")

    results = []

    # Классы и их методы
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        class_name = node.name
        try:
            code_chunk = ast.get_source_segment(source_code, node)
        except Exception:
            code_chunk = ""

        if code_chunk:
            results.append({
                "file_path": rel_path,
                "type": "class",
                "name": class_name,
                "lines": f"{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                "docstring": ast.get_docstring(node) or "",
                "code": code_chunk,
            })

        # Методы внутри класса — имя вида ClassName.method_name
        for child in ast.walk(node):
            if not isinstance(child, ast.FunctionDef):
                continue
            try:
                method_code = ast.get_source_segment(source_code, child)
            except Exception:
                method_code = ""
            if not method_code:
                continue
            results.append({
                "file_path": rel_path,
                "type": "function",
                "name": f"{class_name}.{child.name}",
                "lines": f"{child.lineno}-{getattr(child, 'end_lineno', child.lineno)}",
                "docstring": ast.get_docstring(child) or "",
                "code": method_code,
            })

    # Функции верхнего уровня (не внутри класса)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            try:
                code_chunk = ast.get_source_segment(source_code, node)
            except Exception:
                code_chunk = ""
            if not code_chunk:
                continue
            results.append({
                "file_path": rel_path,
                "type": "function",
                "name": node.name,
                "lines": f"{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                "docstring": ast.get_docstring(node) or "",
                "code": code_chunk,
            })

    return results


def prepare_text_for_embedding(code_data: dict) -> str:
    """Обогащаем текст перед векторизацией: имя + docstring + код."""
    parts = [f'{code_data["type"]}: {code_data["name"]}']
    if code_data["docstring"]:
        parts.append(f'Description: {code_data["docstring"]}')
    parts.append(f'Code:{code_data["code"]}')
    return "\n".join(parts)


def index_codebase(directory: str):
    directory = os.path.abspath(directory)
    print(f"Индексация: {directory}")
    print(f"База данных: {CHROMA_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Удаляем старую коллекцию перед переиндексацией —
    # иначе старые записи накапливаются и счётчик растёт бесконечно
    try:
        client.delete_collection("code_snippets")
        print("[OK] Старая коллекция удалена, создаём новую.")
    except Exception:
        print("[OK] Старой коллекции нет, создаём с нуля.")

    collection = client.create_collection(
        name="code_snippets",
        metadata={"description": "Python code snippets"},
    )

    total_chunks = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            print(f"[Обработка] {file_path}")

            code_chunks = extract_code_with_metadata(file_path, directory)
            if not code_chunks:
                continue

            documents, embeddings, metadatas, ids = [], [], [], []

            for chunk in code_chunks:
                enriched_text = prepare_text_for_embedding(chunk)
                vector = get_text_embedding(enriched_text)
                if not vector:
                    continue

                metadata = {
                    "file_path": chunk["file_path"],
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "lines": chunk["lines"],
                    "docstring": chunk["docstring"][:500] if chunk["docstring"] else "",
                    "code": chunk["code"],
                }

                start_line = chunk["lines"].split("-")[0]
                # Формат: "gymhero/security.py:create_access_token:12"
                chunk_id = f"{chunk['file_path']}:{chunk['name']}:{start_line}"

                documents.append(enriched_text)
                embeddings.append(vector)
                metadatas.append(metadata)
                ids.append(chunk_id)

            if documents:
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_chunks += len(documents)
                print(f"  -> {len(documents)} фрагментов")

    print(f"\nВсего проиндексировано: {total_chunks} фрагментов")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python index.py <папка_с_кодом>")
        sys.exit(1)

    target_directory = sys.argv[1]

    if not os.path.isdir(target_directory):
        print(f"Ошибка: директория '{target_directory}' не существует.")
        sys.exit(1)

    index_codebase(target_directory)
