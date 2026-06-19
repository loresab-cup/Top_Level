import os
import ast
import hashlib

import chromadb
from pathlib import Path
from search_engine import get_text_embedding

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def extract_code_with_metadata(file_path: str) -> list:
    """
    Парсит Python-файл и извлекает все функции и классы.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        print(f"[Пропуск] Файл {file_path} содержит синтаксическую ошибку.")
        return []

    results = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node) or ""

            try:
                code_chunk = ast.get_source_segment(source_code, node)
            except Exception:
                code_chunk = ""

            if not code_chunk:
                continue

            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
            lines_str = f"{start_line}-{end_line}"

            results.append({
                "file_path": file_path,
                "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                "name": node.name,
                "lines": lines_str,
                "docstring": docstring,
                "code": code_chunk
            })

    return results

def prepare_text_for_embedding(code_data):
    """Обогащение текста перед векторизацией"""
    parts = []
    parts.append(f'{code_data["type"]}: {code_data[name]}')
    if code_data['docstring']:
        parts.append(f'Description: {code_data["docstring"]}')
    parts.append(f'Code:{code_data['code']}')

    return '\n'.join(parts)


def index_codebase(directory: str):
    """Индексатор"""
    print(f"Индексация директории: {directory}")

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="code_snippets",
        metadata={"description": "Python code snippets"}
    )

    total_chunks = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            print(f"[Обработка] {file_path}")

            code_chunks = extract_code_with_metadata(file_path)

            if not code_chunks:
                continue

            documents = []
            embeddings = []
            metadatas = []
            ids = []

            for chunk in code_chunks:
                #Обогащенный текст
                enriched_text = prepare_text_for_embedding(chunk)

                #вектор
                vector = get_text_embedding(enriched_text)
                if not vector:
                    continue

                #Метаданные
                metadata = {
                    "file_path": chunk["file_path"],
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "lines": chunk["lines"],
                    "docstring": chunk["docstring"][:500] if chunk["docstring"] else "",
                    "code": chunk["code"]
                }

                #айди через хеш
                unique_key = f"{chunk['file_path']}:{chunk['name']}:{chunk['lines']}"
                chunk_id = hashlib.md5(unique_key.encode("utf-8")).hexdigest()


                documents.append(enriched_text)
                embeddings.append(vector)
                metadatas.append(metadata)
                ids.append(chunk_id)

            if documents:
                collection.upsert(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                total_chunks += len(documents)
                print(f"  → Добавлено/обновлено {len(documents)} фрагментов")

    print(f"\n Всего проиндексировано: {total_chunks} фрагментов кода")
    print(f"База данных: ./chroma_db")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python index.py <папка_с_кодом>")
        print("Пример: python index.py codebase_python")
        sys.exit(1)

    target_directory = sys.argv[1]

    if not os.path.isdir(target_directory):
        print(f"Ошибка: директория '{target_directory}' не существует.")
        sys.exit(1)

    index_codebase(target_directory)