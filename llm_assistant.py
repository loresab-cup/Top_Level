import requests


def generate_llm_answer(user_query: str, search_results: list) -> str:
    """
    Генерирует связный технический ответ на основе найденных фрагментов кода.
    Использует локально запущенную модель через бесплатную программу Ollama.
    """
    # Если поиск ничего не выдал, то и анализировать нечего
    if not search_results:
        return "Фрагменты кода не найдены. Невозможно составить ответ."

    # 1. Собираем контекст: склеиваем тексты пяти найденных функций в один блок
    context_pieces = []
    for index, res in enumerate(search_results, start=1):
        context_pieces.append(
            f"Фрагмент №{index} (Файл: {res['file_path']}, Имя: {res['name']}):\n{res['code']}"
        )
    context_text = "\n\n".join(context_pieces)

    # 2. Формируем четкие правила поведения для языковой модели (Промпт)
    system_instruction = (
        "Ты — AI-ассистент CodeLens, опытный тимлид и эксперт по архитектуре кода. "
        "Используя предоставленные фрагменты кода, ответь на вопрос пользователя. "
        "Отвечай кратко, профессионально, строго по делу и только на русском языке. "
        "Если в коде нет ответа на вопрос, прямо напиши: 'В предоставленном коде нет информации для ответа'."
    )

    # Объединяем инструкцию, код и сам вопрос
    full_prompt = (
        f"{system_instruction}\n\n"
        f"НАЙДЕННЫЙ КОД ДЛЯ АНАЛИЗА:\n{context_text}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}"
    )

    # 3. Отправляем запрос на локальный сервер Ollama
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b",  # Модель mistral отлично справляется с кодом
        "prompt": full_prompt,
        "stream": False  # Просим вернуть ответ сразу целиком
    }

    try:
        # Делаем HTTP-запрос к Ollama (таймаут 30 секунд, так как локальной модели нужно время подумать)
        response = requests.post(ollama_url, json=payload, timeout=30)

        if response.status_code == 200:
            # Извлекаем текст ответа из JSON структуры Ollama
            return response.json().get("response", "Ошибка: ИИ вернул пустой ответ.")
        return f"Ошибка сервера Ollama: код ответа {response.status_code}"

    except requests.exceptions.ConnectionError:
        # Если пользователь забыл включить Ollama на компьютере
        return "Внимание: Локальный ИИ-ассистент недоступен. Убедитесь, что программа Ollama запущена на вашем ПК."
    except Exception as e:
        return f"Произошла непредвиденная ошибка при обращении к ИИ: {str(e)}"