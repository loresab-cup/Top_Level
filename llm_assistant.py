import requests


def generate_llm_answer(user_query: str, search_results: list) -> str:
    # Если ничего не нашли, то и отвечать нечем
    if not search_results:
        return "Фрагменты кода не найдены. Невозможно составить ответ."

    # Собираем найденный код в один большой текст для модели
    context_pieces = []
    for index, res in enumerate(search_results, start=1):
        context_pieces.append(
            f"Фрагмент №{index} (Файл: {res['file_path']}, Имя: {res['name']}):\n{res['code']}"
        )
    context_text = "\n\n".join(context_pieces)

    # Инструкция для модели, чтобы отвечала по делу
    system_instruction = (
        "Ты — AI-ассистент CodeLens, опытный тимлид и эксперт по архитектуре кода. "
        "Используя предоставленные фрагменты кода, ответь на вопрос пользователя. "
        "Отвечай кратко, профессионально, строго по делу и только на русском языке. "
        "Если в коде нет ответа на вопрос, прямо напиши: 'В предоставленном коде нет информации для ответа'."
    )

    # Склеиваем всё в один промпт
    full_prompt = (
        f"{system_instruction}\n\n"
        f"НАЙДЕННЫЙ КОД ДЛЯ АНАЛИЗА:\n{context_text}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}"
    )

    # Отправляем запрос в Ollama
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b",
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json().get("response", "Ошибка: ИИ вернул пустой ответ.")
        return f"Ошибка сервера Ollama: код ответа {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Внимание: Локальный ИИ-ассистент недоступен. Убедитесь, что программа Ollama запущена на вашем ПК."
    except Exception as e:
        return f"Произошла непредвиденная ошибка при обращении к ИИ: {str(e)}"