import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import os
import json
import pandas as pd
import time

# Импортируем функцию сравнения и расчёта из твоего оригинального score.py
from score import parse_chunk_id

st.title("Метрики и оценка качества RAG")
st.caption("Инструмент автоматического тестирования и расчета точности Precision@5 на основе набора валидации")


# Вспомогательная функция расчёта (адаптированная под UI логика score.py)
def evaluate_rag_system(preds, grounds):
    ground_dict = {q["question_id"]: q for q in grounds}
    per_question_details = []

    total_score = 0.0
    count = 0

    for p in preds:
        q_id = p["question_id"]
        if q_id not in ground_dict:
            continue

        g = ground_dict[q_id]
        true_ids = g["correct_chunk_ids"]
        pred_ids = p["top_5_chunks"][:5]

        # Парсим правильные ответы
        true_parsed = []
        for tid in true_ids:
            parsed = parse_chunk_id(tid)
            if parsed: true_parsed.append(parsed)

        n_correct = 0
        for pid in pred_ids:
            p_parsed = parse_chunk_id(pid)
            if not p_parsed: continue

            p_path, p_name, p_line = p_parsed

            # Логика матчинга с допуском ±2 строки из score.py
            match_found = False
            for t_path, t_name, t_line in true_parsed:
                if p_path == t_path and p_name == t_name:
                    if abs(p_line - t_line) <= 2:
                        match_found = True
                        break
            if match_found:
                n_correct += 1

        score = n_correct / 5.0
        total_score += score
        count += 1

        per_question_details.append({
            "ID Вопроса": q_id,
            "Вопрос": g.get("query", ""),
            "Сложность": g.get("difficulty", "unknown"),
            "Язык": g.get("language", "unknown"),
            "Успешных чанков": f"{n_correct} из 5",
            "Precision@5": score
        })

    mean_p5 = total_score / count if count > 0 else 0
    return mean_p5, per_question_details


# --- Кнопка запуска тестирования ---
if st.button("Запустить валидацию и расчёт метрик", type="primary"):

    if not os.path.exists("eval_questions.json"):
        st.error("Ошибка: Файл `eval_questions.json` не найден в корне проекта!")
    else:
        with st.spinner("Генерация свежих предиктов и расчёт точности..."):
            try:
                # Шаг 1: Автоматически запускаем генерацию результатов generate_results.py
                import generate_results

                generate_results.main()
                time.sleep(1)
            except Exception as e:
                st.warning(f"Консольный запуск generate_results вызвал предупреждение, читаем текущие результаты: {e}")

            # Проверяем наличие результатов
            if os.path.exists("results.json"):
                with open("eval_questions.json", "r", encoding="utf-8") as f:
                    ground_truth = json.load(f)
                with open("results.json", "r", encoding="utf-8") as f:
                    predictions = json.load(f)

                # Считаем метрики
                mean_score, details = evaluate_rag_system(predictions, ground_truth)

                # Отрисовываем красивый дашборд
                st.success("Расчёт успешно завершён!")

                # Крупная карточка с главной метрикой
                st.metric(label="Итоговый Mean Precision@5", value=f"{mean_score:.3f}")

                # Создаем DataFrame для графиков и таблиц
                df = pd.DataFrame(details)

                # Группировка данных для визуализации
                st.subheader("Аналитика в разрезе категорий")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**По уровню сложности:**")
                    diff_df = df.groupby("Сложность")["Precision@5"].mean().reset_index()
                    st.bar_chart(data=diff_df, x="Сложность", y="Precision@5", use_container_width=True)

                with col2:
                    st.markdown("**По языку запроса:**")
                    lang_df = df.groupby("Язык")["Precision@5"].mean().reset_index()
                    st.bar_chart(data=lang_df, x="Язык", y="Precision@5", use_container_width=True)

                # Интерактивная таблица с деталями по каждому вопросу
                st.divider()
                st.subheader("Детальные результаты поколлекционно")
                st.dataframe(
                    df,
                    column_config={
                        "Precision@5": st.column_config.ProgressColumn(
                            "Precision@5",
                            help="Точность попадания ответов ретривера в топ-5",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("Файл `results.json` не был создан. Проверьте работоспособность `generate_results.py`.")
else:
    st.info("Нажмите на кнопку выше, чтобы прогнать тестовые вопросы через поисковый движок и рассчитать Precision@5.")