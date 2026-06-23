import streamlit as st

# Конфигурация страницы должна быть вызвана строго ОДИН раз здесь
st.set_page_config(page_title="CodeLens - Аналитика", layout="wide")

# Определяем страницы
search_page = st.Page("views/search_page.py", title="Поиск по коду")
metrics_page = st.Page("views/metrics_page.py", title="Метрики системы (RAG)")

# Инициализируем навигацию
pg = st.navigation([search_page, metrics_page])
pg.run()