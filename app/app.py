import streamlit as st

from data_loader import get_deepseek_key, load_data, load_lemma_forms
from tabs.corpus_tab import render_corpus_tab
from tabs.neologisms_tab import render_neologisms_tab
from tabs.word_tab import render_word_tab
from ui_helpers import show_mayak_index_help


st.set_page_config(page_title="Mayak-2D Prototype", layout="wide")
st.title("Mayak Digit")
st.subheader("Прототип цифрового словаря В. В. Маяковского")

full_corpus = load_data()
lemmas_forms = load_lemma_forms()
deepseek_key = get_deepseek_key()

search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)
count_stopwords = st.sidebar.checkbox("Учитывать служебные слова", value=False)

if full_corpus:
    min_year = min(item["year_finished"] for item in full_corpus)
    max_year = max(item["year_finished"] for item in full_corpus)
else:
    min_year = max_year = 0
    st.error("Корпус не загружен. Проверьте файл data/database.parquet.")

year_range = st.sidebar.slider(
    "Период написания",
    min_year,
    max_year,
    (min_year, max_year),
)

compare_periods = st.sidebar.checkbox("Добавить второй период для сравнения контекстов", value=False)
year_range_2 = None
if compare_periods:
    year_range_2 = st.sidebar.slider(
        "Период написания (для сравнения)",
        min_year,
        max_year,
        (min_year, max_year),
    )

with st.sidebar.expander("Настройки LLM"):
    model_source = st.radio(
        "Модель анализа:",
        ["Локальная (Ollama)", "DeepSeek 4 (API)"],
        index=1,
        help="Ollama требует скачивания модели локально. DeepSeek отправляет запрос через интернет.",
    )

with st.sidebar.expander("Настройки весов (Индекс Маяка)"):
    decay_distance = st.slider(
        "Затухание от расстояния",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Коэффициент затухания для слов, находящихся дальше от таргета.",
    )
    decay_sents = st.slider(
        "Между предложениями",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Коэффициент затухания при переходе к следующему предложению.",
    )
    decay_brks = st.slider(
        "Между разрывами строки (_BRK_)",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Коэффициент затухания за перенос строки или 'лесенку'.",
    )

st.sidebar.divider()
if st.sidebar.button("ℹ️ Что такое индекс Маяка?", use_container_width=True):
    show_mayak_index_help()

tab_search, tab_corpus, tab_neologisms = st.tabs(["🔍 Анализ слова", "📊 Статистика корпуса", "📝 Неологизмы (beta)"])

with tab_search:
    render_word_tab(
        full_corpus=full_corpus,
        lemma_forms=lemmas_forms,
        search_word=search_word,
        year_range=year_range,
        compare_periods=compare_periods,
        year_range_2=year_range_2,
        window_size=window_size,
        count_stopwords=count_stopwords,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
        model_source=model_source,
        deepseek_key=deepseek_key,
    )

with tab_corpus:
    render_corpus_tab(full_corpus=full_corpus, min_year=min_year, max_year=max_year)

with tab_neologisms:
    render_neologisms_tab(full_corpus=full_corpus)
