import sys
import os

# Добавляем корневую директорию проекта (mayak-digit) в пути поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.analyzer import full_word_analysis
from src.text_utils import russian_stopwords

@st.cache_data # Чтобы не перегружать файл каждый раз
def load_data():
    return pd.read_csv('data/database.csv')

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Mayak-2D Prototype", layout="wide")

st.title("Mayak-2D")
st.subheader("Прототип цифрового словаря В. В. Маяковского")

df = load_data()
search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)

min_year = int(df['year_finished'].min())
max_year = int(df['year_finished'].max())

year_range = st.sidebar.slider(
    "Период написания",
    min_year, max_year, (min_year, max_year)
)

with st.sidebar.expander("⚙️ Настройки весов (Индекс Маяка)"):
    decay_distance = st.slider(
        "Затухание от расстояния", 
        min_value=0.5, max_value=1.0, value=0.95, step=0.05,
        help="Коэффициент затухания для слов, находящихся дальше от целевого."
    )
    decay_sents = st.slider(
        "Между предложениями", 
        min_value=0.1, max_value=1.0, value=0.8, step=0.05,
        help="Коэффициент затухания при переходе к следующему предложению."
    )
    decay_brks = st.slider(
        "Между разрывами строки (_BRK_)", 
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
        help="Коэффициент затухания за перенос строки или 'лесенку'."
    )

# Фильтруем данные перед анализом
df_filtered = df[(df['year_finished'] >= year_range[0]) & (df['year_finished'] <= year_range[1])]

if search_word:
    # Вызываем новую всеобъемлющую функцию
    results = full_word_analysis(
        df_filtered, 
        search_word.lower(), 
        window_size=window_size, 
        decay_distance=decay_distance, 
        decay_brks=decay_brks, 
        decay_sents=decay_sents, 
        stopwords=russian_stopwords
    )
    
    if not results:
        st.warning("Слово не найдено в корпусе.")
    else:
        # Извлекаем данные для удобства
        total_occurrences = results['total_occurrences']
        contexts = results['contexts']
        year_dist = results['year_dist']
        top_neighbors = results['window_neighbors'] # Для левой таблицы
        pos_dist = results['pos_dist']
        proximity_weights = results['proximity_weights'] # Для новой правой таблицы

        # --- УРОВЕНЬ 1: Заголовок со словом ---
        st.markdown(f"## Анализ слова: `{search_word.lower()}`")
        st.caption(f"Период поиска: {year_range[0]} — {year_range[1]}")

        st.divider()

        # --- УРОВЕНЬ 2: Три колонки (Метрика, POS, График) ---
        col_metric, col_pos, col_years = st.columns(3)

        with col_metric:
            st.subheader("Статистика")
            st.metric("Всего употреблений", total_occurrences)
            st.write("Метод расчета: сумма всех вхождений леммы в выбранном периоде.")

        with col_pos:
            st.subheader("Части речи")
            pos_df = pd.DataFrame(results['pos_dist'].items(), columns=['Часть речи', 'Кол-во'])
            st.bar_chart(pos_df.set_index('Часть речи'))

        with col_years:
            st.subheader("Динамика")
            year_df = pd.DataFrame(results['year_dist'].items(), columns=['Год', 'Частота']).sort_values('Год')
            st.line_chart(year_df.set_index('Год'))

        st.divider()

        # --- УРОВЕНЬ 3: Сравнение методов (на всю ширину) ---
        st.subheader("Семантические связи")
        
        # Создаем вкладки, чтобы не загромождать экран
        tab_window, tab_index = st.tabs(["🔲 Классическое окно (Частота)", "🕸️ Индекс Маяка (Семантический вес)"])
        
        with tab_window:
            # Твоя старая таблица (по частоте в окне)
            n_df = pd.DataFrame(top_neighbors.most_common(10), columns=['Лемма', 'Частота'])
            n_df.index = range(1, len(n_df) + 1)
            st.table(n_df)
            
        with tab_index:
            
            weights_df = pd.DataFrame(proximity_weights.most_common(10), columns=['Лемма', 'Индекс'])

            if not weights_df.empty:
                max_val = weights_df['Индекс'].max()
                weights_df['Сила связи'] = weights_df['Индекс'] / max_val
                weights_df.index = range(1, len(weights_df) + 1)
            
                st.dataframe(
                    weights_df[['Лемма', 'Сила связи']],
                    column_config={
                        "Сила связи": st.column_config.ProgressColumn(
                            "Близость", format="%.2f", min_value=0, max_value=1
                        )
                    },
                    width='stretch'
                )

        # Таблица контекстов
        contexts_df = pd.DataFrame(contexts)

        st.write("### Контексты употребления")

        if not contexts_df.empty:
            contexts_df = contexts_df.sort_values(by='Год') # Сортируем по году для удобства
            contexts_df.index = range(1, len(contexts_df) + 1) # Индексация с 1 для удобства
            st.dataframe(contexts_df, width='stretch') # Показываем всю ширину таблицы