import sys
import os

# Добавляем корневую директорию проекта (mayak-digit) в пути поиска модулей
# Доделать или убрать, если структура проекта изменится
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------------------
import json
import re
import numpy as np
import altair as alt
import streamlit as st
import pandas as pd
from collections import Counter
from src.analyzer import full_word_analysis, get_unique_synonyms, filter_synonyms_by_corpus, prepare_llm_prompt, synonyms_proximity_index, proximity_neighbours_for_synonyms, navec, calculate_delta_analysis
from src.text_utils import russian_stopwords
from dotenv import load_dotenv

# --------------------------------------------------------------

# Получаем API-ключи
load_dotenv()
claude_key = os.getenv("ANTHROPIC_API_KEY") # API-ключ для доступа к модели Claude от Anthropic
deepseek_key = os.getenv("DEEPSEEK_API_KEY") # API-ключ для доступа к DeepSeek

@st.cache_data
def load_data():
    df = pd.read_parquet('data/database.parquet')
    return df.to_dict('records')  # Список словарей

@st.cache_data
def load_lemma_forms():
    """
    Загружает словарь 'лемма -> все встреченные словоформы'.
    Кешируется через st.cache_data для использования в приложении.
    """
    forms_path = os.path.join('data', 'vocabulary_forms.json')

    if os.path.exists(forms_path):
        try:
            with open(forms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Ошибка при загрузке словаря словоформ: {e}")
            return {}
    else:
        st.error(f"Файл {forms_path} не найден. Запустите препроцессинг.")
        return {}


# --- Функции для статистики корпуса ---

@st.cache_data
def compute_general_metrics(corpus_records):
    """Вычисляет базовые метрики корпуса: тексты, токены, леммы, уникальные леммы."""
    total_texts = len(corpus_records)
    total_tokens = sum(len(item['tokens']) for item in corpus_records)
    all_lemmas = []
    for text in corpus_records:
        for sentence in text['lemmas_cleaned']:
            all_lemmas.extend(sentence)
    total_lemmas = len(all_lemmas)
    unique_lemmas = len(set(all_lemmas))
    texts_by_year = Counter(item['year_finished'] for item in corpus_records)
    return {
        'total_texts': total_texts,
        'total_tokens': total_tokens,
        'total_lemmas': total_lemmas,
        'unique_lemmas': unique_lemmas,
        'texts_by_year': dict(texts_by_year),
    }

@st.cache_data
def compute_frequency_dict(corpus_records, exclude_stopwords=True):
    """Возвращает Counter лемм по корпусу."""
    counter = Counter()
    for item in corpus_records:
        for sentence in item['lemmas_cleaned']:
            counter.update(sentence)
    if exclude_stopwords:
        for sw in russian_stopwords:
            if sw in counter:
                del counter[sw]
    return counter

@st.cache_data
def compute_vocabulary_growth(corpus_records):
    """Считает рост словарного запаса (накопленных уникальных лемм) по годам."""
    from collections import defaultdict
    by_year = defaultdict(list)
    for item in corpus_records:
        for sentence in item['lemmas_cleaned']:
            by_year[item['year_finished']].extend(sentence)

    sorted_years = sorted(by_year.keys())
    seen_lemmas = set()
    growth_data = []
    for year in sorted_years:
        lemmas_year = by_year[year]
        new_lemmas_count = len(set(lemmas_year) - seen_lemmas)
        seen_lemmas.update(lemmas_year)
        unique_in_year = len(set(lemmas_year))
        total_in_year = len(lemmas_year)
        growth_data.append({
            'Год': year,
            'Уникальных лемм накоплено': len(seen_lemmas),
            'Новых лемм': new_lemmas_count,
            'Всего лемм в году': total_in_year,
            'Type-Token Ratio': round(unique_in_year / total_in_year, 3) if total_in_year > 0 else 0,
        })
    return growth_data

@st.cache_data
def cached_get_unique_synonyms(word, top_n=20, depth=50):
    return get_unique_synonyms(word, top_n_to_return=top_n, search_depth=depth)

@st.cache_data
def compute_vector_map(corpus_records, top_n=100, exclude_stopwords=True, pca_base_size=300):
    """
    Возвращает DataFrame с 2D-координатами (PCA) для топ-N лемм корпуса.

    ⚠️  PCA вычисляется на pca_base_size словах, затем берётся подмножество top_n.
    Это гарантирует статичные координаты независимо от top_n!
    """
    # Гарантируем, что pca_base_size >= top_n
    if pca_base_size < top_n:
        pca_base_size = top_n

    freq_counter = compute_frequency_dict(corpus_records, exclude_stopwords=exclude_stopwords)
    # ⭐ Берём все_words_for_pca слов для вычисления PCA
    all_words_for_pca = [word for word, _ in freq_counter.most_common(pca_base_size)]

    words_in_navec = [(word, freq_counter[word]) for word in all_words_for_pca if word in navec]
    if len(words_in_navec) < 3:
        return None

    words, freqs = zip(*words_in_navec)
    matrix = np.array([navec[w] for w in words])

    # PCA через numpy (вычисляется на всех словах из pca_base_size)
    centered = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T

    # ⭐ Берём первые top_n из вычисленного набора
    return pd.DataFrame({
        'Слово': list(words)[:top_n],
        'x': coords[:top_n, 0],
        'y': coords[:top_n, 1],
        'Частота': list(freqs)[:top_n],
    })

def format_context_with_highlight(text):
    """
    Преобразует маркеры <<<form>>> в красивую подсветку для Streamlit (:red[]).
    """
    text = re.sub(r'<<<([^>]+)>>>', r':red[\1]', text)
    return text

def display_contexts_table_simple(contexts):
    """
    Отображает контексты в виде обычной таблицы без выделения.
    Поддерживает встроенную сортировку Streamlit.
    """
    if not contexts:
        st.info("Контексты не найдены.")
        return

    contexts_df = pd.DataFrame(contexts)
    contexts_df = contexts_df.sort_values(by='Год')
    contexts_df.index = range(1, len(contexts_df) + 1)
    st.dataframe(contexts_df, width='stretch')

def display_contexts_table_highlighted(contexts):
    """
    Отображает контексты в виде markdown-таблицы с красивым выделением ключевого слова.
    """
    if not contexts:
        st.info("Контексты не найдены.")
        return

    # Сортируем по году
    sorted_contexts = sorted(contexts, key=lambda x: x['Год'])

    # Строим markdown таблицу
    markdown_table = "| № | Контекст | Произведение | Год |\n"
    markdown_table += "|---|----------|--------------|-----|\n"

    for i, ctx in enumerate(sorted_contexts, 1):
        # Форматируем контекст с выделением
        formatted_text = format_context_with_highlight(ctx['Контекст'])
        # Экранируем | в контексте и в названии произведения чтобы не сломалась таблица
        formatted_text = formatted_text.replace('|', '\\|')
        title_escaped = ctx['Произведение'].replace('|', '\\|')
        markdown_table += f"| {i} | {formatted_text} | {title_escaped} | {ctx['Год']} |\n"

    st.markdown(markdown_table)

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Mayak-2D Prototype", layout="wide")

st.title("Mayak-2D")
st.subheader("Прототип цифрового словаря В. В. Маяковского")

full_corpus = load_data()
lemmas_forms = load_lemma_forms()

@st.cache_data
def cached_full_word_analysis(search_word, year_range, window_size,
                               decay_distance, decay_brks, decay_sents):
    
    filtered = [
        item for item in full_corpus
        if year_range[0] <= item['year_finished'] <= year_range[1]
    ]

    return full_word_analysis(
        filtered_corpus=filtered,
        target_word=search_word,
        window_size=window_size,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
        stopwords=russian_stopwords,
        lemma_forms=lemmas_forms
    )

# --- Боковая панель ---
search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)

count_stopwords = st.sidebar.checkbox('Учитывать служебные слова', value=False)

if full_corpus:
    min_year = min(item['year_finished'] for item in full_corpus)
    max_year = max(item['year_finished'] for item in full_corpus)
else:
    min_year = max_year = 0  # Вставить сюда предпреждение

year_range = st.sidebar.slider(
    "Период написания",
    min_year, max_year, (min_year, max_year)
)

compare_periods = st.sidebar.checkbox("Добавить второй период для сравнения контекстов (beta)", value=False)

if compare_periods:
    year_range_2 = st.sidebar.slider(
        "Период написания (для сравнения)",
        min_year, max_year, (min_year, max_year)
    )

with st.sidebar.expander("🤖 Настройки LLM"):
    model_source = st.radio(
    "Модель анализа:",
    ["Локальная (Ollama)", "DeepSeek 3.5 (API)", "API (Claude 3.5 Sonnet)"],
    help="DeepSeek и Claude требуют ключи в .env и интернет. Ollama требует скачивания модели локально."
    )

with st.sidebar.expander("⚙️ Настройки весов (Индекс Маяка)"):
    decay_distance = st.slider(
        "Затухание от расстояния",
        min_value=0.5, max_value=1.0, value=0.95, step=0.01,
        help="Коэффициент затухания для слов, находящихся дальше от таргета."
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

# --- Глобальные вкладки ---
tab_search, tab_corpus = st.tabs(["🔍 Анализ слова", "📊 Статистика корпуса"])

# ══════════════════════════════════════════════════════════════
# ВКЛАДКА 1: Анализ слова
# ══════════════════════════════════════════════════════════════
with tab_search:

    if search_word:

        search_word = search_word.strip().lower().replace('ё', 'е')

        found_lemma = next(
            (lemma for lemma in lemmas_forms if lemma.replace('ё', 'е') == search_word),
            None
        )
        if not found_lemma:
            found_lemma = next(
                (lemma for lemma, forms in lemmas_forms.items()
                 if any(f.replace('ё', 'е') == search_word for f in forms)),
                None
            )
        if found_lemma:
            target_word = found_lemma
            search_word = found_lemma
        else:
            st.warning("Слово не найдено в корпусе.")
            st.stop()

        filtered_corpus = [
            item for item in full_corpus
            if year_range[0] <= item['year_finished'] <= year_range[1]
        ]

        results = cached_full_word_analysis(
            search_word, year_range, window_size,
            decay_distance, decay_brks, decay_sents
        )

        # Анализ второго периода для сравнения (если включено)
        results_2 = None
        if compare_periods:
            if year_range == year_range_2:
                results_2 = results
            else:
                results_2 = cached_full_word_analysis(
                    search_word, year_range_2, window_size,
                    decay_distance, decay_brks, decay_sents
                )

        if not results:
            st.warning("Слово не найдено в корпусе.")

        else:
            # Извлекаем данные для удобства
            total_occurrences = results['total_occurrences']
            contexts = results['contexts']
            year_dist = results['year_dist']
            top_neighbors = results['window_neighbors']
            pos_dist = results['pos_dist']
            proximity_weights = results['proximity_weights']

            # --- УРОВЕНЬ 1.1: Заголовок со словом ---
            st.markdown(f"## Анализ слова: `{target_word}`")
            if compare_periods:
                st.caption(f"📊 Сравнение периодов: {year_range[0]} — {year_range[1]} vs {year_range_2[0]} — {year_range_2[1]}")
            else:
                st.caption(f"Период поиска: {year_range[0]} — {year_range[1]}")

            # --- УРОВЕНЬ 1.2: Синонимы (из словаря и встречающиеся в корпусе) ---
            st.subheader("Синонимы")
            synonyms = get_unique_synonyms(search_word, top_n_to_return=20, search_depth=50)
            synonyms_filtered = filter_synonyms_by_corpus(synonyms, filtered_corpus) # Фильтруем по корпусу

            if synonyms:
                if compare_periods:
                    st.info("Подсчет списка синонимов происходит без привязки к периоду (на основе общего векторного словаря и полного корпуса Маяковского)")
                show_coefficients = st.checkbox('Показать коэффициенты близости', value=True)
                if show_coefficients:
                    synonyms_str = ', '.join([f"{syn} ({score:.4f})" for syn, score in synonyms])
                    st.write(f"Синонимы по общему корпусу художественной литературы с коэффициентами близости: {synonyms_str}")
                else:
                    st.write(f"Синонимы по общему корпусу художественной литературы (без коэффициентов): {', '.join([syn for syn, score in synonyms])}")
                st.info("Список включает слова из общего векторного словаря, которые могут не встречаться в поэтических текстах Маяковского. Ниже — только те слова, которые действительно есть в корпусе")
                st.write(f"Синонимы, найденные в корпусе: {', '.join(synonyms_filtered)}")
            else:
                st.write("Синонимы не найдены или слово отсутствует в модели.")

            st.divider()

            # --- УРОВЕНЬ 2: Метрики и графики (один период или сравнение) ---
            if compare_periods and results_2:
                # Извлекаем данные для обоих периодов
                total_occurrences_2 = results_2['total_occurrences']
                year_dist_2 = results_2['year_dist']
                pos_dist_2 = results_2['pos_dist']

                # === ПЕРИОД 1 ===
                st.subheader(f"📍 Период {year_range[0]} — {year_range[1]}")
                col1_metric, col1_pos, col1_years = st.columns([1, 1.2, 1.2])

                with col1_metric:
                    st.metric("Всего употреблений", total_occurrences)

                with col1_pos:
                    st.caption("Частеречное окружение")
                    if count_stopwords:
                        pos_data = pos_dist['with_stopwords']
                    else:
                        pos_data = pos_dist['filtered']
                    pos_df = pd.DataFrame(pos_data.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df.set_index('Часть речи'), height=200)

                with col1_years:
                    st.caption("Динамика")
                    year_df = pd.DataFrame(year_dist.items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df.set_index('Год'), height=200)

                # === ПЕРИОД 2 ===
                st.subheader(f"📍 Период {year_range_2[0]} — {year_range_2[1]}")
                col2_metric, col2_pos, col2_years = st.columns([1, 1.2, 1.2])

                with col2_metric:
                    st.metric("Всего употреблений", total_occurrences_2)

                with col2_pos:
                    st.caption("Частеречное окружение")
                    if count_stopwords:
                        pos_data_2 = pos_dist_2['with_stopwords']
                    else:
                        pos_data_2 = pos_dist_2['filtered']
                    pos_df_2 = pd.DataFrame(pos_data_2.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df_2.set_index('Часть речи'), height=200)

                with col2_years:
                    st.caption("Динамика")
                    year_df_2 = pd.DataFrame(year_dist_2.items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df_2.set_index('Год'), height=200)

                # Дельта-метрики
                st.divider()
                occ_delta = total_occurrences_2 - total_occurrences
                occ_pct = occ_delta / max(total_occurrences, 1) * 100
                col_delta, _, _ = st.columns(3)
                with col_delta:
                    st.metric("Δ Употреблений", f"{occ_delta:+d}", f"{occ_pct:+.1f}%")
            else:
                # Режим одного периода: три колонки как было
                col_metric, col_pos, col_years = st.columns(3)

                with col_metric:
                    st.subheader("Статистика")
                    st.metric("Всего употреблений", total_occurrences)
                    st.write("Метод расчета: сумма всех вхождений леммы в выбранном периоде.")

                with col_pos:
                    st.subheader("Частеречное окружение")
                    if count_stopwords:
                        pos_data = pos_dist['with_stopwords']
                    else:
                        pos_data = pos_dist['filtered']
                    pos_df = pd.DataFrame(pos_data.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df.set_index('Часть речи'))

                with col_years:
                    st.subheader("Динамика")
                    year_df = pd.DataFrame(results['year_dist'].items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df.set_index('Год'))

            st.divider()

            # --- УРОВЕНЬ 3: Сравнение методов (на всю ширину) ---
            st.subheader("Семантические связи")

            if compare_periods and results_2:
                # Режим сравнения: семантические связи для обоих периодов
                top_neighbors_2 = results_2['window_neighbors']
                proximity_weights_2 = results_2['proximity_weights']

                # Рассчитываем дельта-анализ
                delta_analysis = calculate_delta_analysis(results, results_2, count_stopwords=count_stopwords)

                tab_window, tab_index, tab_graph, tab_delta = st.tabs(["🔲 Классическое окно", "🕸️ Индекс Маяка", "📊 Интерактивный граф", "📈 Дельта-анализ"])

                with tab_window:
                    # Классическое окно для обоих периодов
                    col_wnd_1, col_wnd_2 = st.columns(2)

                    with col_wnd_1:
                        st.caption(f"Период {year_range[0]}—{year_range[1]}")
                        if count_stopwords:
                            n_df = pd.DataFrame(top_neighbors['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                        else:
                            n_df = pd.DataFrame(top_neighbors['filtered'].most_common(10), columns=['Лемма', 'Частота'])
                        n_df.index = range(1, len(n_df) + 1)
                        st.table(n_df)

                    with col_wnd_2:
                        st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                        if count_stopwords:
                            n_df_2 = pd.DataFrame(top_neighbors_2['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                        else:
                            n_df_2 = pd.DataFrame(top_neighbors_2['filtered'].most_common(10), columns=['Лемма', 'Частота'])
                        n_df_2.index = range(1, len(n_df_2) + 1)
                        st.table(n_df_2)

                with tab_index:
                    # Индекс контекстуальной близости для обоих периодов
                    col_idx_1, col_idx_2 = st.columns(2)

                    with col_idx_1:
                        st.caption(f"Период {year_range[0]}—{year_range[1]}")
                        weights_df = pd.DataFrame(proximity_weights.most_common(10), columns=['Лемма', 'Индекс'])
                        if not weights_df.empty:
                            max_val = weights_df['Индекс'].max()
                            weights_df['Сила связи'] = weights_df['Индекс'] / max_val
                            weights_df.index = range(1, len(weights_df) + 1)
                            st.dataframe(
                                weights_df[['Лемма', 'Сила связи']],
                                column_config={
                                    "Сила связи": st.column_config.ProgressColumn(
                                        "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                    )
                                },
                                width='stretch'
                            )

                    with col_idx_2:
                        st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                        weights_df_2 = pd.DataFrame(proximity_weights_2.most_common(10), columns=['Лемма', 'Индекс'])
                        if not weights_df_2.empty:
                            max_val_2 = weights_df_2['Индекс'].max()
                            weights_df_2['Сила связи'] = weights_df_2['Индекс'] / max_val_2
                            weights_df_2.index = range(1, len(weights_df_2) + 1)
                            st.dataframe(
                                weights_df_2[['Лемма', 'Сила связи']],
                                column_config={
                                    "Сила связи": st.column_config.ProgressColumn(
                                        "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                    )
                                },
                                width='stretch'
                            )

                with tab_graph:
                    st.markdown("### 📊 Интерактивный граф семантических связей")
                    st.info("Отображение интерактивного графа будет реализовано в ближайших версиях.")

                with tab_delta:
                    st.markdown("### 📈 Анализ изменений семантического поля")

                    if delta_analysis is None:
                        st.warning("Нет данных для дельта-анализа.")
                    else:
                        # Появившиеся слова
                        col_app, col_dis = st.columns(2)

                        with col_app:
                            st.subheader("🟢 Топ появившихся слов")
                            if delta_analysis['appeared_words']:
                                app_df = pd.DataFrame(
                                    delta_analysis['appeared_words'],
                                    columns=['Слово', 'Индекс']
                                )
                                app_df.index = range(1, len(app_df) + 1)
                                st.dataframe(app_df.head(10), width='stretch')
                            else:
                                st.info("Нет новых слов.")

                        with col_dis:
                            st.subheader("🔴 Топ исчезнувших слов")
                            if delta_analysis['disappeared_words']:
                                dis_df = pd.DataFrame(
                                    delta_analysis['disappeared_words'],
                                    columns=['Слово', 'Индекс']
                                )
                                dis_df.index = range(1, len(dis_df) + 1)
                                st.dataframe(dis_df.head(10), width='stretch')
                            else:
                                st.info("Нет исчезнувших слов.")

                        st.divider()

                        # Изменяющиеся слова
                        st.subheader("🔄 Самые существенные изменения индекса контекстуальной близости")

                        if delta_analysis['changed_words']:
                            changed_viz_data = []
                            for item in delta_analysis['changed_words'][:10]:
                                changed_viz_data.append({
                                    'Слово': item['word'],
                                    'Индекс период 1': f"{item['index_1']:.3f}",
                                    'Индекс период 2': f"{item['index_2']:.3f}",
                                    'Δ Индекс': f"{item['index_delta']:+.3f}",
                                    'Δ %': f"{item['index_pct']:+.1f}%",
                                    'Статус': '📈' if item['status'] == 'growing' else ('📉' if item['status'] == 'declining' else '➡️')
                                })

                            changed_df = pd.DataFrame(changed_viz_data)
                            changed_df.index = range(1, len(changed_df) + 1)
                            st.dataframe(changed_df, width='stretch', hide_index=False)
                        else:
                            st.info("Нет изменяющихся слов.")
            else:
                # Режим одного периода
                tab_window, tab_index, tab_graph = st.tabs(["🔲 Классическое окно (Частота)", "🕸️ Индекс Маяка (Таблица)", "📊 Интерактивный граф"])

                with tab_window:
                    if count_stopwords:
                        n_df = pd.DataFrame(top_neighbors['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                    else:
                        n_df = pd.DataFrame(top_neighbors['filtered'].most_common(10), columns=['Лемма', 'Частота'])
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
                                    "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                )
                            },
                            width='stretch'
                        )

                with tab_graph:
                    st.markdown("### 📊 Интерактивный граф семантических связей")
                    st.info("Отображение интерактивного графа будет реализовано в ближайших версиях.")

            # Таблица контекстов
            st.write("### Контексты употребления")

            if compare_periods and results_2:
                # Режим сравнения: контексты для обоих периодов
                contexts_2 = results_2['contexts']

                col_ctx_1, col_ctx_2 = st.columns(2)

                with col_ctx_1:
                    st.subheader(f"Период {year_range[0]} — {year_range[1]} ({len(contexts)} контекстов)")
                    if contexts:
                        context_format = st.radio(
                            "Формат отображения (период 1):",
                            ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                            horizontal=True,
                            help="Выберите удобный способ просмотра контекстов",
                            key="ctx_fmt_1"
                        )
                        if context_format == "📝 Таблица (базовая)":
                            display_contexts_table_simple(contexts)
                        else:
                            display_contexts_table_highlighted(contexts)

                with col_ctx_2:
                    st.subheader(f"Период {year_range_2[0]} — {year_range_2[1]} ({len(contexts_2)} контекстов)")
                    if contexts_2:
                        context_format_2 = st.radio(
                            "Формат отображения (период 2):",
                            ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                            horizontal=True,
                            help="Выберите удобный способ просмотра контекстов",
                            key="ctx_fmt_2"
                        )
                        if context_format_2 == "📝 Таблица (базовая)":
                            display_contexts_table_simple(contexts_2)
                        else:
                            display_contexts_table_highlighted(contexts_2)
                    else:
                        st.info("Контексты не найдены в этом периоде.")
            else:
                # Режим одного периода
                if contexts:
                    context_format = st.radio(
                        "Формат отображения:",
                        ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                        horizontal=True,
                        help="Выберите удобный способ просмотра контекстов"
                    )

                    if context_format == "📝 Таблица (базовая)":
                        display_contexts_table_simple(contexts)
                    else:
                        display_contexts_table_highlighted(contexts)

    # --- БЛОК ИНТЕРПРЕТАЦИИ ЧЕРЕЗ LLM ---
    if search_word and results:
        if st.button("🚀 Запустить интерпретацию через LLM"):
            
            # Импорты перемещены сюда для ускорения процесса загрузки страницы
            
            import ollama
            import anthropic

            status_text = st.empty()

            with st.spinner("Собираем статистику для промпта... Пожалуйста, подождите."):

                # Считаем близость синонимов к таргету в этом периоде
                status_text.text("📊 Рассчитываем семантическую близость синонимов...")
                syn_prox_index = synonyms_proximity_index(target_word, synonyms_filtered, results['proximity_weights'])

                # Считаем контекстуальные связи для каждого синонима
                status_text.text("🕸️ Анализирую гравитационные поля синонимов (это может занять время)...")
                neighbors_for_syns = proximity_neighbours_for_synonyms(
                    synonyms_filtered,
                    filtered_corpus,
                    decay_distance, decay_brks, decay_sents,
                    stopwords=russian_stopwords
                )

                # Сборка промпта
                status_text.text("✍️ Формирую аналитическое досье для ИИ...")
                interpr_prompt = prepare_llm_prompt(
                    target_word=target_word,
                    synonyms=synonyms,
                    synonyms_filtered=synonyms_filtered,
                    syn_proximity=syn_prox_index,
                    neighbors_for_synonyms=neighbors_for_syns,
                    total_occurrences=results['total_occurrences'],
                    year_dist=results['year_dist']
                )

                # Убираем временный текст статуса перед выводом результата
                status_text.empty()

            # Наглядный вывод промпта для проверки
            st.subheader("Сгенерированный промпт для ИИ:")
            st.code(interpr_prompt, language="text")

            st.divider()
            st.subheader("📝 Филологический комментарий от LLM:")

            # --- ЛОГИКА ВЫБОРА МОДЕЛИ ---

            if model_source == "Локальная (Ollama)":
                response_container = st.empty()
                full_response = ""

                try:
                    stream = ollama.generate(model='llama3:8b', prompt=interpr_prompt, stream=True)
                    for chunk in stream:
                        full_response += chunk['response']
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)

                except Exception as e:
                    st.error(f"Ошибка при обращении к Ollama: {e}")
                    st.info("Убедитесь, что приложение Ollama запущено и модель llama3:8b скачана.")

            elif model_source == "DeepSeek 3.5 (API)":
                if not deepseek_key:
                    st.error("Ключ DeepSeek не найден в .env! Добавьте DEEPSEEK_API_KEY.")
                else:
                    from openai import OpenAI as DeepSeekClient
                    client_ds = DeepSeekClient(api_key=deepseek_key, base_url="https://api.deepseek.com")

                    with st.spinner("DeepSeek анализирует семантические поля..."):
                        try:
                            response = client_ds.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "Ты — аналитический инструмент для обработки корпусных данных. Обобщай и интерпретируй только те данные, которые тебе предоставлены. Не добавляй внешние знания, биографические сведения или литературоведческие теории, которых нет в переданных данных. Отвечай строго по заданной структуре, кратко и конкретно."
                                    },
                                    {
                                        "role": "user",
                                        "content": interpr_prompt
                                    }
                                ],
                                stream=False
                            )
                            st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Ошибка API DeepSeek: {e}")
                            st.info("Убедитесь, что ключ DEEPSEEK_API_KEY корректно настроен.")

            # elif model_source == "API (Claude 3.5 Sonnet)":
            #     if not claude_key:
            #         st.error("Ключ Anthropic не найден в .env!")
            #     else:
            #         client = anthropic.Anthropic(api_key=claude_key)
            #
            #         with st.spinner("Claude анализирует семантические поля..."):
            #
            #             try:
            #
            #                 message = client.messages.create(
            #                     model="claude-sonnet-4-6",
            #                     max_tokens=1024,
            #                     system="Ты — эксперт-филолог, специализирующийся на творчестве В. В. Маяковского. Ты работаешь над составлением цифрового словаря авторского языка",
            #                     messages=[
            #                         {
            #                             "role": "user",
            #                             "content": interpr_prompt}]
            #                 )
            #                 st.markdown(message.content[0].text)
            #
            #             except Exception as e:
            #                 st.error(f"Ошибка API Claude: {e}")
            #                 st.info("Убедитесь, что ключ Anthropic корректно настроен.")


# ══════════════════════════════════════════════════════════════
# ВКЛАДКА 2: Статистика корпуса
# ══════════════════════════════════════════════════════════════
with tab_corpus:
    st.markdown("## Статистика корпуса")

    # Фильтр по периоду внутри вкладки
    stats_year_range = st.slider(
        "Период",
        min_year, max_year, (min_year, max_year),
        key="stats_year_range"
    )

    filtered_corpus_stats = [
        item for item in full_corpus
        if stats_year_range[0] <= item['year_finished'] <= stats_year_range[1]
    ]

    if not filtered_corpus_stats:
        st.warning("Нет данных за выбранный период.")
    else:
        tab_corp_metrics, tab_corp_freq, tab_corp_growth = st.tabs([
            "📋 Общие метрики",
            "📖 Частотный словарь",
            "📈 Рост словаря",
        ])

        # --- 2.1: Общие метрики ---
        with tab_corp_metrics:
            metrics = compute_general_metrics(filtered_corpus_stats)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Произведений", metrics['total_texts'])
            col2.metric("Токенов", f"{metrics['total_tokens']:,}".replace(',', '\u00a0'))
            col3.metric("Лемм всего", f"{metrics['total_lemmas']:,}".replace(',', '\u00a0'))
            col4.metric("Уникальных лемм", f"{metrics['unique_lemmas']:,}".replace(',', '\u00a0'))

            st.divider()
            st.subheader("Тексты по годам")
            year_texts_df = pd.DataFrame(
                sorted(metrics['texts_by_year'].items()),
                columns=['Год', 'Текстов']
            ).set_index('Год')
            st.bar_chart(year_texts_df)

        # --- 2.2: Частотный словарь ---
        with tab_corp_freq:
            col_sw, col_n = st.columns([1, 2])
            with col_sw:
                exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="freq_exclude_sw")
            with col_n:
                top_n = st.slider("Топ-N слов", 10, 300, 50, key="freq_top_n")

            freq_counter = compute_frequency_dict(
                filtered_corpus_stats,
                exclude_stopwords=exclude_sw
            )
            total_lemmas_count = sum(freq_counter.values())
            top_lemmas = freq_counter.most_common(top_n)

            freq_df = pd.DataFrame(top_lemmas, columns=['Лемма', 'Частота'])
            freq_df['% от корпуса'] = (freq_df['Частота'] / total_lemmas_count * 100).round(3)
            freq_df.index = range(1, len(freq_df) + 1)

            col_table, col_chart = st.columns([1, 1.2])
            with col_table:
                st.dataframe(freq_df, width='stretch')
            with col_chart:
                st.bar_chart(freq_df.set_index('Лемма')['Частота'].head(30))

        # --- 2.3: Рост словарного запаса ---
        with tab_corp_growth:
            growth_data = compute_vocabulary_growth(filtered_corpus_stats)
            growth_df = pd.DataFrame(growth_data).set_index('Год')

            if not growth_df.empty:
                st.subheader("Накопленный словарный запас")
                st.caption("Сколько уникальных лемм встречено в корпусе к каждому году")
                st.line_chart(growth_df[['Уникальных лемм накоплено']])

                st.divider()
                st.subheader("Новых уникальных лемм в год")
                st.caption("Сколько ранее не встречавшихся лемм появилось в текстах каждого года")
                st.bar_chart(growth_df[['Новых лемм']])

                st.divider()
                st.subheader("Type-Token Ratio")
                st.caption("Отношение уникальных лемм к общему числу лемм в году")
                st.line_chart(growth_df[['Type-Token Ratio']])

        st.divider()

        st.title("Векторная карта самых частотных слов")

        col_vm_sw, col_vm_n = st.columns([1, 2])
        with col_vm_sw:
            vm_exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="vm_exclude_sw")
        with col_vm_n:
            vm_top_n = st.slider("Количество слов", 20, 300, 100, key="vm_top_n")

        map_df = compute_vector_map(filtered_corpus_stats, top_n=vm_top_n, exclude_stopwords=vm_exclude_sw, pca_base_size=max(300, vm_top_n))

        if map_df is None:
            st.warning("Недостаточно слов с векторными представлениями для построения карты.")
        else:
            points = alt.Chart(map_df).mark_circle(opacity=0.7).encode(
                x=alt.X('x:Q', axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
                y=alt.Y('y:Q', axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
                size=alt.Size('Частота:Q', scale=alt.Scale(range=[40, 400]), legend=None),
                color=alt.Color('Частота:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['Слово:N', 'Частота:Q'],
            )
            labels = alt.Chart(map_df).mark_text(dx=6, dy=-6, fontSize=11, align='left', color='white').encode(
                x='x:Q',
                y='y:Q',
                text='Слово:N',
                tooltip=['Слово:N', 'Частота:Q'],
            )
            st.altair_chart(
                (points + labels).properties(height=600).configure_view(strokeWidth=0),
                width='stretch',
            )
        
