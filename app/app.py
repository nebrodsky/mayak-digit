import sys
import os

# Добавляем корневую директорию проекта (mayak-digit) в пути поиска модулей
# Доделать или убрать, если структура проекта изменится
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------
import ast
import ollama # Для взаимодействия с локальной Ollama
import anthropic # Для взаимодействия с API Claude от Anthropic
import streamlit as st
import pandas as pd
from src.analyzer import full_word_analysis, get_unique_synonyms, filter_synonyms_by_corpus, prepare_llm_prompt, synonyms_proximity_index, proximity_neighbours_for_synonyms
from src.text_utils import morph, russian_stopwords
from dotenv import load_dotenv

# --------------------------------------------------------------

load_dotenv()
claude_key = os.getenv("ANTHROPIC_API_KEY") # API-ключ для доступа к модели Claude от Anthropic 

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join('data', 'database.csv'))

def prepare_full_corpus(df):
    """
    Превращает весь DF в список текстов, готовых для поиска.
    """
    full_corpus = []
    for _, row in df.iterrows():
        try:
            
            lemmatized_sentences = ast.literal_eval(row['lemmas']) # Парсим леммы один раз для всей сессии
            formatted_sentences = ast.literal_eval(row['formatted_sentences']) # Парсим отформатированные предложения
            full_corpus.append({
                'title': str(row['title']),
                'year_finished': int(row['year_finished']),
                'lemmas': lemmatized_sentences,
                'formatted_sentences': formatted_sentences,
                'raw_text': str(row['raw_text'])
            })
        except Exception as e:
            st.error(f"Ошибка при обработке строки с id: {e}")
            continue
    return full_corpus

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Mayak-2D Prototype", layout="wide")

st.title("Mayak-2D")
st.subheader("Прототип цифрового словаря В. В. Маяковского")

df = load_data()
full_corpus = prepare_full_corpus(df)

search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)

min_year = int(df['year_finished'].min())
max_year = int(df['year_finished'].max())

year_range = st.sidebar.slider(
    "Период написания",
    min_year, max_year, (min_year, max_year)
)
with st.sidebar.expander("🤖 Настройки LLM"):
    model_source = st.radio(
    "Модель анализа:",
    ["Локальная (Ollama)", "API (Claude 3.5 Sonnet)"],
    help="Claude требует ключ в .env и интернет. Ollama работает локально."
    )

with st.sidebar.expander("⚙️ Настройки весов (Индекс Маяка)"):
    decay_distance = st.slider(
        "Затухание от расстояния", 
        min_value=0.5, max_value=1.0, value=0.95, step=0.01,
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

if search_word:

    search_word = search_word.strip().lower() # Убираем лишние пробелы и приводим к нижнему регистру
    target_word = morph.parse(search_word)[0].normal_form # Лемматизируем для надежности

    filtered_corpus = [
        item for item in full_corpus 
        if year_range[0] <= item['year_finished'] <= year_range[1]
    ]

    results = full_word_analysis(
        filtered_corpus=filtered_corpus,
        target_word=target_word, 
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
        top_neighbors = results['window_neighbors']
        pos_dist = results['pos_dist']
        proximity_weights = results['proximity_weights']

        # --- УРОВЕНЬ 1.1: Заголовок со словом ---
        st.markdown(f"## Анализ слова: `{search_word.lower()}`")
        st.caption(f"Период поиска: {year_range[0]} — {year_range[1]}")

        # --- УРОВЕНЬ 1.2: Синонимы (из словаря и встречающиеся в корпусе) ---
        st.subheader("Синонимы")
        synonyms = get_unique_synonyms(search_word, top_n_to_return=20, search_depth=50)
        synonyms_filtered = filter_synonyms_by_corpus(synonyms, filtered_corpus) # Фильтруем по корпусу

        if synonyms:
            show_coefficients = st.checkbox('Показать коэффициенты близости', value=True)
            if show_coefficients:
                synonyms_str = ', '.join([f"{syn} ({score:.4f})" for syn, score in synonyms])
                st.write(f"Синонимы с коэффициентами близости: {synonyms_str}")
            else:
                st.write(f"Синонимы (без коэффициентов): {', '.join([syn for syn, score in synonyms])}")
            st.info("Список включает слова из общего векторного словаря, которые не встречаются в текстах Маяковского. Ниже — только те, что реально есть в корпусе.")
            st.write(f"Синонимы, найденные в корпусе: {', '.join(synonyms_filtered)}")
        else:
            st.write("Синонимы не найдены или слово отсутствует в модели.")

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
                            "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
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

# --- ТЕСТОВЫЙ БЛОК ДЛЯ ПРОВЕРКИ ПРОМПТА ---
if st.button("🚀 Запустить интерпретацию через LLM"):
    
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
            neighbors_for_synonyms=neighbors_for_syns
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

    elif model_source == "API (Claude 3.5 Sonnet)":
        if not claude_key:
            st.error("Ключ Anthropic не найден в .env!")
        else:
            client = anthropic.Anthropic(api_key=claude_key)

            with st.spinner("Claude анализирует семантические поля..."):

                try:

                    message = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        system="Ты — эксперт-филолог, специализирующийся на творчестве В. В. Маяковского. Ты рыботаешь над составлением цифрового словаря авторского языка",
                        messages=[
                            {
                                "role": "user",
                                "content": interpr_prompt}]
                    )
                    st.markdown(message.content[0].text)

                except Exception as e:
                    st.error(f"Ошибка API Claude: {e}")
                    st.info("Убедитесь, что ключ Anthropic корректно настроен.")