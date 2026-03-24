import streamlit as st
import pandas as pd
import re
from collections import Counter
from pymorphy3 import MorphAnalyzer

# Инициализация
morph = MorphAnalyzer()
STOP_WORDS = {'и', 'в', 'во', 'не', 'на', 'с', 'со', 'что', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'от', 'по', 'же', 'вы', 'за', 'бы', 'потому', 'весь', 'кто', 'это', 'я', 'мы', 'они'}

# Словарь для перевода тегов pymorphy в понятные человеку
POS_MAP = {
    'NOUN': 'Существительное', 'VERB': 'Глагол', 'ADJF': 'Прилагательное',
    'ADJS': 'Краткое прил.', 'COMP': 'Компаратив', 'INFN': 'Инфинитив',
    'PRTF': 'Причастие', 'PRTS': 'Краткое прич.', 'GRND': 'Деепричастие',
    'NUMR': 'Числительное', 'ADVB': 'Наречие', 'NPRO': 'Местоимение',
    'PRED': 'Предикатив', 'PREP': 'Предлог', 'CONJ': 'Союз', 'PRCL': 'Частица'
}

@st.cache_data # Чтобы не перегружать файл каждый раз
def load_data():
    return pd.read_csv('data/database.csv')

def analyze_word(df, target_word, window=5):
    target_norm = morph.parse(target_word.lower())[0].normal_form
    contexts = []
    neighbors = []
    neighbor_pos = []
    years = []
    
    for _, row in df.iterrows():
        text = str(row['formatted_text'])
        title = row['text_name']
        year = row['year_finished']
        
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        
        for i, token in enumerate(tokens):
            if token[0].isalpha():
                parsed = morph.parse(token.lower())[0]
                if parsed.normal_form == target_norm:
                    # Сбор контекста
                    start = max(0, i - window)
                    end = min(len(tokens), i + window + 1)
                    ctx = " ".join(tokens[start:end])
                    contexts.append({"Контекст": ctx.replace(token, f"**{token.upper()}**"), "Произведение": title, "Год": year})
                    years.append(year)
                    
                    # Сбор соседей и их POS-тегов
                    for j in range(start, end):
                        if i == j: continue
                        t = tokens[j]
                        if t[0].isalpha():
                            p = morph.parse(t.lower())[0]
                            if p.normal_form not in STOP_WORDS:
                                neighbors.append(p.normal_form)
                                pos_name = POS_MAP.get(p.tag.POS, 'Другое')
                                neighbor_pos.append(pos_name)
                                
    return contexts, Counter(neighbors), Counter(neighbor_pos), Counter(years)

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Маяковский: Лексический профиль", layout="wide")
st.title("📊 Лексический профиль поэта")

df = load_data()
search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)

if search_word:
    contexts, top_neighbors, pos_dist, year_dist = analyze_word(df, search_word, window_size)
    
    if not contexts:
        st.warning("Слово не найдено в корпусе.")
    else:
        # Верхняя панель: Статистика
        col1, col2, col3 = st.columns(3)
        col1.metric("Употреблений", len(contexts))
        col2.write("**Топ семантических соседей:**")
        col2.write(", ".join([f"{w} ({c})" for w, c in top_neighbors.most_common(5)]))
        
        # Графики
        st.divider()
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("Окружающие части речи")
            pos_df = pd.DataFrame(pos_dist.items(), columns=['Часть речи', 'Кол-во'])
            st.bar_chart(pos_df.set_index('Часть речи'))
            
        with c_right:
            st.subheader("Распределение по годам")
            year_df = pd.DataFrame(year_dist.items(), columns=['Год', 'Частота']).sort_values('Год')
            st.line_chart(year_df.set_index('Год'))

        # Таблица контекстов
        st.subheader("Примеры употребления")
        st.table(pd.DataFrame(contexts))