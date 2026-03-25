import re
import ast
from collections import Counter
from src.text_utils import morph, get_sentences, get_words

POS_MAP = {
    'NOUN': 'Существительное', 'VERB': 'Глагол', 'ADJF': 'Прилагательное',
    'ADJS': 'Краткое прил.', 'COMP': 'Компаратив', 'INFN': 'Инфинитив',
    'PRTF': 'Причастие', 'PRTS': 'Краткое прич.', 'GRND': 'Деепричастие',
    'NUMR': 'Числительное', 'ADVB': 'Наречие', 'NPRO': 'Местоимение',
    'PRED': 'Предикатив', 'PREP': 'Предлог', 'CONJ': 'Союз', 'PRCL': 'Частица'
}

# 1. Функция контекста и общей статистики
def get_occurrence_data(df, target_norm):
    """
    Возвращает: 
    - contexts: список словарей для нижней таблицы
    - year_dist: Counter для графика частотности
    - raw_data: структурированные данные для других функций (чтобы не парсить CSV дважды)
    """
    contexts = []
    year_dist = Counter()
    raw_data = []
    total_occurrences = 0

    for _, row in df.iterrows():
        try:
            sentences_lemmas = ast.literal_eval(row['lemmas'])
            sentences_raw = get_sentences(str(row['formatted_text']))
        except:
            continue
        
        text_occurrences = []
        for s_idx, lemmas in enumerate(sentences_lemmas):
            if target_norm in lemmas:
                indices = [i for i, l in enumerate(lemmas) if l == target_norm]
                text_occurrences.append((s_idx, indices))
                
                total_occurrences += len(indices)

                year_dist[row['year_finished']] += len(indices)

                # Достаем оригинал по индексу предложения s_idx
                if s_idx < len(sentences_raw):
                    # 1. Берем оригинал (здесь он со всеми пробелами и знаками)
                    raw_sentence = sentences_raw[s_idx].replace('_BRK_', ' / ').strip(' /–-—')
                    
                    # 2. Получаем токены (объекты с полями .text, .start, .stop)
                    tokens = list(get_words(raw_sentence))
                    
                    # Идем с конца строки к началу, чтобы замены не сбивали индексы
                    display_sentence = raw_sentence
                    for token in reversed(tokens):
                        # Проверяем, что это слово
                        if any(c.isalpha() for c in token.text):
                            p = morph.parse(token.text.lower())[0]
                            if p.normal_form == target_norm:
                                # Вырезаем старое слово и вставляем подсвеченное
                                start, end = token.start, token.stop
                                highlight = f"**{token.text.upper()}**"
                                display_sentence = display_sentence[:start] + highlight + display_sentence[end:]

                else:
                    display_sentence = '[Неизвестный контекст]'

                # Добавляем в таблицу контекстов (целое предложение)
                contexts.append({
                    "Контекст": display_sentence,
                    "Произведение": row['text_name'],
                    "Год": row['year_finished']
                })
        
        if text_occurrences:
            raw_data.append({'lemmas': sentences_lemmas, 'pos': text_occurrences})
            
    return total_occurrences, contexts, year_dist, raw_data

# 2. Функция классического окна (без индексов)
def get_window_neighbors(raw_data, target_norm, window_size, stopwords=None):
    """
    Считает соседей в жестком окне и их части речи.
    """
    neighbors = Counter()
    neighbor_pos = Counter()

    for text in raw_data:
        sentences = text['lemmas']
        for s_idx, t_indices in text['pos']:
            lemmas = sentences[s_idx]
            for t_idx in t_indices:
                start = max(0, t_idx - window_size)
                end = min(len(lemmas), t_idx + window_size + 1)
                
                for j in range(start, end):
                    if j == t_idx: continue
                    lemma = lemmas[j]
                    if lemma != '_BRK_' and lemma not in (stopwords or []) and lemma[0].isalpha():
                        neighbors[lemma] += 1
                        p = morph.parse(lemma)[0]
                        pos_name = POS_MAP.get(p.tag.POS, 'Другое')
                        neighbor_pos[pos_name] += 1
    return neighbors, neighbor_pos

# 3. Функция "Индекса Маяка" (всеобъемлющая связь)
def get_proximity_index(text, target_norm, decay_distance, decay_brks, decay_sents, stopwords=None):
    """
    Считает веса для ВСЕХ слов в тексте на основе дистанции и _BRK_.
    """
    weights = Counter()
    
    for item in text:
        # 1. Формируем "плоское" представление всего произведения
        flat_lemmas = []
        sent_boundaries = []  # Индексы, где заканчиваются предложения
        
        curr_idx = 0
        for sent in item['lemmas']:
            flat_lemmas.extend(sent)
            curr_idx += len(sent)
            sent_boundaries.append(curr_idx)
            
        # 2. Находим все позиции целевого слова (индексы в плоском списке)
        target_positions = [idx for idx, lemma in enumerate(flat_lemmas) if lemma == target_norm]
        
        # 3. Для каждого вхождения таргета сканируем весь текст
        for t_idx in target_positions:
            for s_idx, lemma in enumerate(flat_lemmas):
                
                # Пропускаем только строго ту же самую позицию (само себя)
                # и не-буквенные токены (типа _BRK_, которые мы обработаем отдельно)
                if s_idx == t_idx or not lemma.isalpha():
                    continue
                
                # Проверка на стоп-слова
                if stopwords and lemma in stopwords:
                    continue
                
                # --- МАТЕМАТИКА ДИСТАНЦИИ ---
                # Физическое расстояние в словах
                d = abs(s_idx - t_idx)
                
                # Определяем границы фрагмента между словами для поиска преград
                start, end = min(t_idx, s_idx), max(t_idx, s_idx)
                fragment = flat_lemmas[start:end]
                
                # --- СЧЕТЧИК ПРЕГРАД ---
                # 1. Сколько разрывов строк (_BRK_) встретилось на пути
                n_brks = fragment.count('_BRK_')
                
                # 2. Сколько границ предложений (точек) мы пересекли
                n_sents = len([b for b in sent_boundaries if start < b <= end])
                
                # --- ИТОГОВЫЙ ВЕС СВЯЗИ ---
                # Перемножаем затухания: Дистанция * Разрывы * Предложения
                weight = (decay_distance ** d) * (decay_brks ** n_brks) * (decay_sents ** n_sents)
                
                # Накапливаем вес для этой леммы
                weights[lemma] += weight
                
    return weights

# 4. Главная координирующая функция
def full_word_analysis(df, target_word, window_size=5, decay_distance=0.95, decay_brks=0.9, decay_sents=0.8, stopwords=None):
    target_norm = target_word.lower()
    
    # Шаг 1: База (контексты и вхождения)
    total_occurrences, contexts, year_dist, raw_data = get_occurrence_data(df, target_norm)
    
    if not raw_data:
        return None
    
    # Шаг 2: Классика (окно и POS)
    window_neighbors, pos_dist = get_window_neighbors(raw_data, target_norm, window_size, stopwords)
    
    # Шаг 3: Индекс (связи)
    proximity_weights = get_proximity_index(raw_data, target_norm, decay_distance, decay_brks, decay_sents, stopwords)
    
    return {
        'total_occurrences': total_occurrences,
        'contexts': contexts,
        'year_dist': year_dist,
        'window_neighbors': window_neighbors,
        'pos_dist': pos_dist,
        'proximity_weights': proximity_weights
    }