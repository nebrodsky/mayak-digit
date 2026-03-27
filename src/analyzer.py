import os
import re
import ast
from collections import Counter
from navec import Navec
from src.file_utils import read_text_file
from src.text_utils import morph, get_sentences, tokenizator


path = os.path.join('models', 'navec_hudlit_v1_12B_500K_300d_100q.tar')
navec = Navec.load(path)

POS_MAP = {
    'NOUN': 'Существительное', 'VERB': 'Глагол', 'ADJF': 'Прилагательное',
    'ADJS': 'Краткое прил.', 'COMP': 'Компаратив', 'INFN': 'Инфинитив',
    'PRTF': 'Причастие', 'PRTS': 'Краткое прич.', 'GRND': 'Деепричастие',
    'NUMR': 'Числительное', 'ADVB': 'Наречие', 'NPRO': 'Местоимение',
    'PRED': 'Предикатив', 'PREP': 'Предлог', 'CONJ': 'Союз', 'PRCL': 'Частица'
}

# --- ФУНКЦИИ ДЛЯ АНАЛИЗА СЛОВА ---

def find_word_in_corpus(corpus, target_norm):
    """
    Ищет вхождения любого слова в уже подготовленном корпусе.
    Возвращает структуру raw_data для Индекса Маяка.
    """
    corpus_with_target = []

    for item in corpus:
        text_positions = []
        for s_idx, sentence in enumerate(item['lemmas']):
            if target_norm in sentence:
                indices = [i for i, l in enumerate([l for l in sentence if l != '_BRK_']) if l == target_norm]
                text_positions.append((s_idx, indices))
        
        if text_positions:
            corpus_with_target.append({
                'title': item['title'],
                'year_finished': item['year_finished'],
                'raw_text': item['raw_text'],
                'formatted_sentences': item['formatted_sentences'],
                'lemmas': item['lemmas'], 
                'positions': text_positions
            })

    return corpus_with_target

# 1. Функция контекста и общей статистики
def get_occurrence_data(corpus_with_target, target_norm):
    """
    Возвращает: 
    - contexts: список словарей для нижней таблицы
    - year_dist: Counter для графика частотности
    """
    contexts = []
    year_dist = Counter()
    total_occurrences = 0

    for item in corpus_with_target:
        # Считаем вхождения для графика
        num_in_text = sum(len(indices) for s_idx, indices in item['positions'])
        total_occurrences += num_in_text
        year_dist[item['year_finished']] += num_in_text

        sentences_raw = get_sentences(str(item['raw_text']))

        for s_idx, indicies in item['positions']:
            if s_idx < len(sentences_raw):
                raw_sentence = sentences_raw[s_idx].replace('\n', ' / ').strip(' /–-—')
                
                # Проверка: содержит ли предложение целевое слово
                has_target = False

                tokens = list(tokenizator(raw_sentence))
                for token in tokens:
                    if any(c.isalpha() for c in token.text):
                        p = morph.parse(token.text.lower())[0]
                        if p.normal_form == target_norm:
                            has_target = True
                            break
                
                if not has_target:
                    continue  # Пропускаем предложение, если оно не содержит целевое слово
                
                # Подсветка через reversed tokens
                display_sentence = raw_sentence
                
                for token in reversed(tokens):
                    if any(c.isalpha() for c in token.text):
                        p = morph.parse(token.text.lower())[0]
                        if p.normal_form == target_norm:
                            start, end = token.start, token.stop
                            display_sentence = display_sentence[:start] + f"**{token.text.upper()}**" + display_sentence[end:]
                
                contexts.append({
                    "Контекст": display_sentence,
                    "Произведение": item['title'],
                    "Год": item['year_finished']
                })

    return total_occurrences, contexts, year_dist

# 2. Функция классического окна (без индексов)
def get_window_neighbors(raw_data, target_norm, window_size, stopwords=None):
    """
    Считает соседей в жестком окне и их части речи.
    """
    neighbors = Counter()
    neighbor_pos = Counter()

    for text in raw_data:
        sentences = text['lemmas']
        for s_idx, t_indices in text['positions']:
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

# 3. Функция "Индекса Маяка" (динамический индекс контекстуальной близости)
def get_proximity_index_neighbors(text, target_norm, decay_distance, decay_brks, decay_sents, stopwords=None):
    """
    Для каждого вхождения таргета сканируем весь текст и считаем вес связи с каждой леммой, учитывая:
- Дистанцию в словах (чем дальше, тем слабее связь)
- Количество разрывов строк (_BRK_) на пути (каждый разрыв ослабляет связь)
- Количество границ предложений (точек) на пути (каждая граница ослабляет связь)
- Фильтрация по стоп-словам и не-буквенным токенам (типа _BRK_, которые мы обрабатываем отдельно)  
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
def full_word_analysis(filtered_corpus, target_word, window_size=5, decay_distance=0.95, decay_brks=0.9, decay_sents=0.8, stopwords=None):
    
    # Шаг 1: Находим все вхождения слова и контексты
    corpus_with_target = find_word_in_corpus(filtered_corpus, target_word)
    total_occurrences, contexts, year_dist = get_occurrence_data(corpus_with_target, target_word)
    
    if not filtered_corpus:
        return None  # Если в корпусе нет данных, возвращаем None или пустой результат
    
    # Шаг 2: Классическое окно контекстов
    window_neighbors, pos_dist = get_window_neighbors(corpus_with_target, target_word, window_size, stopwords)
    
    # Шаг 3: Динамический индекс для всех слов в тексте
    proximity_weights = get_proximity_index_neighbors(corpus_with_target, target_word, decay_distance, decay_brks, decay_sents, stopwords)
    
    return {
        'total_occurrences': total_occurrences,
        'contexts': contexts,
        'year_dist': year_dist,
        'window_neighbors': window_neighbors,
        'pos_dist': pos_dist,
        'proximity_weights': proximity_weights
    }

# --- ФУНКЦИИ ДЛЯ СИНОНИМОВ И ФИЛЬТРАЦИИ ---

def get_unique_synonyms(target_word, top_n_to_return=20, search_depth=50):
    """
    1. Находит глубокий топ синонимов (50 шт).
    2. Приводит их к нормальной форме (лемме).
    3. Оставляет только уникальные леммы, отличные от самого target_word.
    4. Возвращает срез нужной длины.
    """
    if target_word not in navec:
        return []

    raw_sims = []
    for word in navec.vocab.words:
        if not word.isalpha():
            continue
        score = navec.sim(target_word, word)
        raw_sims.append((word, score))
    
    raw_sims.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Фильтрация через лемматизацию
    unique_lemmas = []
    seen_lemmas = {target_word.lower()} # Сразу игнорируем само искомое слово
    
    for word, score in raw_sims:
        # Лемматизируем кандидата

        lemma = morph.parse(word)[0].normal_form 
        
        if lemma not in seen_lemmas:
            unique_lemmas.append((lemma, score))
            seen_lemmas.add(lemma)
            
        # Как только набрали нужное количество уникальных понятий — выходим
        if len(unique_lemmas) >= search_depth:
            break

    # 3. Возвращаем срез (20 или сколько запрошено)
    final_cutoff = min(len(unique_lemmas), top_n_to_return)
    return unique_lemmas[:final_cutoff]

def filter_synonyms_by_corpus(synonyms, corpus_df):
    """
    Дополнительная фильтрация синонимов через корпус.
    Оставляет только те, которые реально встречаются в текстах.
    """

    author_vocab = read_text_file('data/author_vocabulary.txt').splitlines()
    filtered_synonyms = [syn for syn, score in synonyms if syn in author_vocab]
    
    return filtered_synonyms

# --- ФУНКЦИИ ДЛЯ ПОДГОТОВКИ ЛЛМ-ИНТЕРПРЕТАЦИИ ---

def synonyms_proximity_index(target_word, synonyms_filtered, proximity_weights):
    """
    Создает словарь для LLM, который показывает вес связи каждого синонима с таргетом по кастомному индексу.
    Это поможет модели понять, какие синонимы были более "близки" к таргету в корпусе.
    """
    syn_proximity = {}

    for syn in synonyms_filtered:
        syn_proximity[syn] = proximity_weights.get(syn, 0.0)
    
    syn_proximity_sorted = dict(sorted(syn_proximity.items(), key=lambda x: x[1], reverse=True))
    
    return syn_proximity_sorted

def proximity_neighbours_for_synonyms(synonyms_filtered, raw_data, decay_distance=0.95, decay_brks=0.9, decay_sents=0.8, stopwords=None):
    """
    Для каждого отфильтрованного синонима считаем его соседей по "Индексу Маяка".
    Позволяет LLM понять, какие слова были близки к каждому синониму, а не только к таргету.
    """
    neightbors_for_synonyms = {}

    for syn in synonyms_filtered:
        weights = get_proximity_index_neighbors(raw_data, syn, decay_distance=0.95, decay_brks=0.9, decay_sents=0.8, stopwords=stopwords)
        # Сортируем и берем топ-10 соседей для каждого синонима
        neightbors_for_synonyms[syn] = weights.most_common(10)

    return neightbors_for_synonyms

def prepare_llm_prompt(target_word, synonyms, synonyms_filtered, syn_proximity, neighbors_for_synonyms):
    """
    Формирует расширенный текстовый промпт для LLM.

    Включаем:
    - Список синонимов с их весами связи
    - Топ соседей для каждого синонима по "Индексу Маяка"
    Это позволит модели делать более обоснованные выводы о том, какие синонимы были действительно релевантными в корпусе.
    """
    syn_blocks = []

    syn_proximity = dict(sorted(syn_proximity.items(), key=lambda x: x[1], reverse=True))

    for syn, neighbors in neighbors_for_synonyms.items():
        neighbors_line = ", ".join([f"{n} ({w:.2f})" for n, w in neighbors])
        syn_blocks.append(f"  - '{syn}': {neighbors_line}")
    
    neighbors_for_synonyms_str = "\n".join(syn_blocks)
 
    for syn, prox in syn_proximity.items():
        synonyms_str = ", ".join([f"{syn} ({prox:.4f})" for syn, prox in syn_proximity.items()])

    # Сборка финального текста
    prompt = f"""
    Ты — ведущий эксперт по цифровой филологии и творчеству Владимира Маяковского.
    Тебе нужно провести глубокий сравнительный анализ слова "{target_word.upper()}" на основе предоставленных данных для составления словарной статьи.

    Тебе предоставлены следующие данные:

    1. Основное слово "{target_word}".

    2. Синонимы этого слова из общего корпуса художественной русской литературы (расположены по убыванию близости): "{', '.join([syn for syn, score in synonyms])}".

    3. Семантические лакуны (синонимы из общего языка, которые поэт ПОЛНОСТЬЮ игнорирует): "{', '.join([syn for syn, score in synonyms if syn not in synonyms_filtered])}".

    4. Синонимы этого слова, которые реально встречаются в текстах Маяковского вместе с индексом их контекстуальной близости к основному слову: "{synonyms_str}".

    5. Для каждого синонима из пункта 4 — топ-10 слов, которые были наиболее тесно связаны с этим синонимом в текстах (по динамическому индексу контекстуальной близости), вместе с их весами связи:
    {neighbors_for_synonyms_str}.

    Перед тобой поставлены следующие вопросы для анализа:
    1. Сравни контекстуальное окружение таргета и его используемых синонимов. В чем функциональное различие таргета и синонимов в текстах?
    2. Проанализируй список использованных синонимов и сравни, насколько они близки к таргету по дданным общего корпуса и насколько по данным корпуса Маяковского.
    3. Поясни, в каких случаях поэт выбирает менее "близкие" синонимы или наоборот избегает "популярных"?
    4. Проанализируй список проигнорированных синонимов. Почему, на твой взгляд, Маяковский полностью избегает этих слов?

    Напиши аналитическое заключение. Будь конкретен, опирайся на предоставленные веса и леммы. Язык должен быть понятен как специалисту, так и заинтересованным читателям, не погруженным в контекст.
    """
    return prompt