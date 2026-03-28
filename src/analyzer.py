import os
import re
import ast
from collections import Counter
from navec import Navec
from src.file_utils import read_text_file
from src.text_utils import morph, get_sentences, tokenizator, count_words


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

def highlight_lemma_forms_in_text(raw_text, vocabulary_forms, case_insensitive=True):
    """
    Подсвечивает все встреченные словоформы целевой леммы в тексте.
    Использует специальный маркер вместо звёздочек для совместимости со Streamlit.

    Args:
        raw_text: Исходный текст
        vocabulary_forms: Список всех словоформ целевой леммы
        case_insensitive: Игнорировать ли регистр при поиске

    Returns:
        Текст с маркированными формами (для последующей обработки в UI)
    """
    if not vocabulary_forms:
        return raw_text

    # Сортируем по длине (длинные первыми) чтобы избежать partial matches
    sorted_forms = sorted(set(vocabulary_forms), key=len, reverse=True)
    result = raw_text

    for form in sorted_forms:
        if case_insensitive:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(form) + r'\b'
            # Используем специальный маркер для выделения
            result = re.sub(pattern, f'<<<{form}>>>', result, flags=re.IGNORECASE)
        else:
            pattern = r'\b' + re.escape(form) + r'\b'
            result = re.sub(pattern, f'<<<{form}>>>', result)

    return result


# 1. Функция контекста и общей статистики
def find_all_form_occurrences(text, forms):
    """
    Находит все вхождения словоформ в тексте (case-insensitive).
    Возвращает список кортежей (form, start_pos, end_pos, char_position)
    в порядке появления в тексте.
    """
    occurrences = []
    # Сортируем по длине (длинные первыми) чтобы избежать partial matches
    sorted_forms = sorted(set(forms), key=len, reverse=True)

    for form in sorted_forms:
        pattern = r'\b' + re.escape(form) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            occurrences.append((form, match.start(), match.end()))

    # Сортируем по позиции в тексте
    return sorted(occurrences, key=lambda x: x[1])


def get_occurrence_data(filtered_corpus, target_norm, lemma_forms):
    """
    Ищет все вхождения словоформ целевой леммы в formatted_sentences.
    Для каждого вхождения:
    - Выводит предложение целиком как контекст
    - Если < 3 слов, добавляет самое короткое соседнее предложение
    - Подсвечивает все найденные формы
    - Считает вхождения по годам

    Возвращает:
    - total_occurrences: total count
    - contexts: список словарей для таблицы
    - year_dist: Counter по годам
    """
    contexts = []
    year_dist = Counter()
    total_occurrences = 0

    target_forms = lemma_forms.get(target_norm, [])

    if not target_forms:
        print(f"⚠️  Словоформы для '{target_norm}' не найдены в словаре")
        return 0, [], year_dist

    for item in filtered_corpus:
        sentences = item['formatted_sentences']

        # Для каждого предложения ищем вхождения
        for sent_idx, sentence in enumerate(sentences):
            occurrences = find_all_form_occurrences(sentence, target_forms)

            if not occurrences:
                continue

            # Есть вхождения — добавляем контекст
            # Если предложение < 3 слов, пытаемся добавить самое короткое соседнее
            context_sentence = sentence.strip(' / —–-')

            if count_words(context_sentence, remove_punct=False) < 3:
                neighbor_sentences = []

                if sent_idx > 0:
                    prev_sentence = sentences[sent_idx - 1].strip().strip(' / —–-')
                    neighbor_sentences.append((prev_sentence, count_words(prev_sentence, remove_punct=False)))

                if sent_idx < len(sentences) - 1:
                    next_sentence = sentences[sent_idx + 1].strip().strip(' / —–-')
                    neighbor_sentences.append((next_sentence, count_words(next_sentence, remove_punct=False)))

                if neighbor_sentences:
                    # Берём самое короткое из соседних
                    shortest_neighbor = min(neighbor_sentences, key=lambda x: x[1])[0]
                    context_sentence = f"{context_sentence} {shortest_neighbor}".strip(' / —–-')

            # Подсвечиваем все словоформы в контексте
            display_context = highlight_lemma_forms_in_text(context_sentence, target_forms)

            # Для каждого вхождения в этом предложении добавляем отдельную строку
            for form, _, _ in occurrences:
                total_occurrences += 1
                year_dist[item['year_finished']] += 1

                contexts.append({
                    "Контекст": display_context,
                    "Произведение": item['title'],
                    "Год": item['year_finished']
                })

    return total_occurrences, contexts, year_dist

# 2. Функция классического окна (без индексов)
def get_window_neighbors(raw_data, target_norm, window_size, stopwords=None):
    """
    Считает соседей в жестком окне и их части речи, используя lemmas_cleaned.
    Находит позиции целевого слова в lemmas_cleaned.
    """
    neighbors = Counter()
    neighbor_pos = Counter()

    for text in raw_data:
        # Формируем плоское представление lemmas_cleaned
        flat_lemmas = []
        for sent_lemmas in text['lemmas_cleaned']:
            flat_lemmas.extend(sent_lemmas)

        # Находим все позиции целевого слова
        target_positions = [idx for idx, lemma in enumerate(flat_lemmas) if lemma == target_norm]

        # Для каждого вхождения берём соседей в окне
        for t_idx in target_positions:
            start = max(0, t_idx - window_size)
            end = min(len(flat_lemmas), t_idx + window_size + 1)

            for j in range(start, end):
                if j == t_idx:
                    continue
                lemma = flat_lemmas[j]
                if lemma and lemma[0].isalpha() and lemma not in (stopwords or []):
                    neighbors[lemma] += 1
                    p = morph.parse(lemma)[0]
                    pos_name = POS_MAP.get(p.tag.POS, 'Другое')
                    neighbor_pos[pos_name] += 1

    return neighbors, neighbor_pos

# 3. Функция "Индекса Маяка" (динамический индекс контекстуальной близости)
def get_proximity_index_neighbors(filtered_corpus, target_norm, decay_distance, decay_brks, decay_sents, stopwords=None):
    """
    Для каждого вхождения таргета сканируем весь текст и считаем вес связи с каждой леммой, учитывая:
- Дистанцию в словах (чем дальше, тем слабее связь)
- Количество разрывов строк (_BRK_) на пути (каждый разрыв ослабляет связь)
- Количество границ предложений (точек) на пути (каждая граница ослабляет связь)
- Фильтрация по стоп-словам и не-буквенным токенам (типа _BRK_, которые мы обрабатываем отдельно)  
    """
    weights = Counter()

    for item in filtered_corpus:
        # 1. Формируем "плоское" представление из lemmas_cleaned (без _BRK_)
        flat_lemmas_clean = []
        flat_lemmas_orig = []  # Для подсчета _BRK_
        sent_boundaries = []

        curr_idx = 0
        for sent_clean, sent_orig in zip(item['lemmas_cleaned'], item['lemmas']):
            flat_lemmas_clean.extend(sent_clean)
            flat_lemmas_orig.extend(sent_orig)
            curr_idx += len(sent_clean)
            sent_boundaries.append(curr_idx)

        # 2. Находим все позиции целевого слова в clean версии
        target_positions = [idx for idx, lemma in enumerate(flat_lemmas_clean) if lemma == target_norm]

        # 3. Для каждого вхождения таргета сканируем весь текст
        for t_idx in target_positions:
            for s_idx, lemma in enumerate(flat_lemmas_clean):

                if s_idx == t_idx or not lemma.isalpha():
                    continue

                if stopwords and lemma in stopwords:
                    continue

                # Расстояние в словах (в clean версии - без _BRK_)
                d = abs(s_idx - t_idx)

                start, end = min(t_idx, s_idx), max(t_idx, s_idx)

                # Сколько разрывов строк (_BRK_) встретилось в оригинальной версии
                fragment_orig = flat_lemmas_orig[start:end]
                n_brks = fragment_orig.count('_BRK_')

                # Сколько границ предложений пересекли
                n_sents = len([b for b in sent_boundaries if start < b <= end])
                
                # --- ИТОГОВЫЙ ВЕС СВЯЗИ ---
                # Перемножаем затухания: Дистанция * Разрывы * Предложения
                weight = (decay_distance ** d) * (decay_brks ** n_brks) * (decay_sents ** n_sents)
                weights[lemma] += weight

    return weights

# 4. Главная координирующая функция
def full_word_analysis(filtered_corpus, target_word, window_size=5, decay_distance=0.95, decay_brks=0.9, decay_sents=0.8, stopwords=None, lemma_forms=None):

    if lemma_forms is None:
        lemma_forms = {}

    # Шаг 1: Находим все вхождения слова в formatted_sentences и собираем контексты
    total_occurrences, contexts, year_dist = get_occurrence_data(filtered_corpus, target_word, lemma_forms)

    if not filtered_corpus or not contexts:
        return None  # Если в корпусе нет данных, возвращаем None

    # Шаг 2: Классическое окно контекстов
    window_neighbors, pos_dist = get_window_neighbors(filtered_corpus, target_word, window_size, stopwords)

    # Шаг 3: Динамический индекс для всех слов в тексте
    proximity_weights = get_proximity_index_neighbors(filtered_corpus, target_word, decay_distance, decay_brks, decay_sents, stopwords)

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