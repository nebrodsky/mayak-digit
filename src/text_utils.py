import os
import re
import pymorphy3
from razdel import sentenize, tokenize #для членения текста используем модуль razdel
from src.file_utils import read_csv_file, read_text_file
from pymystem3 import Mystem

morph = pymorphy3.MorphAnalyzer()
ms = Mystem(disambiguation=True, grammar_info=True, entire_input=False)
ms.start()

# --- СПИСОК СТОП-СЛОВ ДЛЯ РУССКОГО ЯЗЫКА (можно расширить) ---
russian_stopwords = set("""
    и в во не что он на я с со как а то все она так его но да ты к у же вы за б бы по 
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли 
    если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя 
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без 
    будто чего раз тоже себе под будет ж тогда кто это этот того потому этого какой 
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех 
    никогда можно при наконец два об другой хоть после над больше тот через эти 
    нас про всего них какая много разве три эту моя впрочем хорошо свою этой 
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
""".split())

# --- ФУНКЦИИ ОЧИСТКИ ТЕКСТА ---

def clean_punctuation(text):
    '''
    Удаляет знаки пунктуации из текста, заменяя их на пробелы.
    
    Args:
        text (str): Исходный текст.
    Returns:
        str: Текст в нижнем регистре без знаков препинания.
    '''
    punct = '\'\",.!?;:—-“”«»()[]/'
    clean_text = text.lower()
    for char in punct:
        clean_text = clean_text.replace(char, ' ')
    return clean_text

def clean_mayakovsky_fragmentation(text):
    # 1. Склеиваем переносы строк внутри слова: 'сло- / во' -> 'слово /'
    # Ищем: буква + дефис + пробелы? + слэш + пробелы? + буква
    # Меняем на: первая буква + вторая буква + пробел + слэш
    text = re.sub(r'(\w+)-\s*/\s*(\w+)', r'\1\2 /', text)

    # 2. Склеиваем акцентные дефисы: 'н-а-ш-е' -> 'наше'
    # Ищем цепочки одиночных букв через дефис: н-а-ш-е
    # Это регулярка находит последовательности типа А-Б-В... и убирает дефисы
    text = re.sub(r'(\b\w)(-\w)+\b', lambda m: m.group(0).replace('-', ''), text)
    
    return text

# --- ФУНКЦИИ ДЛЯ ПРОСТЫХ СТАТИСТИЧЕСКИХ ПОДСЧЕТОВ ---

def count_words(text, remove_punct=True):
    """
    Подсчитывает количество слов в тексте.

    Args:
        text (str): Текст для анализа
        remove_punct (bool): Нужно ли очищать текст от знаков пунктуации.
    Returns:
        int: Количество слов
    """
    clean_text = clean_punctuation(text) if remove_punct else text

    words = clean_text.split()
    return len(words)

def count_unique_words(text, remove_punct=True):
    """
    Подсчитывает количество уникальных слов в тексте.

    Args:
        text (str): Текст для анализа
        remove_punct (bool): Нужно ли очищать текст от знаков пунктуации.

    Returns:
        int: Количество уникальных слов
    """
    clean_text = clean_punctuation(text) if remove_punct else text

    unique_words = set(clean_text.split())
    return len(unique_words)

def count_unique_lemmas(text, remove_punct=True):
    """
    Подсчитывает количество уникальных лемм (начальных форм слов).

    Более точная оценка словарного запаса, чем просто уникальные слова.

    Args:
        text (str): Текст для анализа
        remove_punct (bool): Нужно ли очищать текст от знаков пунктуации.

    Returns:
        int: Количество уникальных лемм
    """
    clean_text = clean_punctuation(text) if remove_punct else text

    words = clean_text.split()
    lemmas = set()

    for word in words:
        if word:
            parsed = morph.parse(word)[0]
            lemmas.add(parsed.normal_form)
    
    return len(lemmas)

def count_lines(text):
    '''
    Подсчитывает количество непустых строк в тексте.
    '''
    lines = text.splitlines()
    lines = [line for line in lines if line.strip()]

    return len(lines)

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ТОКЕНАМИ И ЛЕММАМИ ---

def tokenizator(text, remove_punct=False):
    '''
    Разбивает текст на токены (слова) и удаляет пунктуацию при необходимости.
    '''

    tokens = list(tokenize(text))

    if remove_punct:
        for token in tokens:
            token.text = clean_punctuation(token.text)

    return tokens

def lemmatize(text, remove_punct=True):
    '''
    Производит лемматизацию текста
    '''
    clean_text = clean_punctuation(text) if remove_punct else text

    words = clean_text.split()
    lemmas = []

    for word in words:
        if word:
            parsed = morph.parse(word)[0]
            lemmas.append(parsed.normal_form)
    
    return list(lemmas)

def lemmatize_with_mystem(sentence):
    """
    Лемматизирует предложение используя razdel для токенизации и MyStem для лемматизации.

    """

    sentence_cleaned = clean_mayakovsky_fragmentation(sentence)

    tokens = list(tokenize(sentence_cleaned))
    tokens = [token.text for token in tokens]

    sentence_mystem = sentence_cleaned.replace('/', ' ')

    ms_analysis = ms.analyze(sentence_mystem)

    ms_words = [word for word in ms_analysis if word['text'].strip() and any(c.isalpha() for c in word['text'])]

    lemmas = []
    lemmas_clean = []
    pos_tags = []
    lemmas_pos = []

    last_mystm_pos = 0

    for tk in tokens:
        
        token_text = tk.lower()

        if token_text == '/':
            lemmas.append('_BRK_')
            continue

        if not any(c.isalnum() for c in token_text):
            continue

        # Если в токене есть цифры, мы не будем пытаться его лемматизировать, а просто добавим как есть
        if any(c.isdigit() for c in token_text):
            lemmas.append(token_text)
            lemmas_clean.append(token_text)
            pos_tags.append('NUM')
            lemmas_pos.append(f"{token_text}/NUM")
            continue

        # Если в токене есть дефис, мы тоже не будем пытаться его лемматизировать, а просто добавим как есть
        if '-' in token_text:
            ms_result = ms.analyze(token_text)
            if len(ms_result) == 1 and ms_result[0]['analysis']:
                info = ms_result[0]['analysis'][0]
                lemma = info['lex']
                pos = info.get('gr', '').split(',')[0].split('=')[0]
            elif len(ms_result) == 2 and any(info['analysis'] for info in ms_result):
                token_text_parts = token_text.split('-')
                lemma_parts = []
                for indx, res in enumerate(ms_result):
                    part_result = ms_result[indx]['analysis']
                    if not part_result:
                        lemma_parts.append(token_text_parts[indx])
                        print(f"⚠️ Внимание! Не удалось лемматизировать часть дефисного слова '{token_text_parts[indx]}' в токене '{token_text}'. Добавляем как есть.")
                        continue
                    part_info = part_result[0]
                    lemma_parts.append(part_info['lex'])
                lemma = '-'.join(lemma_parts)
                pos = 'HYPHCOMP'
            else:
                lemma = token_text
                pos = 'UNK'
            lemmas.append(lemma)
            lemmas_clean.append(lemma)
            pos_tags.append(pos)
            lemmas_pos.append(f"{lemma}/{pos}")
            continue

        while last_mystm_pos < len(ms_words):

            ms_word = ms_words[last_mystm_pos]
            ms_initial = ms_word['text']

            if ms_initial == tk:
                
                if ms_word['analysis']:
                    info = ms_word['analysis'][0]
                    lemma = info['lex']
                    pos = info.get('gr', '').split(',')[0].split('=')[0]
                else:
                    lemma = ms_initial
                    pos = 'UNK'

                lemmas.append(lemma)
                lemmas_clean.append(lemma)
                pos_tags.append(pos)
                lemmas_pos.append(f"{lemma}/{pos}")

                last_mystm_pos += 1
                break

            else:
                last_mystm_pos += 1
                continue
    
    if last_mystm_pos != len(ms_words):
        print(f"⚠️ Внимание! Не все слова были обработаны MyStem! Обработано {last_mystm_pos} из {len(ms_words)} слов.")
        print(f"Необработанные слова MyStem: {[w['text'] for w in ms_words[last_mystm_pos:]]}")

    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'lemmas_clean': lemmas_clean,
        'pos_tags': pos_tags,
        'lemmas_pos_tagged': lemmas_pos
    }

def calculate_ttr_lemmatized(text, remove_punct=True):
    """
    Вычисляет Type-Token Ratio на основе лемм.

    TTR = уникальные леммы / все слова

    Более точная оценка лексического разнообразия для
    флективных языков (русский, итальянский, немецкий).

    Args:
        text (str): Текст для анализа

    Returns:
        float: TTR от 0 до 1
    """
    clean_text = clean_punctuation(text) if remove_punct else text

    words = [w for w in clean_text.split() if w]

    if not words:
        return 0.0

    lemmas = set()
    for word in words:
        parsed = morph.parse(word)[0]
        lemmas.add(parsed.normal_form)

    return round(len(lemmas) / len(words), 4)

def get_pos_statistics(text, remove_punct):
    '''
    Подсчитывает статистику по частям речи.
    '''
    clean_text = clean_punctuation(text) if remove_punct else text
    
    pos_counts = {} 
    words = clean_text.split()

    for word in words:
        if word:
            parsed = morph.parse(word)[0]
            pos = parsed.tag.POS
            if pos:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

    return pos_counts

def get_lines(text, without_blanks=True):
    '''
    Возвращает список строк в тексте в виде списка. При необходимости также сохраняет пустые строки.
    '''
    lines = text.splitlines()

    if without_blanks:
        lines = [line for line in lines if line.strip()]

    return lines

def get_sentences(text):
    '''
    Разбивает текст на предложения
    '''
    sentences = [sentence.text.replace('\n', ' ') for sentence in list(sentenize(text))]

    return sentences

def count_sentences(text):
    '''
    Считает количество предложений в тексте
    '''
    sentences = [sentence.text.replace('\n', ' ').capitalize() for sentence in list(sentenize(text))]

    return len(sentences)

def words_per_sent(text):
    '''
    Считает среднее количество слов в предложениях в тексте.

    Args:
        text (str): Исходный необработанный текст в виде строки.

    Returns:
        avg_sent_len (float): Среднее количество слов в предложениях в тексте с округлением до двух знаков после запятой.
    '''
    total_words = count_words(text)
    total_sent = count_sentences(text)

    if not total_sent:
        return 0.0

    avg_sent_len = round(total_words / total_sent, 2)

    return avg_sent_len



def avg_punct_rate(text):
    '''
    Считает среднее количество пунктуационных знаков в предложениях текста.
    
    Args:
        text (str): Исходный необработанный текст в виде строки.

    Returns:
        avg_punct_rate (float): Среднее количество пунктуационных знаков с округлением до двух знаков после запятой.
    '''
    total_punct = 0
    
    sentences = get_sentences(text)
    total_sent = len(sentences)

    for sentence in sentences:
        for ch in sentence:
            if ch in '.,!?;:—–-"«»()':
                total_punct += 1

    if not total_sent:
        return 0.0
    
    return round(total_punct / total_sent, 2)

def get_most_common_words(text, n=0):
    '''
    Выводит n самых частотных слов в тексте.
    
    Args:
        text (str): Текст для анализа.
        n (int): Количество выводимых слов (если 0 — вывода не будет)
    '''
    # russian_stopwords = set(stopwords.words('russian'))

    frequencies = {}
    punct = '\'\",.!?;:—-“”«»'
    clean_text = text.lower()
    for ch in punct:
        clean_text = clean_text.replace(ch, '')
    words = clean_text.split()

    for w in words:
        if w not in russian_stopwords and w.isalpha():
            frequencies[w] = frequencies.get(w, 0) + 1

    sorted_dict = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))

    if n != 0:
        print(f'Самые частые слова в тексте ({n} первых позиций):')
        for i, pair in enumerate(list(sorted_dict)[:n], 1):
                print(f'{i} - {pair[0]}: {pair[1]}')

def get_most_common_lemmas(clean_text, n=0):
    '''
    Выводит n самых частотных лемм в тексте.
    
    Args:
        text (str): Текст для анализа.
        n (int): Количество выводимых лемм (если 0 — вывода не будет)
    '''
    # russian_stopwords = set(stopwords.words('russian'))

    frequencies = {}
    lemmas = list()

    words = clean_text.split()

    for word in words:
        if word.isalpha() and word not in russian_stopwords:
            parsed = morph.parse(word)[0]
            lemmas.append(parsed.normal_form)

    for lemma in lemmas:
        frequencies[lemma] = frequencies.get(lemma, 0) + 1

    sorted_dict = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))

    if n != 0:
        print(f'Самые частые слова в тексте ({n} первых позиций):')
        for i, pair in enumerate(list(sorted_dict)[:n], 1):
                print(f'{i} - {pair[0]}: {pair[1]}')

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С КОРПУСОМ И СТАТИСТИКОЙ ПО АВТОРУ ---

def list_by_years(corpus_folder):
    '''
    Группирует файлы корпуса по годам и жанрам на основе метаданных.
    
    Args:
        corpus_folder (str): Путь к папке с корпусом.
    Returns:
        list: Отсортированный список кортежей (год, {жанр: [файлы]})
    '''
    metadata_filepath = os.path.join('data', 'metadata.csv')
    statistics_filepath = os.path.join('results', 'statistics.csv')
    metadata = read_csv_file(metadata_filepath)

    list_by_years = {}

    for text_data in metadata:
        year = text_data['year_finished']
        filename = text_data['filename']
        genre = text_data['genre']
        if list_by_years.get(year) == None:
            list_by_years[year] = {genre: [filename]}
        else:
            if list_by_years[year].get(genre) == None:
                list_by_years[year][genre] = [filename]
            else:
                list_by_years[year][genre].append(filename)

    list_by_years = sorted(list_by_years.items())

    return list_by_years

def form_author_vocab(corpus_folder):
    '''
    Собирает все слова и леммы автора, распределенные по годам и жанрам.
    
    Args:
        corpus_folder (str): Путь к папке с текстами.
    Returns:
        dict: Вложенный словарь со статистикой лексики по годам.
    '''
    metadata_filepath = os.path.join('data', 'metadata.csv')
    metadata = read_csv_file(metadata_filepath)

    author_vocab = {}

    for text_data in metadata:

        year = int(text_data['year_finished'])
        filename = text_data['filename']
        genre = text_data['genre']

        if not year:
            continue

        filefolder = os.path.join(corpus_folder, genre)
        filepath = os.path.join(filefolder, filename)

        text = read_text_file(filepath)
        new_words = list(tokenizator(text))
        new_lemmas = set(lemmatize(text))

        if year not in author_vocab:
            author_vocab[year] = {}
        
        if genre not in author_vocab[year]:
            author_vocab[year][genre] = {'words': [], 'lemmas': set()}
        
        author_vocab[year][genre]['words'].extend(new_words)
        author_vocab[year][genre]['lemmas'] = author_vocab[year][genre]['lemmas'] | new_lemmas
    
    sorted_by_years = sorted(author_vocab.items())

    return dict(sorted_by_years)

def format_separate_poem(text):
    '''
    Разделяет стихотворение на строки и строфы, используя символы переноса строки.
    '''

    formatted_text = text.strip()
    formatted_text = formatted_text.replace('\n\n', ' // ').replace('\n', ' / ')

    return formatted_text

# if __name__ == "__main__":


    