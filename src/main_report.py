import os
from itertools import product
from src.file_utils import get_files_in_folder, read_text_file, write_csv_file, read_csv_file, write_text_file
from src.text_utils import count_words, count_unique_words, count_unique_lemmas, clean_punctuation, calculate_ttr_lemmatized, count_sentences, get_words, lemmatize, form_author_vocab, words_per_sent, avg_punct_rate, get_most_common_words, get_most_common_lemmas

def main_text_report():
    '''
    Главная функция программы для генерации развернутого текстового отчета по корпусу.
    '''
    print("=" * 60)
    print("📂 Начинаю анализ текстовых файлов в корпусе!")
    print("=" * 60)

    corpus_folder = 'corpus'

    print(f"\n🔍 Проверка жанровых подкорпусов в корпусе '{corpus_folder}'...")

    corpus_genres = os.listdir('corpus')

    if not corpus_genres:
        print("❌ Не было найдено ни одного жанрового подкорпуса! Попробуйте добавить новый.")
        return None
    else:
        print(f'Жанровые подкорпуса {corpus_genres} успешно найдены! Начинаю обработку...')
    
    results = analyze_corpus(corpus_folder)
    metadata = read_csv_file(os.path.join('data', 'metadata.csv'))

    intervals = [[1906, 1922], [1922, 1930], [1930, 1938]]
    periods_vocab = lexical_comparison(corpus_folder, intervals=intervals, step=None)

    comparison_res = cross_genre_comparison(periods_vocab)

    generate_report(results, metadata, comparison_res)

    print('Анализ всего корпуса завершен УСПЕШНО! Очет доступен в папке "results"')

def analyze_single_text(filepath, filename):
    '''
    Производит анализ текста из файла и возвращает статистику по ряду заданных параметров в виде словаря.
    
    :param filepath: Описание
    :param filename: Описание
    '''
    headers = ['filename', 'word_count', 'unique_words', 'unique_lemmas', 'ttr', 'sentences_count', 'words_per_sent', 'avg_punct_rate']

    text = read_text_file(filepath)
    clean_text = clean_punctuation(text)
    
    word_count = count_words(clean_text, remove_punct=False)
    unique_word_count = count_unique_words(clean_text, remove_punct=False)
    lemmas_count = count_unique_lemmas(clean_text, remove_punct=False)
    ttr_count = calculate_ttr_lemmatized(clean_text, remove_punct=False)
    sentences_count = count_sentences(text)
    words_per_sent_count = words_per_sent(text)
    avg_punct_rate_count = avg_punct_rate(text)

    row = {
        headers[0]: filename,
        headers[1]: word_count,
        headers[2]: unique_word_count,
        headers[3]: lemmas_count,
        headers[4]: ttr_count,
        headers[5]: sentences_count,
        headers[6]: words_per_sent_count,
        headers[7]: avg_punct_rate_count
            }
    
    return row

def analyze_corpus(corpus_folder):
    """
    Анализирует все тексты в папке, сохраняет результаты и выводит статистику.

    Args:
        corpus_folder (str): Путь к папке с текстами (например, 'corpus')
    """
    data = [] # собирает статистику для записи в CSV файл
    total_words = 0 # считает общее количество слов во всех текстах всего корпуса
    total_files_counter = 0
    
    statistics_filepath = os.path.join('results', 'statistics.csv')

    corpus_genres = os.listdir('corpus')
    # добавить защиту

    for genre in corpus_genres:

        print(f'Начинаю анализ жанрового подкорпуса {genre}...')

        total_corpora_words = 0 # считает общее количество слов во всех текстах подкорпуса
        files_list = get_files_in_folder(os.path.join(corpus_folder, genre), '.txt')

        if not files_list:
            print(f'В корпусе поджанра {genre} нет ни одного текста! Корпус пропущен.')
            continue

        total_files_counter += len(files_list)

        for filename in files_list:
            filepath = os.path.join(corpus_folder, genre, filename)
            
            row = analyze_single_text(filepath, filename)

            total_corpora_words += row['word_count']
            data.append(row)

        print("=" * 45)
        print(f"\n✓ Проанализировано файлов: {len(files_list)}")

        print(f"\n📊 Общая статистика по подкорпусу {genre}:")
        print(f"Всего слов в корпусе: {total_corpora_words}")
        print(f"Среднее количество слов на текст: {total_corpora_words // len(files_list)}")
    
    headers = list(data[0].keys())
    write_csv_file(statistics_filepath, data, headers, rewrite=True)

    print(f"✓ Анализ текстов во всём корпусе завершен! Результаты сохранены в {statistics_filepath}")
    print(f"Всего проанализировано файлов: {total_files_counter}")
    print("\n---")

    statistics_data = read_csv_file(statistics_filepath)

    return statistics_data

def generate_report(results, metadata, comparison_res):
    '''
    Docstring для generate_report
    
    :param results: Описание
    :param metadata: Описание
    '''
    read_metadata = read_csv_file(os.path.join('data', 'metadata.csv'))
    read_statistics = read_csv_file(os.path.join('results', 'statistics.csv'))

    report_filepath = os.path.join('results', 'report.txt')

    all_strings = []

    header = f'''{'='*60}
             ОБЩИЙ ОТЧЕТ ПО АНАЛИЗУ КОРПУСА
             {'='*60}
             Всего файлов в базе данных: {len(metadata)}
    '''

    all_strings.append(header)
    
    for file_dict in metadata:
        filename = file_dict['filename']
        title = file_dict['title']
        year_finished = file_dict['year_finished']

        for metadata_dict in results:
            metadata_filename = metadata_dict['filename']
            if metadata_filename == filename:
                words_in_text = metadata_dict['word_count']
                unique_words = metadata_dict['unique_words']
                unique_lemmas = metadata_dict['unique_lemmas']
                sentences_count = metadata_dict['sentences_count']
                words_per_sentence = metadata_dict['words_per_sent']
                punctuation_rate = metadata_dict['avg_punct_rate']
                ttr = metadata_dict['ttr']
        
        file_info = f'''
        {'='*60}
        Проанализирован файл: {filename}

        Заглавие текста: {title}
        Текст закончен в {year_finished} году

        Количество слов в тексте: {words_in_text}
        Количество уникальных словоформ в тексте: {unique_words}
        Количество уникальных лемм в тексте: {unique_lemmas}

        Количество предложений в тексте: {sentences_count}
        Среднее количество слов в предложении: {words_per_sentence}
        Среднее количество знаков пунктуации в предложениях: {punctuation_rate}
        ***
        Показатель TTR по тексту: {ttr}
        {'='*60}
        '''
        all_strings.append(file_info)
    
    all_strings.append(f"\n\n{'='*60}")
    all_strings.append("ИНТЕРЕСНЫЕ НАБЛЮДЕНИЕ ПО КОРПУСУ")
    all_strings.append(f"{'='*60}")

    sorted_by_ttr = sorted(read_statistics, key=lambda x: float(x['ttr']), reverse=True)
    sorted_by_wordsentences = sorted(read_statistics, key=lambda x: float(x['words_per_sent']), reverse=True)
    sorted_by_punctsentences = sorted(read_statistics, key=lambda x: float(x['avg_punct_rate']), reverse=True)

    max_ttr = sorted_by_ttr[0]
    min_ttr = sorted_by_ttr[-1]
    max_wordsent = sorted_by_wordsentences[0]
    min_wordsent = sorted_by_wordsentences[-1]
    max_punctsent = sorted_by_punctsentences[0]
    min_punctsent = sorted_by_punctsentences[-1]

    for file_dict in read_metadata:
        if max_ttr['filename'] == file_dict['filename']:
            text_max_ttr = file_dict['title']
        elif min_ttr['filename'] == file_dict['filename']:
            text_min_ttr = file_dict['title']
        elif max_wordsent['filename'] == file_dict['filename']:
            text_max_wordsent = file_dict['title']
        elif min_wordsent['filename'] == file_dict['filename']:
            text_min_wordsent = file_dict['title']
        elif max_punctsent['filename'] == file_dict['filename']:
            text_max_punctsent = file_dict['title']
        elif min_punctsent['filename'] == file_dict['filename']:
            text_min_punctsent = file_dict['title']

    funfacts_string = f'''
    Текст с самым высоким уровнем TTR: {text_max_ttr} - {max_ttr['ttr']}
    Текст с самым низким уровнем TTR: {text_min_ttr} - {min_ttr['ttr']}
    Текст с самими длинными предложениями: {text_max_wordsent} - {max_wordsent['words_per_sent']}
    Текст с самыми короткими предложениями: {text_min_wordsent} - {min_wordsent['words_per_sent']}
    Самый пунктуационно насыщенный текст: {text_max_punctsent} - {max_punctsent['avg_punct_rate']}
    Текст с минимальным количеством пунктуации в предложениях: {text_min_punctsent} - {min_punctsent['avg_punct_rate']}
    '''

    all_strings.append(funfacts_string)

    all_strings.append(f"\n\n{'='*60}")
    all_strings.append("📊 АНАЛИЗ ДИНАМИКИ СЛОВАРЯ (МЕЖДУ ПЕРИОДАМИ)")
    all_strings.append(f"{'='*60}")

    current_transition = ""
    for item in comparison_res:
        transition = f"{item['period_from']} ➜ {item['period_to']}"
        
        if transition != current_transition:
            all_strings.append(f"\n📅 Переход: {transition}")
            current_transition = transition

        line = (f"   ▫️ [{item['comparison_type']}] "
                f"Сходство (Jaccard): {item['jaccard_index']} | "
                f"Общих лемм: {item['shared_lemmas_count']}")
        all_strings.append(line)

    report_data = '\n'.join(all_strings)

    write_text_file(report_filepath, report_data)

def lexical_comparison(corpus_folder, step=5, intervals=None):
    '''
    Docstring for lexical_comparison
    
    :param corpus_folder: Description

    Args:
        corpus_folder (str): Папка с текстами корпуса. Ожидается, что тексты лежат в отдельных подпапках в соответствии с их жанровой принадлежностью.
        step (int): Интервал, в рамках которого будет происходить сравнение текстов. Например, при интервале 3 тексты сравниваются группами по три года, жанровое разделение присутствует.
        intervals (list): Можно передать установленные вручную интервалы для сравнения по годам в виде списка списков с записанными попарно годами.
    '''
    if step and intervals:
        print('Ошибка!!! Установить одновременно два типа интервалов невозможно, пожалуйста, выберите только один параметр')
        return None
    
    author_vocab = form_author_vocab(corpus_folder)

    if not author_vocab:
        return None
    
    all_years = sorted(author_vocab.keys())
    
    first_year = int(all_years[0])
    last_year = int(all_years[-1])

    periods = []
    if intervals:
        periods = intervals
    elif step:
        for current_year in range(first_year, last_year + 1, step):
            end_year = current_year + step
            periods.append((current_year, end_year))

    aggregated_vocab = {}

    for start, end in periods:
        period_key = f"{start}-{end-1}"
        
        aggregated_vocab[period_key] = {}
        
        for year in range(start, end):
            if year in author_vocab.keys():
                year_data = author_vocab[year]
                for genre, vocab_data in year_data.items():
                    
                    if genre not in aggregated_vocab[period_key].keys():
                        aggregated_vocab[period_key][genre] = {'words': list(), 'lemmas': set()}

                    aggregated_vocab[period_key][genre]['words'].extend(vocab_data['words'])
                    aggregated_vocab[period_key][genre]['lemmas'].update(vocab_data['lemmas'])
        
    return aggregated_vocab

def cross_genre_comparison(aggregated_data):
    """
    Сравнивает жанры между соседними периодами по принципу "каждый с каждым".
    aggregated_data: результат работы lexical_comparison_by_genre
    """
    stats = []
    
    sorted_periods = sorted(aggregated_data.keys())
    
    for i in range(len(sorted_periods) - 1):
        period_from = sorted_periods[i]
        period_to = sorted_periods[i+1]
        
        stats.extend(compare_two_periods(aggregated_data, period_from, period_to))

    first_period = sorted_periods[0]
    last_period = sorted_periods[-1]
    
    borderline_stats = compare_two_periods(aggregated_data, first_period, last_period)
    stats.extend(borderline_stats)

    return stats

def compare_two_periods(data, p_from, p_to):
    '''
    Вспомогательная функция для сравнения двух конкретных периодов по жанрам на основании уже сформированного словаря лемм и слов
    '''
    results = []
    data_A, data_B = data[p_from], data[p_to]
    
    for genre_a, genre_b in product(data_A.keys(), data_B.keys()):

        vocab_a = data_A[genre_a]['lemmas']
        vocab_b = data_B[genre_b]['lemmas']

        intersection = vocab_a.intersection(vocab_b)
        union = vocab_a.union(vocab_b)

        jaccard = len(intersection) / len(union) if union else 0
        
        results.append({
            'period_from': p_from,
            'period_to': p_to,
            'genre_from': genre_a,
            'genre_to': genre_b,
            'jaccard_index': round(jaccard, 3),
            'shared_lemmas_count': len(intersection),
            'comparison_type': f"{genre_a} -> {genre_b}"
        })
    
    return results

def get_all_words(author_vocab):
    '''
    Docstring для get_all_words
    
    :param author_vocab: Описание
    '''
    all_words_dict = {}

    for year in author_vocab:
        year_data = author_vocab[year]
        for genre, vocab_data in year_data.items():
            
            if genre not in all_words_dict.keys():
                all_words_dict[genre] = []

            all_words_dict[genre].extend(vocab_data['words'])

    return all_words_dict

if __name__ == "__main__":
    # main()
    all_words_list = get_all_words(form_author_vocab('corpus'))
    # get_most_common_words(' '.join(all_words_list['poetry']), n=25)
    get_most_common_lemmas(' '.join(all_words_list['poetry']), n=200)

