import os
import requests
import csv
import pandas as pd

def read_text_file(filepath, encoding='utf-8'):
    """
    Читает содержимое текстового файла.

    Args:
        filepath (str): Путь к файлу
        encoding (str): Тип кодировки. По умолчанию установлен utf-8.

    Returns:
        str: Содержимое файла или сообщение об ошибке
    """
    content = None
    filename = os.path.basename(filepath)

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
    except FileNotFoundError:
        print(f'Ошибка: Файл "{filename}" не найден')
    except UnicodeDecodeError:
        print(f'Ошибка: Неверная кодировка файла {filename}')
    return content

def read_csv_file(filepath):
    '''
    Считывает CSV-файл и преобразует его в список словарей.

    Args:
        filepath (str): Путь к CSV-файлу.

    Returns:
        list: Список словарей, где ключи — названия колонок. 
              Возвращает пустой список, если файл не найден.
    '''
    try:
        with open(filepath, 'r', encoding='utf-8') as csv_file:
            content = list(csv.DictReader(csv_file))
    except FileNotFoundError:
        print('ОШИБКА! Файл не найден.')
        content = []
    return content

def write_csv_file(filepath, data, headers, rewrite=True):
    '''
    Записывает данные в CSV файл.

    Args:
        filepath (str): Полный путь к файлу, включая папку и название файла
                       Например: 'results/statistics.csv'
        data (list): Список cловарей [{col1: val1}, {col2, value2}, ...]
        headers (list): Список заголовков ['col1', 'col2']
        rewrite (bool): Указывает, нужно ли перезаписывать файл, если он уже существует, или дополнить его

    Returns:
        bool: True если успешно
    '''
    folder = os.path.dirname(filepath)
    os.makedirs(folder, exist_ok=True) 
    
    try:
        with open(filepath, 'x', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        return True
    except FileExistsError:
        if rewrite:
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)
                return True
            except FileNotFoundError:
                return False
        else:
            try:
                with open(filepath, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writerows(data)
                return True
            except FileNotFoundError:
                return False
    except FileNotFoundError:
        return False

def write_text_file(filepath, content, rewrite=True):
    '''
    Записывает текст в файл, автоматически создавая необходимые директории.

    Args:
        filepath (str): Путь к файлу (например, 'results/output.txt').
        content (str): Текст для записи.
        rewrite (bool): Если True — перезаписывает файл, если False — добавляет текст в конец.

    Returns:
        bool: True, если запись прошла успешно, False — если возникла ошибка.
    '''
    folder = os.path.dirname(filepath)
    os.makedirs(folder, exist_ok=True)

    try:
        with open(filepath, 'x', newline='', encoding='utf-8') as f:
            f.write(content)
        return True
    except FileExistsError:
        if rewrite:
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    f.write(content)
                    return True
            except FileNotFoundError:
                return False
        else:
            try:
                with open(filepath, 'a', newline='', encoding='utf-8') as f:
                    f.write(content)
                return True
            except FileNotFoundError:
                return False
    except FileNotFoundError:
        return False

def get_files_in_folder(folder_path, extension='.txt'):
    """
    Получает список файлов в указанной папке с заданным расширением.

    Args:
        folder_path (str): Путь к папке
        extension (str): Расширение файлов (по умолчанию '.txt')

    Returns:
        list: Список имен файлов с указанным расширением
    """
    files_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            files_list.append(filename)

    return files_list

def get_text_metadata(filename, metadata_path):
    '''
    Docstring для get_text_metadata
    
    :param filename: Описание
    :param metadata_path: Описание
    '''

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
    else:
        print(f'ОШИБКА! Файл с исходными метаданными не найден')
        return None

    row = metadata_df[metadata_df['filename'] == filename]

    if not row.empty:
        return row.iloc[0].to_dict() # Возвращает словарь со всеми полями (title, year, genre)
    return {"title": "Unknown", "year": "Unknown", "genre": "Unknown"}

# if __name__ == "__main__":
    # import_poetree_corpora(18, "Маяковский", os.path.join('corpus', 'poetry'))
    # import_corpora_files('raw_entries', 'corpus')