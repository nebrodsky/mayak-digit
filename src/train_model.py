import pandas as pd
import ast
from gensim.models import Word2Vec

# --- НА ДАННОМ ЭТАПЕ ИСПОЛЬЗОВАНИЕ МОДЕЛИ НЕ ПЛАНИРУЕТСЯ, ЭТО ПРОСТО СКРИПТ ДЛЯ ОБУЧЕНИЯ НОВОЙ МОДЕЛИ НА НАШЕМ КОРПУСЕ ---

# Подготовка лемм для обучения Word2Vec
def get_clean_sentences(df):
    sentences = []
    for l_str in df['lemmas']:
        try:
            nested = ast.literal_eval(l_str)
            # Убираем _BRK_ и пустые токены
            clean = [word for sublist in nested for word in sublist if not word.startswith('_')]
            if clean:
                sentences.append(clean)
        except:
            continue
    return sentences

def model_trainer():
    '''
    Docstring для model_trainer
    '''
    print("Загрузка базы...")
    df = pd.read_csv('data/database.csv')

    print("Подготовка текстов...")
    sentences = get_clean_sentences(df)

    print("Обучение Word2Vec (это может занять минуту)...")
    model = Word2Vec(
        sentences, 
        vector_size=100, 
        window=5, 
        min_count=4, # игнорируем совсем редкие опечатки
        sg=1,        # Skip-gram лучше для малых данных
        epochs=150   # Те самые 100 эпох для глубины
    )

    model.save("models/mayakovsky_v3.model")
    print("Модель успешно сохранена в файл mayakovsky_v3.model!")

if __name__ == "__main__":
    model_trainer()