from gensim.models import Word2Vec

# Загрузка модели
model = Word2Vec.load("models/mayak_w2v_skipgram.model")

word = "коняга"
if word in model.wv:
    print(f"--- Соседи слова '{word}' ---")
    similar = model.wv.most_similar(word, topn=20)
    for i, (w, score) in enumerate(similar, 1):
        print(f"{i}. {w:15} | близость: {score:.4f}")
else:
    print(f"Слово '{word}' не найдено в словаре.")