import pandas as pd
import dask.dataframe as dd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import streamlit as st
import io
import torch
import os

# Загрузка предобученной модели Sentence-BERT
model_path = 'finetuned_model.pth'

# Функция для загрузки модели
def load_model(path):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    try:
        model.load_state_dict(torch.load(path))
        st.write("Загружена дообученная модель.")
    except FileNotFoundError:
        st.write("Загружена предобученная модель.")
    return model

# Функция для очистки кэша
def clear_cache():
    try:
        os.remove(model_path)
        st.write("Кэш очищен.")
    except FileNotFoundError:
        st.write("Кэш уже очищен.")

# Функция для сохранения модели
def save_model(model, path):
    torch.save(model.state_dict(), path)
    st.write("Модель сохранена.")

# Загрузка моделей
main_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
fine_tuned_model = load_model(model_path)

# Функция для загрузки данных
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return dd.from_pandas(df, npartitions=2)

# Функция для вычисления эмбеддингов
@st.cache_data
def compute_embeddings(names, _model):
    # Убираем вызов .compute() так как он не применим к PyTorch тензору
    return _model.encode(names, convert_to_tensor=True)

# Создание интерфейса
st.title("Неро")
st.write("Система для поиска совпадений между названиями товаров")

# Кнопка для очистки кэша
if st.button("Очистить кэш"):
    clear_cache()

# Выбор, использовать ли дообученную модель
use_finetuned_model = st.checkbox("Использовать дообученную модель")

# Загрузка модели
if use_finetuned_model:
    model = load_model(model_path)
else:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    st.write("Загружена предобученная модель.")

# Функция для дообучения модели
def train_model(model, examples, new_data=None):
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    if new_data is not None:
        new_examples = [InputExample(texts=[new_data[i]], label=1) for i in range(len(new_data))]
        new_dataloader = DataLoader(new_examples, shuffle=True, batch_size=16)
        model.fit(train_objectives=[(train_dataloader, train_loss), (new_dataloader, train_loss)], epochs=1)
    else:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
    save_model(model, model_path)

# Функция для нахождения совпадающих названий
def find_matching_names(our_product_names, competitor_product_names, threshold=0.8, model=None):
    if model is None:
        raise ValueError("Модель не передана")

    main_embeddings_our = compute_embeddings(our_product_names, model)
    main_embeddings_competitor = compute_embeddings(competitor_product_names, model)

    fine_tuned_embeddings_our = compute_embeddings(our_product_names, fine_tuned_model)
    fine_tuned_embeddings_competitor = compute_embeddings(competitor_product_names, fine_tuned_model)

    # Вычисление средней схожести между предобученной и дообученной моделью
    similarities = (util.pytorch_cos_sim(main_embeddings_our, main_embeddings_competitor) +
                    util.pytorch_cos_sim(fine_tuned_embeddings_our, fine_tuned_embeddings_competitor)) / 2

    matching_names = []
    for i, our_name in enumerate(our_product_names):
        for j, competitor_name in enumerate(competitor_product_names):
            if similarities[i][j] >= threshold:
                matching_names.append((our_name, competitor_name, similarities[i][j].item()))

    # Сортируем по схожести (по убыванию), а затем по названию товара (по возрастанию)
    matching_names = sorted(matching_names, key=lambda x: (-x[2], x[0]))

    return matching_names

# Создание визуального интерфейса с использованием Streamlit
st.title('Сравнение названий товаров')
st.write('Загрузите файлы с названиями ваших товаров и товаров конкурента для поиска совпадений.')

our_file = st.file_uploader('Загрузите файл с вашими товарами', type=['xlsx'])
competitor_file = st.file_uploader('Загрузите файл с товарами конкурента', type=['xlsx'])
examples_file = st.file_uploader('Загрузите файл с примерами соответствий (необязательно)', type=['xlsx'])

if our_file and competitor_file:
    our_data = load_data(our_file)
    competitor_data = load_data(competitor_file)

    our_columns = our_data.columns.tolist()
    competitor_columns = competitor_data.columns.tolist()

    our_column = st.selectbox('Выберите колонку с названиями ваших товаров', our_columns)
    competitor_column = st.selectbox('Выберите колонку с названиями товаров конкурента', competitor_columns)

    if examples_file:
        examples_data = load_data(examples_file)
        examples_column_our = st.selectbox('Выберите колонку с вашими названиями из примеров',
                                           examples_data.columns.tolist())
        examples_column_competitor = st.selectbox('Выберите колонку с названиями конкурента из примеров',
                                                  examples_data.columns.tolist())

        examples = []
        for index, row in examples_data.iterrows():
            example = InputExample(texts=[row[examples_column_our], row[examples_column_competitor]])
            examples.append(example)

        st.write('Дообучение модели на примерах...')
        train_model(fine_tuned_model, examples)

    st.write('Обработка данных...')

    # Извлекаем данные из Dask DataFrame и приводим их к Pandas DataFrame
    our_product_names = our_data[our_column].compute().tolist()
    competitor_product_names = competitor_data[competitor_column].compute().tolist()

    st.write('Вычисление эмбеддингов и поиск совпадений...')

    matching_names = find_matching_names(our_product_names, competitor_product_names, threshold=0.8, model=main_model)

    if matching_names:
        # Создаем DataFrame со всеми найденными совпадениями
        results_df = pd.DataFrame(matching_names, columns=['Наше название', 'Название конкурента', 'Схожесть'])

        # Отображаем все совпадения
        st.write('Все совпадения:')
        st.dataframe(results_df)

        # Создание файла Excel для скачивания всех совпадений
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Записываем все совпадения на лист "Results"
            results_df.to_excel(writer, index=False, sheet_name='Results')

            # Получаем доступ к книге и проверяем листы
            workbook = writer.book
            # Проверяем, есть ли листы, и делаем первый лист видимым и активным
            if workbook.worksheets:
                first_sheet = workbook.worksheets[0]
                first_sheet.sheet_state = 'visible'  # Убедимся, что лист видимый
                workbook.active = 0  # Устанавливаем первый лист как активный

        processed_data = output.getvalue()

        st.download_button(
            label="Скачать результаты",
            data=processed_data,
            file_name='matching_products.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    else:
        st.write('Совпадений не найдено.')
