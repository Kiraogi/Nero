import pandas as pd
import dask.dataframe as dd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import streamlit as st
import io
import torch
import os
import re
from datetime import datetime

"""
Задачи на ближайшие время:
1) Доработать модель: 
    ∟Добавить проверку на Англ языке 
    ∟Дообучить
2) Добавить кнопки для выбора модели, сохранения ее
3) 
"""

model_directory = './models/'  # Путь к директории для хранения моделей

# Функция для получения следующего имени модели с увеличением версии
def get_next_model_name():
    current_date = datetime.now().strftime("%d.%m.%Y")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_files = [f for f in os.listdir(model_directory) if re.match(r'^\d{2}\.\d{2}\.\d{4} Nero V\d+\.\d+$', f)]
    if not model_files:
        return f"{current_date} Nero V1.0"
    latest_model = sorted(model_files)[-1]
    version_match = re.search(r'V(\d+)\.(\d+)$', latest_model)
    if version_match:
        major, minor = map(int, version_match.groups())
        new_version = f"V{major}.{minor + 1}"
        return f"{current_date} Nero {new_version}"
    return f"{current_date} Nero V1.0"

# Функция для загрузки модели
def load_model(path):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    try:
        model.load_state_dict(torch.load(path))
        st.write("Загружена дообученная модель.")
    except FileNotFoundError:
        st.write("Файл модели не найден. Загружена предобученная модель.")
    except Exception as e:
        st.write(f"Произошла ошибка при загрузке модели: {e}")
    return model

# Функция для сохранения модели
def save_model(model, path):
    torch.save(model.state_dict(), path)
    st.write(f"Модель сохранена как: {path}")

# Загрузка данных из файла Excel с использованием pandas
def load_data(file):
    try:
        data = pd.read_excel(file, engine='openpyxl')
        return data
    except Exception as e:
        st.write(f"Ошибка при загрузке файла: {e}")
        return None

# Загрузка модели
model_path = os.path.join(model_directory, get_next_model_name())

# Создание интерфейса
st.title("Неро")
st.write("Система для поиска совпадений между названиями товаров")

# Инициализация модели
main_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
fine_tuned_model = None  # Объявляем переменную, но инициализируем позже, если понадобится

# Кнопка для дообучения модели
if st.button("Дообучить модель", key="fine_tune_model"):
    fine_tuned_model = main_model  # Замените на вашу логику дообучения
    st.write("Модель дообучена.")

# Кнопка для сохранения модели с увеличением версии
if st.button("Сохранить модель", key="save_model"):
    if fine_tuned_model:
        save_model(fine_tuned_model, model_path)
    else:
        st.write("Нет дообученной модели для сохранения.")

# Выбор модели для использования
model_files = sorted(os.listdir(model_directory))
selected_model = st.selectbox("Выберите модель", model_files)

# Загрузка выбранной модели
if selected_model:
    model = load_model(os.path.join(model_directory, selected_model))
    st.write(f"Используется модель: {selected_model}")

@st.cache_data
def compute_embeddings(names, _model):
    return _model.encode(names, convert_to_tensor=True)

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

def find_matching_names(our_product_names, competitor_product_names, threshold=0.8, model=None):
    if model is None:
        raise ValueError("Модель не передана")

    main_embeddings_our = compute_embeddings(our_product_names, model)
    main_embeddings_competitor = compute_embeddings(competitor_product_names, model)

    if fine_tuned_model:
        fine_tuned_embeddings_our = compute_embeddings(our_product_names, fine_tuned_model)
        fine_tuned_embeddings_competitor = compute_embeddings(competitor_product_names, fine_tuned_model)
        similarities = (util.pytorch_cos_sim(main_embeddings_our, main_embeddings_competitor) +
                        util.pytorch_cos_sim(fine_tuned_embeddings_our, fine_tuned_embeddings_competitor)) / 2
    else:
        similarities = util.pytorch_cos_sim(main_embeddings_our, main_embeddings_competitor)

    matching_names = []
    for i, our_name in enumerate(our_product_names):
        for j, competitor_name in enumerate(competitor_product_names):
            if similarities[i][j] >= threshold:
                matching_names.append((our_name, competitor_name, similarities[i][j].item()))

    matching_names = sorted(matching_names, key=lambda x: (-x[2], x[0]))

    return matching_names

# Интерфейс для загрузки файлов
st.title('Сравнение названий товаров')
st.write('Загрузите файлы с названиями ваших товаров и товаров конкурента для поиска совпадений.')

our_file = st.file_uploader('Загрузите файл с вашими товарами', type=['xlsx'])
competitor_file = st.file_uploader('Загрузите файл с товарами конкурента', type=['xlsx'])
examples_file = st.file_uploader('Загрузите файл с примерами соответствий (необязательно)', type=['xlsx'])

# Переменные для данных
our_data, competitor_data = None, None

# Функция для выполнения всех шагов вычислений
# Функция для выполнения вычислений и отображения результатов
def run_computation(our_data, our_column, competitor_data, competitor_column, examples_data=None, examples_column_our=None, examples_column_competitor=None):
    if our_data is None or competitor_data is None:
        st.write("Не удалось загрузить данные. Проверьте файлы и попробуйте снова.")
        return

    # Извлечение данных из выбранных колонок
    our_product_names = our_data[our_column].tolist()
    competitor_product_names = competitor_data[competitor_column].tolist()

    examples = []
    if examples_data is not None and examples_column_our and examples_column_competitor:
        for index, row in examples_data.iterrows():
            example = InputExample(texts=[row[examples_column_our], row[examples_column_competitor]])
            examples.append(example)
        st.write('Дообучение модели на примерах...')
        # Запуск дообучения модели (замените fine_tuned_model на вашу модель)
        train_model(fine_tuned_model, examples)

    st.write('Вычисление эмбеддингов и поиск совпадений...')
    
    # Здесь вызывается функция для нахождения совпадений (замените main_model на вашу модель)
    matching_names = find_matching_names(our_product_names, competitor_product_names, threshold=0.8, model=main_model)

    if matching_names:
        results_df = pd.DataFrame(matching_names, columns=['Наше название', 'Название конкурента', 'Схожесть'])
        st.write('Все совпадения:')
        st.dataframe(results_df)
    else:
        st.write('Совпадений не найдено.')

    # Интерфейс для загрузки файлов и выбора колонок
    st.title('Сравнение названий товаров')
    st.write('Загрузите файлы с названиями ваших товаров и товаров конкурента для поиска совпадений.')

    # Добавляем уникальные ключи для каждого file_uploader
    our_file = st.file_uploader('Загрузите файл с вашими товарами', type=['xlsx'], key='our_file')
    competitor_file = st.file_uploader('Загрузите файл с товарами конкурента', type=['xlsx'], key='competitor_file')
    examples_file = st.file_uploader('Загрузите файл с примерами соответствий (необязательно)', type=['xlsx'], key='examples_file')

    # Если файлы загружены, предложим выбрать колонки для работы
    if our_file and competitor_file:
        our_data = load_data(our_file)
        competitor_data = load_data(competitor_file)

        if our_data is not None and competitor_data is not None:
            our_columns = our_data.columns.tolist()
            competitor_columns = competitor_data.columns.tolist()

        # Добавляем уникальные ключи для selectbox
            our_column = st.selectbox('Выберите колонку с названиями ваших товаров', our_columns, key='our_column')
            competitor_column = st.selectbox('Выберите колонку с названиями товаров конкурента', competitor_columns, key='competitor_column')

            # Если загружен файл с примерами, предложим выбрать колонки
            if examples_file:
                examples_data = load_data(examples_file)
                if examples_data is not None:
                    examples_column_our = st.selectbox('Выберите колонку с вашими названиями из примеров', examples_data.columns.tolist(), key='examples_column_our')
                    examples_column_competitor = st.selectbox('Выберите колонку с названиями конкурента из примеров', examples_data.columns.tolist(), key='examples_column_competitor')
                    run_computation(our_data, our_column, competitor_data, competitor_column, examples_data, examples_column_our, examples_column_competitor)
            else:
                run_computation(our_data, our_column, competitor_data, competitor_column)

# Кнопка для запуска вычислений
if st.button("Начать вычисления", key="run_computation"):
    if our_file and competitor_file:
        our_data = load_data(our_file)
        competitor_data = load_data(competitor_file)

        if our_data is not None and competitor_data is not None:
            our_columns = our_data.columns.tolist()
            competitor_columns = competitor_data.columns.tolist()

            our_column = st.selectbox('Выберите колонку с названиями ваших товаров', our_columns, key='our_column')
            competitor_column = st.selectbox('Выберите колонку с названиями товаров конкурента', competitor_columns, key='competitor_column')

            if examples_file:
                examples_data = load_data(examples_file)
                if examples_data is not None:
                    examples_column_our = st.selectbox('Выберите колонку с вашими названиями из примеров', examples_data.columns.tolist(), key='examples_column_our')
                    examples_column_competitor = st.selectbox('Выберите колонку с названиями конкурента из примеров', examples_data.columns.tolist(), key='examples_column_competitor')
                    run_computation(our_data, our_column, competitor_data, competitor_column, examples_data, examples_column_our, examples_column_competitor)
            else:
                run_computation(our_data, our_column, competitor_data, competitor_column)