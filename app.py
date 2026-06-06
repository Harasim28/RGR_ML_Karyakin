"""
Streamlit дашборд для инференса моделей ML
Датасет: smoke_detector (классификация Fire Alarm)
"""

import os
import json

import numpy as np
import pandas as pd
import joblib

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(
    page_title="ML Inference Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "smoke_detector_task_main.csv")

@st.cache_resource
def load_models():
    """Загрузка всех моделей и scaler"""
    models = {
        'ML1: LogisticRegression': joblib.load(os.path.join(MODELS_DIR, 'ml1_logreg.joblib')),
        'ML2: GradientBoosting': joblib.load(os.path.join(MODELS_DIR, 'ml2_boosting.joblib')),
        'ML3: CatBoost': joblib.load(os.path.join(MODELS_DIR, 'ml3_catboost.joblib')),
        'ML4: RandomForest': joblib.load(os.path.join(MODELS_DIR, 'ml4_rf.joblib')),
        'ML5: Stacking': joblib.load(os.path.join(MODELS_DIR, 'ml5_stack.joblib')),
        'ML6: MLP (FCNN)': joblib.load(os.path.join(MODELS_DIR, 'ml6_mlp.joblib')),
    }
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    metrics = pd.read_csv(os.path.join(MODELS_DIR, 'metrics_summary.csv'))
    return models, scaler, feature_names, metrics

# Загружаем модели один раз
models, scaler, feature_names, metrics_df = load_models()

# ============================================================
# Боковое меню
# ============================================================
st.sidebar.title("ML Inference Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Навигация:",
    ["О разработчике", "О датасете", "Визуализации", "Инференс"]
)
st.sidebar.markdown("---")
st.sidebar.info("Выберите страницу для навигации по дашборду.")

# ============================================================
# Страница 1: О разработчике
# ============================================================
if page == "О разработчике":
    st.title("Информация о разработчике")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        photo_path = os.path.join(BASE_DIR, "assets", "photo.jpg")
        if os.path.exists(photo_path):
            st.image(photo_path, width=200, caption="Разработчик")
        else:
            st.info("Добавьте фото в папку `assets/photo.jpg`")
            # Placeholder
            st.markdown("<div style='width:200px;height:200px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;border-radius:10px;'><span style='color:#999;'>Фото</span></div>", unsafe_allow_html=True)
    
    with col2:
        st.header("Расчетно-графическая работа")
        st.write("**Тема:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")
        st.write("**Дисциплина:** Машинное обучение и большие данные")
        st.markdown("---")
        st.write("**ФИО:** Карякин Алексей Петрович")
        st.write("**Группа:** МО-241")
        st.write("**Университет:** ОмГТУ")
    
    st.markdown("---")
    st.subheader("Краткое описание проекта")
    st.write("""
    Данный дашборд позволяет:
    - Ознакомиться с набором данных о датчиках дыма (smoke detector)
    - Провести разведочный анализ данных (EDA)
    - Визуализировать зависимости в данных
    - Выполнить инференс (предсказание) с помощью 6 обученных моделей ML
    
    **Задача классификации:** определение наличия пожара (Fire Alarm: Yes/No)
    на основе показаний датчиков (температура, влажность, концентрация газов и т.д.)
    """)
    
    st.subheader("Модели ML")
    st.dataframe(metrics_df.style.highlight_max(subset=['accuracy','precision','recall','f1'], color='green'))

# ============================================================
# Страница 2: О датасете
# ============================================================
elif page == "О датасете":
    st.title("О наборе данных")
    st.markdown("---")
    
    # Загрузка датасета
    df = pd.read_csv(DATA_PATH)
    df_clean = df.drop(['Unnamed: 0', 'UTC', 'CNT'], axis=1, errors='ignore')
    df_clean['Fire Alarm'] = df_clean['Fire Alarm'].map({'No': 0, 'Yes': 1})
    df_clean = df_clean.dropna()
    
    st.subheader("Описание предметной области")
    st.write("""
    Набор данных **Smoke Detection** содержит показания множества датчиков, 
    установленных в помещении для обнаружения пожара. 
    
    **Целевая переменная:** `Fire Alarm` (Yes/No) — сработал ли датчик пожарной сигнализации.
    
    **Признаки:**
    - **Temperature[C]** — температура в градусах Цельсия
    - **Humidity[%]** — относительная влажность воздуха
    - **TVOC[ppb]** — общее содержание летучих органических соединений (частицы на миллиард)
    - **eCO2[ppm]** — эквивалентная концентрация CO2 (частицы на миллион)
    - **Raw H2** — сырые данные датчика водорода
    - **Raw Ethanol** — сырые данные датчика этанола
    - **Pressure[hPa]** — атмосферное давление
    - **PM1.0, PM2.5** — концентрация твердых частиц (particulate matter)
    - **NC0.5, NC1.0, NC2.5** — числовая концентрация частиц разного размера
    """)
    
    st.subheader("Основные характеристики датасета")
    col1, col2, col3 = st.columns(3)
    col1.metric("Количество записей", len(df_clean))
    col2.metric("Количество признаков", len(df_clean.columns) - 1)
    col3.metric("Пропусков (%)", f"{df_clean.isnull().sum().sum() / (df_clean.shape[0]*df_clean.shape[1]) * 100:.2f}%")
    
    st.markdown("---")
    st.subheader("Первые 10 строк данных")
    st.dataframe(df_clean.head(10))
    
    st.subheader("Статистическое описание")
    st.dataframe(df_clean.describe().round(2))
    
    st.subheader("Распределение классов")
    col_counts = df_clean['Fire Alarm'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **No Fire (0):** {col_counts[0]} ({col_counts[0]/len(df_clean)*100:.1f}%)")
        st.write(f"- **Fire (1):** {col_counts[1]} ({col_counts[1]/len(df_clean)*100:.1f}%)")
    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        colors = ['#4CAF50', '#F44336']
        ax.pie(col_counts, labels=['No Fire', 'Fire'], autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    st.subheader("Предобработка данных")
    st.write("""
    1. Удалены служебные колонки (`Unnamed: 0`, `UTC`, `CNT`)
    2. Целевая переменная преобразована из строк в числа (`No` → 0, `Yes` → 1)
    3. Удалены строки с пропущенными значениями
    4. Применено масштабирование признаков (`StandardScaler`) перед обучением моделей
    """)

# ============================================================
# Страница 3: Визуализации
# ============================================================
elif page == "Визуализации":
    st.title("Визуализации данных")
    st.markdown("---")
    
    df = pd.read_csv(DATA_PATH)
    df_clean = df.drop(['Unnamed: 0', 'UTC', 'CNT'], axis=1, errors='ignore')
    df_clean['Fire Alarm'] = df_clean['Fire Alarm'].map({'No': 0, 'Yes': 1})
    df_clean = df_clean.dropna()
    
    # Для ускорения визуализации берем случайную подвыборку
    df_sample = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)
    
    st.subheader("1. Распределение температуры по классам")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_sample, x='Temperature[C]', hue='Fire Alarm', kde=True, bins=50, ax=ax1, palette=['#4CAF50', '#F44336'])
    ax1.set_title('Распределение температуры (Fire Alarm)')
    ax1.legend(['Fire', 'No Fire'])
    st.pyplot(fig1)
    st.write("На графике видно, что при пожарах температура значительно выше.")
    
    st.markdown("---")
    st.subheader("2. Влажность по классам")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_sample, x='Fire Alarm', y='Humidity[%]', ax=ax2, palette=['#4CAF50', '#F44336'])
    ax2.set_xticklabels(['No Fire', 'Fire'])
    ax2.set_title('Влажность по классам')
    st.pyplot(fig2)
    st.write("При пожарах влажность обычно ниже из-за испарения.")
    
    st.markdown("---")
    st.subheader("3. Корреляционная матрица признаков")
    numeric_df = df_sample.select_dtypes(include=[np.number])
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax3, square=True)
    ax3.set_title('Корреляционная матрица')
    st.pyplot(fig3)
    st.write("Тепловая карта показывает сильные корреляции между газовыми датчиками и целевой переменной.")
    
    st.markdown("---")
    st.subheader("4. TVOC vs eCO2 (цвет = класс)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_sample, x='TVOC[ppb]', y='eCO2[ppm]', hue='Fire Alarm', 
                    alpha=0.6, ax=ax4, palette=['#4CAF50', '#F44336'])
    ax4.set_title('TVOC vs eCO2 (цвет = наличие пожара)')
    ax4.legend(['No Fire', 'Fire'])
    st.pyplot(fig4)
    st.write("При пожарах наблюдаются резкие всплески концентрации летучих органических соединений и CO2.")

# ============================================================
# Страница 4: Инференс
# ============================================================
elif page == "Инференс":
    st.title("Инференс моделей ML")
    st.markdown("---")
    
    st.subheader("Выберите способ ввода данных")
    input_method = st.radio("", ["Загрузка CSV файла", "Ручной ввод признаков"], horizontal=True)
    
    st.markdown("---")
    
    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель для предсказания:",
        list(models.keys())
    )
    
    selected_model = models[model_choice]
    
    # Показываем метрики выбранной модели
    model_metrics = metrics_df[metrics_df['name'] == model_choice].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{model_metrics['precision']:.4f}")
    col3.metric("Recall", f"{model_metrics['recall']:.4f}")
    col4.metric("F1-score", f"{model_metrics['f1']:.4f}")
    
    st.markdown("---")
    
    if input_method == "Загрузка CSV файла":
        uploaded_file = st.file_uploader("Загрузите CSV файл с признаками", type=['csv'])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(input_df.head())
            
            # Проверка наличия нужных колонок
            missing_cols = set(feature_names) - set(input_df.columns)
            if missing_cols:
                st.error(f"В загруженном файле отсутствуют колонки: {missing_cols}")
            else:
                # Берем только нужные признаки в нужном порядке
                input_features = input_df[feature_names]
                input_scaled = scaler.transform(input_features)
                
                if st.button("Предсказать", type="primary"):
                    with st.spinner("Выполняется предсказание..."):
                        predictions = selected_model.predict(input_scaled)
                        
                        # Вероятности, если модель поддерживает
                        if hasattr(selected_model, 'predict_proba'):
                            probabilities = selected_model.predict_proba(input_scaled)
                            proba_fire = probabilities[:, 1]
                        else:
                            proba_fire = [None] * len(predictions)
                        
                        # Формируем результат
                        result_df = input_df.copy()
                        result_df['Prediction'] = ['Fire' if p == 1 else 'No Fire' for p in predictions]
                        if proba_fire[0] is not None:
                            result_df['Fire Probability'] = [f"{p:.2%}" for p in proba_fire]
                        
                        st.success("Предсказание выполнено!")
                        st.write("### Результаты:")
                        st.dataframe(result_df)
                        
                        # Статистика
                        fire_count = sum(predictions)
                        no_fire_count = len(predictions) - fire_count
                        st.write(f"**Итого:** {fire_count} случаев пожара, {no_fire_count} — норма")
                        
                        # График
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.bar(['No Fire', 'Fire'], [no_fire_count, fire_count], color=['#4CAF50', '#F44336'])
                        ax.set_title('Распределение предсказаний')
                        st.pyplot(fig)
    
    else:  # Ручной ввод
        st.write("Введите значения признаков:")
        
        # Словарь для хранения введенных значений
        user_input = {}
        
        # Два столбца для компактности
        col_left, col_right = st.columns(2)
        
        # Примерные min/max из датасета
        ranges = {
            'Temperature[C]': (-10.0, 60.0, 20.0),
            'Humidity[%]': (0.0, 100.0, 50.0),
            'TVOC[ppb]': (0.0, 2000.0, 0.0),
            'eCO2[ppm]': (300.0, 1000.0, 400.0),
            'Raw H2': (12000.0, 14000.0, 12900.0),
            'Raw Ethanol': (18000.0, 21000.0, 19500.0),
            'Pressure[hPa]': (930.0, 950.0, 939.0),
            'PM1.0': (0.0, 30.0, 0.0),
            'PM2.5': (0.0, 30.0, 0.0),
            'NC0.5': (0.0, 200.0, 10.0),
            'NC1.0': (0.0, 30.0, 1.5),
            'NC2.5': (0.0, 3.0, 0.04),
        }
        
        for i, feat in enumerate(feature_names):
            min_val, max_val, default = ranges.get(feat, (0.0, 100.0, 0.0))
            if i % 2 == 0:
                with col_left:
                    user_input[feat] = st.number_input(
                        f"{feat}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=(max_val - min_val) / 100
                    )
            else:
                with col_right:
                    user_input[feat] = st.number_input(
                        f"{feat}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=(max_val - min_val) / 100
                    )
        
        st.markdown("---")
        
        if st.button("Предсказать", type="primary"):
            # Формируем массив
            input_array = np.array([[user_input[feat] for feat in feature_names]])
            input_scaled = scaler.transform(input_array)
            
            prediction = selected_model.predict(input_scaled)[0]
            
            # Вероятность
            if hasattr(selected_model, 'predict_proba'):
                proba = selected_model.predict_proba(input_scaled)[0]
                proba_fire = proba[1]
                proba_no_fire = proba[0]
            else:
                proba_fire = None
            
            # Вывод результата
            if prediction == 1:
                st.error("**ВНИМАНИЕ! Обнаружены признаки пожара!**")
                st.write("### Fire Alarm: **YES**")
            else:
                st.success("**Норма. Пожар не обнаружен.**")
                st.write("### Fire Alarm: **NO**")
            
            if proba_fire is not None:
                st.write(f"**Вероятность пожара:** {proba_fire:.2%}")
                st.write(f"**Вероятность нормы:** {proba_no_fire:.2%}")
                
                # Прогресс-бар
                st.progress(float(proba_fire))
            
            st.markdown("---")
            st.write("**Введенные данные:**")
            st.json(user_input)
