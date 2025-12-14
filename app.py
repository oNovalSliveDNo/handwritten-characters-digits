# =========================================================
# app.py — Streamlit UI для классификации рукописных символов
# (0–9, A–Z, a–z) с моделью handwritten_cnn_best.pt
# =========================================================

import os
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

from psycopg2.extras import Json
import torch
import torch.nn as nn
import torch.nn.functional as F

from dotenv import load_dotenv
from db.db import get_conn, put_conn  # твой пул соединений с БД
import random

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Characters Recognition",
    layout="centered"
)

st.title("✍️ Распознавание рукописных символов (0–9, A–Z, a–z)")

# ---------------------------------------------------------
# Загрузка переменных окружения
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# CNN модель (ТОЧНО ТА ЖЕ, что при обучении)
# ---------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------
# Загрузка модели (кэшируется Streamlit)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("notebooks/handwritten_cnn_best.pt", map_location="cpu")

    model = CNN(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        checkpoint["class_to_idx"],
        checkpoint["idx_to_class"]
    )


model, class_to_idx, idx_to_class = load_model()


# ---------------------------------------------------------
# Utils: preprocessing изображения
# ---------------------------------------------------------
def preprocess_pil_28x28(pil_img, invert=False, center_crop=True):
    """
    Приводит изображение к формату, ожидаемому моделью:
    grayscale → 28x28 → нормализация → tensor
    """
    img = pil_img.convert("L")

    if invert:
        img = ImageOps.invert(img)

    if center_crop:
        arr = np.array(img)
        mask = arr < 250
        if mask.any():
            ys, xs = np.where(mask)
            img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    img = ImageOps.pad(img, (28, 28), color=255)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, 1))  # [1,1,28,28]

    return torch.tensor(arr, dtype=torch.float32), img


def preprocess_db_test(img_bytes):
    """
    ⚠️ preprocessing ДЛЯ TEST ИЗ БД
    1 в 1 как в Jupyter
    """
    # BYTEA → numpy uint8
    img_array = np.frombuffer(img_bytes, np.uint8)

    # decode → grayscale
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # resize (на случай если вдруг не 28x28)
    img = cv2.resize(img, (28, 28))

    # normalization
    img = img.astype(np.float32) / 255.0

    # (1, 28, 28)
    img = np.expand_dims(img, axis=0)

    # → torch
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1,1,28,28]

    return x, img



# ---------------------------------------------------------
# Utils: inference
# ---------------------------------------------------------
def predict_tensor(x):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    pred_char = idx_to_class[pred_idx]

    return pred_char, probs


def log_inference(true_label, predicted_label, probs, source="random_test"):
    confidence = float(np.max(probs))

    probs_dict = {
        idx_to_class[i]: float(probs[i])
        for i in range(len(probs))
    }

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ml.inference_logs
                (true_label, predicted_label, confidence, probabilities, source)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                true_label,
                predicted_label,
                confidence,
                Json(probs_dict),
                source
            ))
            conn.commit()
    finally:
        put_conn(conn)



# ---------------------------------------------------------
# Блок 1 — Тестовые данные из БД
# ---------------------------------------------------------
st.subheader("📦 Случайные тестовые примеры из базы данных")


def fetch_random_image_for_class(class_name):
    """
    Возвращает одно случайное изображение для заданного класса
    """
    table = (
        f"class_{class_name}_caps" if class_name.isupper()
        else f"class_{class_name}"
    )

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT image_data
                FROM train."{table}"
                ORDER BY random()
                LIMIT 1
                '''
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        put_conn(conn)


def fetch_random_samples(limit=20):
    conn = get_conn()
    try:
        out = []
        with conn.cursor() as cur:
            cur.execute("""
                SELECT image_data, table_name
                FROM (
                    SELECT image_data, table_name,
                           ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY random()) rn
                    FROM (
                        SELECT image_data, table_name
                        FROM information_schema.tables t
                        JOIN train.class_0 c ON false
                    ) s
                ) q
                WHERE rn = 1
                LIMIT %s
            """, (limit,))
        return out
    finally:
        put_conn(conn)


# кнопка обновления
if st.button("🔄 Обновить примеры"):
    st.session_state["random_classes"] = random.sample(list(class_to_idx.keys()), 5)

# при первом запуске
if "random_classes" not in st.session_state:
    st.session_state["random_classes"] = random.sample(list(class_to_idx.keys()), 5)

cols = st.columns(5)
clicked = None

for col, cls in zip(cols, st.session_state["random_classes"]):
    img_bytes = fetch_random_image_for_class(cls)

    if img_bytes is None:
        col.write("—")
        continue

    # BYTEA → numpy → PIL
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    pil_img = Image.fromarray(img)

    col.image(pil_img, caption=f"Class: {cls}", use_container_width=True)

    if col.button(f"Выбрать {cls}", key=f"pick_{cls}"):
        x, _ = preprocess_db_test(img_bytes)
        pred_char, probs = predict_tensor(x)
        clicked = (cls, pred_char, probs)

if clicked:
    true_cls, pred_cls, probs = clicked

    # логируем тестовый инференс
    log_inference(
        true_cls,
        pred_cls,
        probs,
        source="random_test"
    )

    if true_cls == pred_cls:
        st.success(f"✅ Верно! True: **{true_cls}**, Pred: **{pred_cls}**")
    else:
        st.error(f"❌ Ошибка. True: **{true_cls}**, Pred: **{pred_cls}**")

    st.bar_chart({
        idx_to_class[i]: float(probs[i])
        for i in range(len(probs))
    })

# ---------------------------------------------------------
# Блок 2 — Загрузка своего изображения
# ---------------------------------------------------------
st.divider()
st.subheader("📤 Загрузите своё изображение")

file = st.file_uploader("PNG / JPG / JPEG", type=["png", "jpg", "jpeg"])

left, right = st.columns(2)
with left:
    invert = st.checkbox("Инвертировать цвета", value=False)
with right:
    center_crop = st.checkbox("Автоцентрирование", value=True)

if file is not None:
    pil = Image.open(file)

    x, pre_img = preprocess_pil_28x28(
        pil,
        invert=invert,
        center_crop=center_crop
    )

    c1, c2 = st.columns(2)
    c1.image(pil, caption="Исходное изображение", use_container_width=True)
    c2.image(pre_img, caption="28×28 для модели", use_container_width=True)

    if st.button("🔍 Распознать"):
        pred_char, probs = predict_tensor(x)

        st.success(f"Предсказанный символ: **{pred_char}**")

        st.bar_chart({
            idx_to_class[i]: float(probs[i])
            for i in range(len(probs))
        })


def fetch_global_accuracy():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE true_label = predicted_label)::float
                    / NULLIF(COUNT(*), 0)
                FROM ml.inference_logs
                WHERE source = 'random_test'
            """)
            row = cur.fetchone()
            return row[0] if row else 0.0
    finally:
        put_conn(conn)


def fetch_top_errors(limit=10):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    true_label,
                    predicted_label,
                    COUNT(*) AS cnt
                FROM ml.inference_logs
                WHERE true_label != predicted_label
                GROUP BY true_label, predicted_label
                ORDER BY cnt DESC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()
    finally:
        put_conn(conn)


def fetch_confusion_matrix():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT true_label, predicted_label, COUNT(*)
                FROM ml.inference_logs
                GROUP BY true_label, predicted_label
            """)
            return cur.fetchall()
    finally:
        put_conn(conn)


st.divider()
st.subheader("📊 Аналитика модели (по логам инференса)")

acc = fetch_global_accuracy()

if acc is not None:
    st.metric(
        label="🌍 Global Accuracy",
        value=f"{acc:.4f}"
    )
else:
    st.info("Пока нет данных для расчёта accuracy")

errors = fetch_top_errors(limit=10)

if errors:
    st.subheader("❌ Самые частые ошибки модели")

    st.table([
        {
            "True label": t,
            "Predicted label": p,
            "Count": c
        }
        for t, p, c in errors
    ])
else:
    st.info("Ошибок пока нет — либо нет данных")

from collections import defaultdict
import pandas as pd

cm_data = fetch_confusion_matrix()

if cm_data:
    st.subheader("📊 Confusion Matrix")

    labels = sorted(class_to_idx.keys())
    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels
    )

    for true_lbl, pred_lbl, cnt in cm_data:
        if true_lbl in matrix.index and pred_lbl in matrix.columns:
            matrix.loc[true_lbl, pred_lbl] += cnt

    st.dataframe(matrix)
else:
    st.info("Недостаточно данных для confusion matrix")
