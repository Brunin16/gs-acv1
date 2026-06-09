from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

IMG_SIZE = (128, 128)
CLASS_NAMES = ["nowildfire", "wildfire"]  # ordem alfabética = ordem do label binário
MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model.keras"

modelo = tf.keras.models.load_model(MODEL_PATH)


def classificar(imagem):
    """Recebe uma imagem (numpy RGB), retorna probabilidades por classe."""
    img = tf.image.resize(imagem, IMG_SIZE)
    arr = tf.expand_dims(img, 0)
    p_wildfire = float(modelo.predict(arr, verbose=0)[0][0])
    return {"wildfire": p_wildfire, "nowildfire": 1.0 - p_wildfire}


demo = gr.Interface(
    fn=classificar,
    inputs=gr.Image(type="numpy", label="Imagem de satélite"),
    outputs=gr.Label(num_top_classes=2, label="Classificação"),
    title="🛰️🔥 OrbitalFire — Detecção de Queimada em Imagem de Satélite",
    description="Envie uma imagem de satélite. O modelo (CNN treinada do zero) "
                "estima a probabilidade de indício de queimada.",
)

if __name__ == "__main__":
    demo.launch()
