# app.py
import gradio as gr, numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("traffic_sign_model.h5")

def predict(img: Image.Image):
    img = img.convert("RGB").resize((32,32)); x = np.expand_dims(np.array(img)/255.0,0)
    p = model.predict(x)[0]; i = int(p.argmax()); return {f"Class {i}": float(p[i])}

gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label()).launch()
