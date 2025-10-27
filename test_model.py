# test_model.py
import sys, numpy as np
from PIL import Image
import tensorflow as tf

if len(sys.argv) < 2:
    print("Usage: python test_model.py path/to/image.png"); exit()

img_path = sys.argv[1]
model = tf.keras.models.load_model("traffic_sign_model.h5")
img = Image.open(img_path).convert("RGB").resize((32,32))
arr = np.expand_dims(np.array(img)/255.0, 0)
pred = model.predict(arr)[0]
idx = int(pred.argmax()); conf = float(pred.max())
print(f"Predicted class: {idx}, confidence: {conf:.3f}")
