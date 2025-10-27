import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load trained model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Path of test image
img_path = "sample_test.jpg"  # اپنی کوئی traffic sign image یہاں رکھو

# Load and preprocess image
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print(f"Predicted Traffic Sign Class: {predicted_class}")

# Show image
plt.imshow(image.load_img(img_path))
plt.title(f"Predicted Class: {predicted_class}")
plt.show()
