import cv2
import numpy as np
from tensorflow.keras.models import load_model

# NSFW modelini yuklash
model = load_model('/Users/uzmacbook/Portfolio/Prohibited-Content/nsfw_model.h5')

def is_nsfw(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0
    predictions = model.predict(image)
    return predictions[0][0] > 0.5  # Agar 50% dan yuqori bo'lsa, NSFW deb hisoblanadi

def blur_image(image_path):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    cv2.imwrite('blurred_' + image_path, blurred_image)
    return 'blurred_' + image_path

# Tasvirni tekshirish va xiralashtirish
image_path = '2024-07-16 12.09.19.jpg'
if is_nsfw(image_path):
    blurred_image_path = blur_image(image_path)
    print(f'Nomaqbul kontent topildi. Tasvir xiralashtirildi: {blurred_image_path}')
else:
    print('Tasvir nomaqbul emas.')
