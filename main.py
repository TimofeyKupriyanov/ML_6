from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Настройка генератора данных
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# загрузка тренировачных данных
train_data = train_datagen.flow_from_directory(
   'dataset/',
   target_size=(64, 64),
   batch_size=32,
   class_mode='categorical',
   subset='training'
)

# загрузка валидационных данных
val_data = train_datagen.flow_from_directory(
   'dataset/',
   target_size=(64, 64),
   batch_size=32,
   class_mode='categorical',
   subset='validation'
)

# Создание модели нейросети
model = keras.models.Sequential([
   keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
   keras.layers.MaxPooling2D(2,2),
   keras.layers.Conv2D(64, (3,3), activation='relu'),
   keras.layers.MaxPooling2D(2,2),
   keras.layers.Flatten(),
   keras.layers.Dense(128, activation='relu'),
   keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# компиляция и обучение
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

# точность
loss, accuracy = model.evaluate(val_data)
print(f'Точность модели: {accuracy * 100:.2f}%')

# работа с тестовым изображением
img = image.load_img('Vertical.jpg', target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Предсказание класса
prediction = model.predict(img_array)
class_names = list(train_data.class_indices.keys())
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_names[predicted_class_index]
print(f'Предсказанный класс: {predicted_class_name}')
