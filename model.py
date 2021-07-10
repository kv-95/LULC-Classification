import standard
import dataset
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

num_classes = len(dataset.train_ds.class_names)

model = Sequential([
  standard.data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#TRAINING 

epochs = 15
history = model.fit(
  standard.train_ds,
  validation_data = standard.val_ds,
  epochs=epochs
)
