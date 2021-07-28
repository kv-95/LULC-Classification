from tensorflow.python.keras.layers.core import Activation, Dense
import standard
import dataset
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_visible_devices(gpus[0], 'GPU')

num_classes = len(dataset.train_ds.class_names)

model = Sequential([
  standard.data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),

  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
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
