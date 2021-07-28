import dataset
import numpy as np
import tensorflow as tf
import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_visible_devices(gpus[0], 'GPU')

#Load Model
model1 = tf.keras.models.load_model('Model/')

#Checking for random image
test_path = 'image.jpg'

img = keras.preprocessing.image.load_img(
    test_path, target_size=(dataset.img_height, dataset.img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model1.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(dataset.train_ds.class_names[np.argmax(score)], 100 * np.max(score))
    )
