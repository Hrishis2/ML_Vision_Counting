import tensorflow as tf
import os
import numpy as np



batch_size = 32
img_height = 1080
img_width = 1920

data_dir = os.getcwd() + "/fingers_dataset"

ds_train = tf.keras.utils.image_dataset_from_directory(
  data_dir+"/training_set",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

ds_test = tf.keras.utils.image_dataset_from_directory(
  data_dir+"/testing_set",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)



# tf.image.resize(
#     ds_train,
#     (1080,1920),
#     method=tf.image.ResizeMethod.BILINEAR,
#     preserve_aspect_ratio=True,
#     antialias=False,
#     name=None
# )

# tf.image.resize(
#     ds_test,
#     (1080,1920),
#     method=tf.image.ResizeMethod.BILINEAR,
#     preserve_aspect_ratio=True,
#     antialias=False,
#     name=None
# )

train_num_examples = len(np.concatenate([i for x, i in ds_train], axis=0))





def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [256,256])
    return tf.cast(image, tf.float32) / 255., label


batch_size = 67

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(train_num_examples)

ds_train = ds_train.unbatch()
ds_train = ds_train.batch(batch_size)

ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)



ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_test = ds_test.unbatch()
ds_test = ds_test.batch(batch_size)

ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)





model = tf.keras.models.Sequential([
    # tf.keras.layers.Reshape(target_shape=(384*216,), input_shape=(216,384)),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=192, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
#                  activation='relu'),
#   tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
#                  activation='relu'),
#   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
model.fit(
    ds_train,
    epochs=12,
    validation_data=ds_test,
)


model.save('saved_model/model')