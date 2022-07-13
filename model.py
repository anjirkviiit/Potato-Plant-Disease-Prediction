import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import warnings
warnings.filterwarnings('ignore')

IMAZE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

dataset = keras.preprocessing.image_dataset_from_directory(
    "LeafData",
    shuffle=True,
    image_size = (IMAZE_SIZE,IMAZE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names

#Splitting the dataset into train, validation and testing sets
def get_dataset_partitionss_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size = len(ds)    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)    
    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds = get_dataset_partitionss_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

#Performing feature scaling for the dataset
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAZE_SIZE, IMAZE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

#Performed data augementation to make model robust for any kind of samples
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

input_shape = (BATCH_SIZE, IMAZE_SIZE, IMAZE_SIZE, 3)
n_classes = 3

model = models.Sequential([
    layers.experimental.preprocessing.Resizing(IMAZE_SIZE, IMAZE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.build(input_shape=input_shape)

model.compile(
    optimizer = 'adam',
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']    
)

history = model.fit(train_ds, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = val_ds)

history.history['accuracy']
Model_accuracy = history.history['accuracy']
Model_loss = history.history['loss']
Model_val_loss = history.history['val_loss']
Model_val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), Model_accuracy, label="Training Accuracy")
plt.plot(range(EPOCHS), Model_val_accuracy, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), Model_loss, label="Training Loss")
plt.plot(range(EPOCHS), Model_val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Saving model to pickle file
model.save('potatoDiseaseModel.h5')

#loading the model
new_model = models.load_model('potatoDiseaseModel.h5')
scores = new_model.evaluate(test_ds)

def predictClass(new_model,img):
    img_array = keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)
    
    predictions = new_model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.argmax(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize = (15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))        
        predicted_class, confidence = predictClass(new_model, images[i].numpy())
        actual_class = class_names[labels[i]]        
        plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")