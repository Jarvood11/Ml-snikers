# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:40:42 2024

@author: ewent
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print((train_images, train_labels))
#print((test_images, test_labels))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


#model.compile(optimizer = tf.keras.optimizers.Adam(),
#            loss = 'sparse_categorical_crossentropy',
#           metrics=['accuracy'])
   
   
model.compile(optimizer = 'adam',
           loss = 'sparse_categorical_crossentropy',
           metrics=['accuracy'])


model.summary()
  #model.fit(test_images,test_labels,epochs=6)
model.fit(train_images,train_labels,epochs=6)

test_loss, test_acc=model.evaluate(test_images,test_labels)

print("Model accuracy based on test data: ", test_acc)

print("Model test_loss based on test data: ", test_loss)


train_loss, train_acc=model.evaluate(train_images,train_labels)

print("Model accuracy based on train data: ", train_acc)

print("Model train_loss based on train data: ", train_loss)

# Model accuracy based on test data:  0.8802000284194946
#Model test_loss based on test data:  0.3361927270889282
 # 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 687us/step - accuracy: 0.9064 - loss: 0.2513
#  Model accuracy based on train data:  0.906416654586792
 #  Model train_loss based on train  data:  0.25329330563545227

