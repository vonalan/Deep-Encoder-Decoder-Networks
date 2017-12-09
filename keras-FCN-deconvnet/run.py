import os 
import sys 
import itertools

import numpy as np 
import tensorflow as tf 
from keras import losses, optimizers
from keras import backend as K

import PIL
from PIL import Image 

import deconvnet 
# import utils

# no GPU supplied
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = deconvnet.deconvnet()
print(model.input)
print(model.output) 

optimizer = optimizers.SGD(lr=1e-5, momentum=9e-1, decay=1e-6)
model.compile(loss=losses.mean_squared_logarithmic_error, optimizer=optimizer)

image_dir = 'images'
image_list = os.listdir(image_dir)
images = np.zeros((len(image_list), 224, 224, 3))
for i, image in enumerate(image_list): 
    image_path = os.path.join(image_dir, image)
    images[i] = np.array(Image.open(image_path).resize((224,224)))

np.random.shuffle(images)
train_images = images[:80]
test_images = images[80:]
print(images.shape, train_images.shape, test_images.shape)

for i in itertools.count(): 
    model.fit(train_images, train_images, epochs=100, batch_size=4)
    pred_rgbs = model.predict(test_images)
    print(pred_rgbs.shape)
    for j in range(pred_rgbs.shape[0]): 
        raw_image = test_images[j]
        pred_image = pred_rgbs[j]
        combined_image = np.concatenate((raw_image, pred_image), axis=1).astype(np.uint8)
        print(combined_image.shape)
        # print(combined_image)
        combined_image = Image.fromarray(combined_image)
        # combined_image.show()
        combined_image.save('outputs/%d.png'%(j+1))
