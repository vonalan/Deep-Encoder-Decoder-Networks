import itertools

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.losses import categorical_crossentropy

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.xception import Xception

def load_classes():
    return []

image_size = (299, 299, 3)
classes = load_classes()
# classes = 51

base_model = InceptionV3(weights=None, include_top=False, pooling='max')
x = base_model.output
x = Dense(1024, activation='relu', name='fc_layer')(x)
predictions = Dense(len(classes), activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=predictions, name='inception_v3')


# step 01
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=RMSprop(), loss=categorical_crossentropy)

model.fit_generator()


# step 02
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers:
    layer.trainable=True
model.compile(optimizer=SGD(lr=1e-4, momentum=1e-9), loss=categorical_crossentropy)

model.fit_generator()


def create_image_tree(sround=1):
    pass

# todo: tensorflow dataset
def create_data_generator(image_tree, batch_size=4):
    images = []
    labels = []
    for class_name, sub_image_tree in image_tree.items():
        for video_name, images in sub_image_tree.items():
            images.extend(images)
            labels.extend([classes.index(class_name)] * len(images))
    assert len(images) == len(labels)

    import math
    from PIL import Image

    num_batchs = int(math.ceil(len(images)/float(batch_size)))
    for batch in range(num_batchs):
        sidx = max(0, batch * batch_size)
        eidx = min((batch + 1) * batch_size)

        cur_batch_size = eidx - sidx
        image_batch = np.zeros((cur_batch_size, 299, 299, 3))
        label_batch = np.zeros((cur_batch_size, 1))
        for idx in range(cur_batch_size):
            image = Image.open(images[sidx + idx]).resize((299, 299))
            label = labels[sidx + idx]
            image_batch[idx] = image
            label_batch[idx] = label
        yield image_batch, label_batch

image_tree = create_image_tree()

train_data_generator = create_data_generator(image_tree['train'])
valid_data_generator = create_data_generator(image_tree['valid'])
infer_date_generator = create_data_generator(image_tree['infer'])

for i in itertools.count():
    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=1,
                        epochs=1,
                        validation_data=valid_data_generator,
                        validation_steps=1)
    model.save_weights('../model/inception_v3.h5')