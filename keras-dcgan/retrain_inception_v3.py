import os
import sys
import itertools
import argparse

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.xception import Xception

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--device', default='gpu', type=str)
parser.add_argument('--base_model', default='inception_v3', type=str)
parser.add_argument('--input_shape', default=(299,299,3), type=tuple)
parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
parser.add_argument('--split_round', default='1', type=str)
parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='../hmdb51_org_images/', type=str)
parser.add_argument('--init_weights_path', default=None, type=str)
parser.add_argument('--logdir', default='../temp/', type=str)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--train_steps', default=2000, type=int)
parser.add_argument('--val_steps', default=500, type=int)
args, _ = parser.parse_known_args()

'''retrain inception_v3 with hmdb51'''
import data_utils
''''''

def build(classes, weights=None):
    base_model = InceptionV3(weights=weights, include_top=False) # [299,299,3]
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='fc_layer')(x)
    predictions = Dense(len(classes), activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions, name='inception_v3')

    # # TODO: call base_model
    # inputs = Input(shape=args.input_shape)
    # x = inputs
    # x = base_model(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu', name='fc_layer')(x)
    # predictions = Dense(len(classes), activation='softmax', name='predictions')(x)
    # outputs = predictions
    # model = Model(inputs=inputs, outputs=outputs, name='inception_v3')

    return base_model, model

def valid(args, classes, base_model, model):
    pass

def infer(args, classes, base_model, model):
    infer_generator = data_utils.iter_mini_batches(args, 'testing',classes, batch_size=args.batch_size, shuffle=True)

    count = [0,0]
    for batch, (image_batch, label_batch) in enumerate(infer_generator):
        if batch >=1000: break
        output_batch = model.predict(image_batch)
        print(output_batch.shape)
        ys_true = np.argmax(label_batch, axis=1)
        ys_pred = np.argmax(output_batch, axis=1)
        for i in range(label_batch.shape[0]):
            flag = 1 if ys_true[i] == ys_pred[i] else 0
            count[flag] += 1
            print('y_true: %d, y_pred: %d, flag: %d'%(ys_true[i], ys_pred[i], flag))
        print(batch, count)


def train(args, classes, base_model, model):
    save_model_path = os.path.join(args.logdir, "trained_models")
    if not os.path.exists(save_model_path): os.makedirs(save_model_path)

    # define callbacks
    # reduce_lr=ReduceLROnPlateau(monitor="val_loss", factor=args.lr_decay_mult"], patience=2)
    csv_logger = CSVLogger(filename=os.path.join(args.logdir, "history.log"))
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(save_model_path, "epoch_{epoch:04d}--trainloss_{loss:.5f}--valloss_{val_loss:.5f}.hdf5"),
        period=2)
    tensorboard = TensorBoard(log_dir=os.path.join(args.logdir, "tf_logs"), write_images=True)

    # train on generator 
    train_generator = data_utils.iter_mini_batches(args, 'training', classes, batch_size=args.batch_size)
    valid_generator = data_utils.iter_mini_batches(args, 'validation', classes, batch_size=args.batch_size)

    # step 01 
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
    model.compile(optimizer=RMSprop(lr=0.001), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    model.fit_generator(generator=train_generator,
                              steps_per_epoch=args.train_steps,
                              epochs=args.epoches,
                              validation_data=valid_generator,
                              validation_steps=args.val_steps,
                              max_q_size=100, # 100
                              workers=1, # num_gpus/num_cpus
                            #   pickle_safe=True,
                              callbacks=[csv_logger, checkpointer, tensorboard]
                              )

    # step 02
    for i, layer in enumerate(base_model.layers):
        layer.trainable=True
    model.compile(optimizer=SGD(lr=1e-4, momentum=9e-1), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    model.fit_generator(generator=train_generator,
                              steps_per_epoch=args.train_steps,
                              epochs=args.epoches,
                              validation_data=valid_generator,
                              validation_steps=args.val_steps,
                              max_q_size=100, # 100
                              workers=1, # num_gpus/num_cpus
                            #   pickle_safe=True,
                              callbacks=[csv_logger, checkpointer, tensorboard]
                              )

def main(args):
    classes = data_utils.get_classes(args.classes_path)
    base_model, model = build(classes, weights='imagenet')
    base_model.summary()
    model.summary()

    if args.init_weights_path is not None: 
        model.load_weights(args.init_weights_path, by_name=True)
    
    if args.mode == 'train': 
        train(args, classes, base_model, model)
    elif args.mode == 'infer':
        infer(args, classes, base_model, model)
    else: 
        raise ValueError('--mode [train | infer]')

if __name__ == '__main__': 
    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            main(args)
    else:
        main(args)