import os
import sys
import itertools
import argparse

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.losses import mean_squared_error, mean_squared_logarithmic_error
from keras.metrics import mean_squared_error, mean_squared_logarithmic_error
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.xception import Xception

import data_utils 

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--device', default='gpu', type=str)
parser.add_argument('--base_model', default='vgg16', type=str)
parser.add_argument('--input_shape', default=(224,224,3), type=tuple)
parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
parser.add_argument('--split_round', default='1', type=str)
parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='../hmdb51_org_images/', type=str)
parser.add_argument('--init_weights_path', default=None, type=str)
parser.add_argument('--logdir', default='../temp_for_deconvnet/', type=str)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--train_steps', default=2000, type=int)
parser.add_argument('--val_steps', default=500, type=int)
args, _ = parser.parse_known_args()

# import deconvnet
import dcgan

def build(classes):
    base_model, model = dcgan.deconvnet()
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    return base_model, model

def valid(args, classes, base_model, model):
    pass

def infer(args, classes, base_model, model):
    pass

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
    train_generator = data_utils.iter_mini_batches_for_deconvnet(args, 'training', len(classes), batch_size=args.batch_size)
    valid_generator = data_utils.iter_mini_batches_for_deconvnet(args, 'validation', len(classes), batch_size=args.batch_size)

    # # step 01
    # # for i, layer in enumerate(base_model.layers):
    # #     layer.trainable = False
    # model.compile(optimizer=SGD(lr=1e-4, momentum=9e-1, decay=1e-6), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # model.fit_generator(generator=train_generator,
    #                           steps_per_epoch=args.train_steps,
    #                           epochs=args.epoches,
    #                           validation_data=valid_generator,
    #                           validation_steps=args.val_steps,
    #                           max_q_size=100, # 100
    #                           workers=1, # num_gpus/num_cpus
    #                           # pickle_safe=True,
    #                           callbacks=[csv_logger, checkpointer, tensorboard]
    #                           )

    # step 02
    # for i, layer in enumerate(base_model.layers):
    #     layer.trainable=True
    model.compile(optimizer=SGD(lr=1e-5, momentum=9e-1, decay=1e-6), loss=mean_squared_logarithmic_error, metrics=[mean_squared_error])
    model.fit_generator(generator=train_generator,
                              steps_per_epoch=args.train_steps,
                              epochs=args.epoches,
                              validation_data=valid_generator,
                              validation_steps=args.val_steps,
                              max_q_size=100, # 100
                              workers=1, # num_gpus/num_cpus
                              # pickle_safe=True,
                              callbacks=[csv_logger, checkpointer, tensorboard]
                              )

def main(args):
    classes = data_utils.get_classes(args.classes_path)
    base_model, model = build(classes)
    
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