import os
import sys
import math
import random
import itertools
import argparse

import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.losses import mean_squared_error, mean_squared_logarithmic_error
from keras.metrics import mean_squared_error, mean_squared_logarithmic_error
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

import deconvnet 
# import utils

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--base_model', default='vgg16', type=str)
parser.add_argument('--input_shape', default=(224,224,3), type=tuple)
# parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
# parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
# parser.add_argument('--split_round', default='1', type=str)
# parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='./images/', type=str)
parser.add_argument('--init_weights_path', default=None, type=str)
parser.add_argument('--logdir', default='../temp/', type=str)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--train_steps', default=2000, type=int)
parser.add_argument('--val_steps', default=500, type=int)
args, _ = parser.parse_known_args()

# no GPU supplied
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_image_lists(args):
    image_list = os.listdir(args.image_dir)
    image_list = [os.path.join(args.image_dir, image) for image in image_list]
    image_dict = dict()

    random.seed(0)
    random.shuffle(image_list)
    idx1 = int(len(image_list) * 0.8)
    idx2 = int(len(image_list) * 0.9)
    image_dict['train'] = image_list[:idx1]
    image_dict['valid'] = image_list[idx1:idx2]
    image_dict['infer'] = image_list[idx2:]
    return image_dict

def iter_mini_batches(args, image_list, batch_size=4, shuffle=True):
    while True:
        if shuffle:
            random.seed()
            random.shuffle(image_list)

        num_batchs = int(math.ceil(len(image_list) / float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(image_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx
            image_batch = np.zeros(tuple([batch_size] + list(args.input_shape)))
            # label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image_batch[idx] = cv2.resize(cv2.imread(image_list[sidx + idx]), args.input_shape[:2])
                # label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield image_batch, image_batch

def build(args):
    base_model, model = deconvnet.vgg16_convnet(weights=None)
    model = deconvnet.vgg16_deconvnet(model)
    for i, layer in enumerate(model.layers): 
        print(i, layer.name)
    return base_model, model

def main(args):
    base_model, model = build(args)

    if args.init_weights_path is not None:
        model.load_weights(args.init_weights_path, by_name=True)

    image_dict = get_image_lists(args)
    if args.mode == 'train':
        train(args, image_dict, base_model, model)
    elif args.mode == 'infer':
        pass
    else:
        raise ValueError('--mode [train | infer]')

def train(args, image_dict, base_model, model):
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
    train_generator = iter_mini_batches(args, image_dict['train'], batch_size=args.batch_size)
    valid_generator = iter_mini_batches(args, image_dict['valid'], batch_size=args.batch_size)

    # step 01
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
    optimizer = SGD(lr=1e-5, momentum=9e-1, decay=1e-6)
    model.compile(optimizer=optimizer, loss=mean_squared_logarithmic_error, metrics=[mean_squared_logarithmic_error])
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
    optimizer = SGD(lr=1e-5, momentum=9e-1, decay=1e-6)
    model.compile(optimizer=optimizer, loss=mean_squared_logarithmic_error, metrics=[mean_squared_logarithmic_error])
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

if __name__ == '__main__': 
    main(args)