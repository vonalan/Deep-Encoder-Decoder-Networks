import os
import sys
import itertools
import argparse

import numpy as np
import tensorflow as tf
from keras import backend as keras
from keras.layers import Input, Dense, Lambda, GlobalAveragePooling2D
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


'''retrain inception_v3_attention with hmdb51'''
import attention as attention
import retrain_inception_v3 as inception
import data_utils
''''''


def build(classes):
    # inception_v3 retrained with hmdb51
    base_model, model = inception.build(classes, weights=None)
    topless_model = Model(model.input, model.layers[-2].output)

    # unified model
    # TODO: but can't be trained???
    inputs = Input(batch_shape=(tuple([1, None] + list(args.input_shape))))
    x = inputs

    x = Lambda(lambda x: keras.squeeze(x, axis=0))(x)
    x = topless_model(x)

    x = attention.SingleAttentionBlock(1)(x)
    x = attention.CascadedAttentionBlock(1024)(x)

    x = Dense(len(classes), activation='softmax', name='predictions')(x)
    outputs = x

    model = Model(inputs, outputs)

    return base_model, model


def build_old(classes):
    # inception_v3 retrained with hmdb51
    base_model, model = inception.build(classes, weights=None)
    topless_model = Model(model.input, model.layers[-2].output)

    # TODO: how to build a unified model???
    # add attention layer
    inputs = Input(shape=args.input_shape)
    x = inputs
    x = topless_model(x)
    x = attention.SingleAttentionBlock(1)(x)
    x = attention.CascadedAttentionBlock(1024)(x)
    x = Dense(len(classes), activation='softmax', name='predictions')(x)
    outputs = x
    model = Model(inputs, outputs)

    return topless_model, model


def valid(args, classes, base_model, model):
    pass

def train_test_keras(args, classes, base_model, model):
    base_model.trainable = False
    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-2), loss=categorical_crossentropy, metrics=[categorical_accuracy])

    import data_utils_mini as data_utils
    image_dict = data_utils.get_image_lists('../images')
    image_generator = data_utils.iter_mini_batches_for_dcgan(args.input_shape, image_dict['train'], batch_size=args.batch_size)
    for _, image_batch in image_generator:
        # np.random.seed(20180110)
        noise_batch = np.random.random((image_batch.shape))
        noise_batch = (noise_batch * 127.5 + 127.5).astype(np.uint8)
        noise_batch = np.expand_dims(noise_batch, axis=0)
        label_batch = np.zeros((noise_batch.shape[0],51))
        label_batch[:,4] = 1
        print(np.argmax(model.predict(noise_batch), axis=1))
        loss, acc = model.train_on_batch(noise_batch, label_batch)
        print(np.argmax(model.predict(noise_batch), axis=1))
        print('loss: %f, acc: %f'%(loss, acc))

        image_batch = np.expand_dims(image_batch, axis=0)
        label_batch = np.zeros((image_batch.shape[0],51))
        label_batch[:,2] = 1
        print(np.argmax(model.predict(image_batch), axis=1))
        loss, acc = model.train_on_batch(image_batch, label_batch)
        print(np.argmax(model.predict(image_batch), axis=1))
        print('loss: %f, acc: %f' % (loss, acc))

        print('\n')

def infer_test_keras(args, classes, base_model, model):
    import data_utils_mini as data_utils
    image_dict = data_utils.get_image_lists('../images')
    image_generator = data_utils.iter_mini_batches_for_dcgan(args.input_shape, image_dict['train'],
                                                             batch_size=args.batch_size)
    for _, image_batch in image_generator:
        image_batch = np.expand_dims(image_batch, axis=0)
        label_batch = model.predict(image_batch)
        print(label_batch, np.argmax(label_batch, axis=1))

        print('\n')


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
    train_generator = data_utils.iter_mini_batches(args, 'training', len(classes), batch_size=args.batch_size)
    valid_generator = data_utils.iter_mini_batches(args, 'validation', len(classes), batch_size=args.batch_size)

    # step 01
    base_model.trainable = False
    model.summary()
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
    base_model.trainable = True
    model.summary()
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
    base_model, model = build(classes)
    model.summary()

    if args.init_weights_path is not None:
        model.load_weights(args.init_weights_path, by_name=True)

    infer_test_keras(args, classes, base_model, model)

    # if args.mode == 'train':
    #     train(args, classes, base_model, model)
    # elif args.mode == 'infer':
    #     pass
    # else:
    #     raise ValueError('--mode [train | infer]')

if __name__ == '__main__': 
    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            main(args)
    else:
        main(args)