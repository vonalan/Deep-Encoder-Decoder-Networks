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
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--base_model', default='inception_v3', type=str)
parser.add_argument('--input_shape', default=(299, 299, 3), type=tuple)
parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
parser.add_argument('--split_round', default='1', type=str)
parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='../hmdb51_org_images/', type=str)
parser.add_argument('--init_weights_path', default=None, type=str)
parser.add_argument('--logdir', default='../temp_for_attention/', type=str)
# parser.add_argument('--batch_size', default=10, type=int) # inception
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--train_steps', default=2000, type=int)
parser.add_argument('--val_steps', default=500, type=int)

parser.add_argument('--batch_size', default=1, type=int)
args, _ = parser.parse_known_args()

'''retrain inception_v3_attention with hmdb51'''
import attention_dev as attention
import retrain_inception_v3 as inception
import data_utils

''''''


def build_dev(classes):
    # inception_v3 retrained with hmdb51
    base_model, model = inception.build(classes, weights=None)
    topless_model = Model(model.input, model.layers[-2].output)

    # unified model
    # TODO: but can't be trained???
    inputs = Input(batch_shape=(tuple([1, None] + list(args.input_shape))))
    x = inputs

    x = Lambda(lambda x: keras.squeeze(x, axis=0))(x)
    x = topless_model(x)

    x = Lambda(lambda x: keras.expand_dims(x, axis=0))(x)
    x = attention.SingleAttentionBlock(1)(x)
    x = attention.CascadedAttentionBlock(1024)(x)

    x = Lambda(lambda x: keras.squeeze(x, axis=1))(x)
    x = Dense(len(classes), activation='softmax', name='predictions')(x)
    outputs = x

    model = Model(inputs, outputs)

    return base_model, model


def build(classes):
    # inception_v3 retrained with hmdb51
    base_model, model = inception.build(classes, weights=None)
    topless_model = Model(model.input, model.layers[-2].output)

    # TODO: how to build a unified model???
    # # add attention layer
    # inputs = Input(shape=args.input_shape)
    # x = inputs
    # x = topless_model(x)
    # x = attention.SingleAttentionBlock(1)(x)
    # x = attention.CascadedAttentionBlock(1024)(x)
    # x = Dense(len(classes), activation='softmax', name='predictions')(x)
    # outputs = x
    # model = Model(inputs, outputs)

    # attention model
    # import attention_dev as attention
    inputs = Input(batch_shape=(1, None, 1024))
    x = inputs
    x = attention.SingleAttentionBlock(1)(x)
    x = attention.CascadedAttentionBlock(1024)(x)
    x = Lambda(lambda x: keras.squeeze(x, axis=1))(x)
    x = Dense(len(classes), activation='softmax', name='predictions')(x)
    outputs = x

    model = Model(inputs, outputs)

    return topless_model, model


def valid(args, classes, base_model, model):
    pass


def train_test_keras(args, classes, base_model, model):
    # base_model.trainable = False
    model.compile(optimizer=RMSprop(lr=0.001), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    model.summary()

    import data_utils_mini as data_utils
    image_dict = data_utils.get_image_lists('../images')
    image_generator = data_utils.iter_mini_batches_for_dcgan(args.input_shape, image_dict['train'],
                                                             batch_size=args.batch_size)
    for _, image_batch in image_generator:
        num_samples = np.random.randint(15, 30, (1,))[0]
        input_batch = np.random.random((1, num_samples, 1024))
        label_batch = np.zeros((1, 51))
        idx = np.random.randint(0, 51, (1,))[0]
        label_batch[:,idx] = 1
        print(np.argmax(model.predict(input_batch), axis=1))
        loss, acc = model.train_on_batch(input_batch, label_batch)
        print(np.argmax(model.predict(input_batch), axis=1))
        print('loss: %f, acc: %f' % (loss, acc))

        input_batch = base_model.predict(image_batch)
        input_batch = np.expand_dims(input_batch, axis=0)
        label_batch = np.zeros((1, 51))
        label_batch[:, 2] = 1
        print(np.argmax(model.predict(input_batch), axis=1))
        loss, acc = model.train_on_batch(input_batch, label_batch)
        print(np.argmax(model.predict(input_batch), axis=1))
        print('loss: %f, acc: %f' % (loss, acc))

        print('\n')


def infer(args, classes, base_model, model):
    infer_generator = data_utils.iter_mini_batches_for_attention(args, base_model, 'training', classes,
                                                                 batch_size=args.batch_size)
    for feat_batch, label_batch in infer_generator:
        print(feat_batch.min(), feat_batch.max(), feat_batch.mean())
        print(np.argmax(label_batch, axis=1), np.argmax(model.predict(feat_batch), axis=1))
        print('\n')


def train(args, classes, base_model, model):
    save_model_path = os.path.join(args.logdir, "trained_models")
    if not os.path.exists(save_model_path): os.makedirs(save_model_path)

    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-3), loss=categorical_crossentropy, metrics=[categorical_accuracy])

    for epoch in itertools.count():
        train_generator = data_utils.iter_mini_batches_for_attention(args, base_model, 'training', classes,
                                                                     batch_size=args.batch_size, infinite=False)
        valid_generator = data_utils.iter_mini_batches_for_attention(args, base_model, 'validation', classes,
                                                                     batch_size=args.batch_size, infinite=False)

        # # TODO: tensorboard
        # # TODO: moving average
        train_loss = 0.0
        train_acc = 0.0
        train_count = 0
        for batch, (feat_batch, label_batch) in enumerate(train_generator):
            loss, acc = model.train_on_batch(feat_batch, label_batch)
            train_acc = train_acc * train_count + acc * feat_batch.shape[0]
            train_loss = train_loss * train_count + loss * feat_batch.shape[0]
            train_count = train_count + feat_batch.shape[0]
            train_loss = train_loss / train_count
            train_acc = train_acc / train_count
            print('epoch_%4d--batch_%4d--trainloss_%.5f--trainacc_%.5f_trainloss_%.5f--trainacc_%.5f'%(epoch+1, batch+1, loss, acc, train_loss, train_acc))

        valid_loss = 0.0
        valid_acc = 0.0
        valid_count = 0
        for batch, (feat_batch, label_batch) in enumerate(valid_generator):
            loss, acc = model.test_on_batch(feat_batch, label_batch)
            valid_acc = valid_acc * valid_count + acc * feat_batch.shape[0]
            valid_loss = valid_loss * valid_loss + loss * feat_batch.shape[0]
            valid_count = valid_count + feat_batch.shape[0]
            valid_loss = valid_loss / valid_count
            valid_acc = valid_acc / valid_count
            print('epoch_%4d--batch_%4d--validloss_%.5f--validacc_%.5f_validloss_%.5f--validacc_%.5f' % (epoch + 1, batch + 1, loss, acc, valid_loss, valid_acc))

        checkpointer = os.path.join(save_model_path, "epoch_%4d--trainloss_%.5f--valloss_%.5f.hdf5"%(epoch+1, train_loss, valid_loss))
        model.save_weights(checkpointer)
        print("epoch_%4d--trainloss_%.5f--trainacc_%.5f_valloss_%.5f_validacc_%.5f" % (epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
        print('\n')


def train_v1(args, classes, base_model, model):
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
    train_generator = data_utils.iter_mini_batches_for_attention(args, base_model, 'training', classes,
                                                                 batch_size=args.batch_size)
    valid_generator = data_utils.iter_mini_batches_for_attention(args, base_model, 'validation', classes,
                                                                 batch_size=args.batch_size)

    # step 01
    # base_model.trainable = False
    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-3), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=args.train_steps,
                        epochs=args.epoches,
                        validation_data=valid_generator,
                        validation_steps=args.val_steps,
                        max_q_size=100,  # 100
                        workers=1,  # num_gpus/num_cpus
                        # pickle_safe=True,
                        callbacks=[csv_logger, checkpointer, tensorboard]
                        )

    # step 02
    # base_model.trainable = True
    model.summary()
    model.compile(optimizer=SGD(lr=1e-4, momentum=9e-1), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=args.train_steps,
                        epochs=args.epoches,
                        validation_data=valid_generator,
                        validation_steps=args.val_steps,
                        max_q_size=100,  # 100
                        workers=1,  # num_gpus/num_cpus
                        # pickle_safe=True,
                        callbacks=[csv_logger, checkpointer, tensorboard]
                        )


def main(args):
    classes = data_utils.get_classes(args.classes_path)
    base_model, model = build(classes)
    model.summary()

    if args.init_weights_path is not None:
        model.load_weights(args.init_weights_path, by_name=True)

    # train_test_keras(args, classes, base_model, model)

    if args.mode == 'train':
        train(args, classes, base_model, model)
    elif args.mode == 'infer':
        pass
    else:
        raise ValueError('--mode [train | infer]')


if __name__ == '__main__':
    assert args.device == 'cpu'

    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            main(args)
    else:
        main(args)
