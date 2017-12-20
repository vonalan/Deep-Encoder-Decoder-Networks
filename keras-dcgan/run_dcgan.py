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
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, binary_crossentropy
from keras.metrics import mean_squared_error, mean_squared_logarithmic_error, binary_crossentropy
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

# import deconvnet as deconvnet
import vgg16_dcgan as dcgan
# import utils

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--device', default='gpu', type=str)
parser.add_argument('--base_model', default='vgg16', type=str)
parser.add_argument('--input_image_shape', default=(224,224,3), type=tuple)
parser.add_argument('--input_noise_shape', default=(1024,), type=tuple)
# parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
# parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
# parser.add_argument('--split_round', default='1', type=str)
# parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='../images/', type=str)
parser.add_argument('--output_dir', default='../outputs/', type=str)
parser.add_argument('--logdir', default='../temp/', type=str)
parser.add_argument('--init_weights_path', default=None, type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--train_steps', default=2000, type=int)
parser.add_argument('--val_steps', default=500, type=int)
args, _ = parser.parse_known_args()

# no GPU supplied
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
            image_batch = np.zeros(tuple([batch_size] + list(args.input_image_shape)))
            # label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image_batch[idx] = cv2.resize(cv2.imread(image_list[sidx + idx]), args.input_image_shape[:2])
                # label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield batch, image_batch

def iter_mini_batches_for_deconvnet(args, image_list, batch_size=4, shuffle=True):
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
            yield batch, image_batch, image_batch

def build(args):
    generator = dcgan.generator_model(Input(shape=args.input_noise_shape))
    discriminator = dcgan.discriminator_model(Input(shape=args.input_image_shape))
    model = dcgan.generator_containing_discriminator(generator, discriminator)
    return generator, discriminator, model

def main(args):
    generator, discriminator, mix_model = build(args)

    # for i, model in enumerate(mix_model.layers):
    #     for j, layer in enumerate(model.layers):
    #         print(i, model.name, j, layer.name, layer.trainable)

    if args.init_weights_path is not None:
        mix_model.load_weights(args.init_weights_path, by_name=True)

    image_dict = get_image_lists(args)
    if args.mode == 'train':
        train(args, image_dict, generator, discriminator, mix_model)
    elif args.mode == 'infer':
        generate(args, image_dict, generator, discriminator, mix_model)
    else:
        raise ValueError('--mode [train | infer]')

def generate(args, image_dict, generator, discriminator, mix_model):
    for i, image_path in enumerate(image_dict['infer']):
        if i >= 10: break
        raw_rgbs = cv2.resize(cv2.imread(image_path), args.input_shape[:2])
        raw_rgbs = np.expand_dims(raw_rgbs, axis=0)
        pred_rgbs = model.predict(raw_rgbs)
        composed = np.concatenate((raw_rgbs[0], pred_rgbs[0]), axis=1)
        composed = composed.astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, '%d.jpg'%(i+1)), composed)

def train(args, image_dict, generator, discriminator, mix_model):
    save_model_path = os.path.join(args.logdir, "trained_models")
    if not os.path.exists(save_model_path): os.makedirs(save_model_path)

    # define callbacks
    # reduce_lr=ReduceLROnPlateau(monitor="val_loss", factor=args.lr_decay_mult"], patience=2)
    csv_logger = CSVLogger(filename=os.path.join(args.logdir, "history.log"))
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(save_model_path, "epoch_{epoch:04d}--trainloss_{loss:.5f}--valloss_{val_loss:.5f}.hdf5"),
        period=1)
    tensorboard = TensorBoard(log_dir=os.path.join(args.logdir, "tf_logs"), write_images=True)

    # train on generator
    train_generator = iter_mini_batches(args, image_dict['train'], batch_size=args.batch_size)
    valid_generator = iter_mini_batches(args, image_dict['valid'], batch_size=args.batch_size)

    for epoch in range(args.epoches):
        for total_batch_index, (cur_batch_index, raw_image_batch) in enumerate(train_generator):
            # TODO: sigmoid
            raw_noise_batch = np.random.uniform(0.0, 1.0, size=(args.batch_size, 1024))
            gen_image_batch = generator.predict(raw_noise_batch)

            if total_batch_index % 10 == 0:
                # TODO: SAVE IMAGES
                for i in range(gen_image_batch.shape[0]):
                    if i > 0: break
                    gen_image = gen_image_batch[i,...]
                    print(gen_image.shape, gen_image.sum())

                    gen_image = (gen_image - gen_image.min()) / (gen_image.max() - gen_image.min())
                    gen_image = np.round(gen_image * 255.0).astype(np.uint8)
                    cv2.imwrite('../outputs/batch_%d.jpg'%(total_batch_index), gen_image)

            # train discriminator model
            discriminator.trainable = True
            generator.trainable= False
            ''''''
            for i, model in enumerate(mix_model.layers):
                for j, layer in enumerate(model.layers):
                    print(i, model.name, j, layer.name, layer.trainable)
            for j, layer in enumerate(generator.layers):
                print(j, layer.name, layer.trainable)
            for j, layer in enumerate(discriminator.layers):
                    print(j, layer.name, layer.trainable)
            ''''''
            optimizer = SGD(lr=5e-4, momentum=9e-1, nesterov=True)
            discriminator.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_crossentropy])

            print((discriminator.layers[-5].get_weights())[0][0][0][0][:5])
            input_batch = np.concatenate((raw_image_batch, gen_image_batch), axis=0)
            label_batch = raw_image_batch.shape[0] * [1] + gen_image_batch.shape[0] * [0] # !!! tf.one_like() | tf.zero_like()
            d_loss, _ = discriminator.train_on_batch(input_batch, label_batch)
            print((discriminator.layers[-5].get_weights())[0][0][0][0][:5])

            # train generator model, error propagates back through discriminator
            discriminator.trainable = True
            generator.trainable = True
            ''''''
            for i, model in enumerate(mix_model.layers):
                for j, layer in enumerate(model.layers):
                    print(i, model.name, j, layer.name, layer.trainable)
            ''''''
            optimizer = SGD(lr=5e-4, momentum=9e-1, nesterov=True)
            mix_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_crossentropy])

            g_loss = 0.0
            for i in range(10):
                # TODO: sigmoid
                input_batch = np.random.uniform(0, 1, size=(args.batch_size, 1024))
                label_batch = input_batch.shape[0] * [1] # !!! tf.one_like() | tf.zero_like()
                loss, _ = mix_model.train_on_batch(input_batch, label_batch)
                # print((discriminator.layers[-5].get_weights())[0][0][0][0][:5])
                print((generator.layers[-5].get_weights())[0][0][0][0][:5])
                # pred_label = mix_model.predict(input_batch)
                # print(pred_label.tolist(), loss)
                g_loss += loss
            g_loss = g_loss / 10
            print('epoch: %d, batch: %d, d_loss: %.8f, g_loss_: %.8f' % (epoch, total_batch_index, d_loss, g_loss))

        if epoch % 10 == 0:
            # TODO: SAVE WEIGHTS
            pass

if __name__ == '__main__': 
    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            main(args)
    else:
        main(args)