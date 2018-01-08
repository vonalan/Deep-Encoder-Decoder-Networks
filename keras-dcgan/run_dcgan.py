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
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, binary_crossentropy
from keras.metrics import mean_squared_error, mean_squared_logarithmic_error, binary_crossentropy, binary_accuracy
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model

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
                image = cv2.resize(cv2.imread(image_list[sidx + idx]), args.input_image_shape[:2])
                # TODO: tanh | sigmoid | relu
                image_batch[idx] = (image - (255.0/2.0)) / (255.0/2.0)
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

# def build(args):
#     generator = dcgan.generator_model(Input(shape=args.input_noise_shape))
#     discriminator = dcgan.discriminator_model(Input(shape=args.input_image_shape))
#     model = dcgan.generator_containing_discriminator(generator, discriminator)
#     return generator, discriminator, model

def build_models(args):
    input_noise = Input(shape=args.input_noise_shape)
    input_image = Input(shape=args.input_image_shape)

    # gen_output = dcgan.generator_model(input_noise)
    # dis_output = dcgan.discriminator_model(input_image)
    # # mix_output = dcgan.discriminator_model(gen_output)
    #
    # generator = Model(input_noise, gen_output)
    # discriminator = Model(input_image, dis_output) # d_loss
    # # mix_model = Model(input_noise, mix_output) # g_loss

    generator = dcgan.create_generator_model(input_noise)
    discriminator = dcgan.create_discriminator_model(input_image)

    print(id(generator), id(discriminator))
    # discriminator.trainable = False
    gan_input = Input(shape=args.input_noise_shape)
    gen_output = generator(gan_input)
    gan_output = discriminator(gen_output)
    gan_model = Model(gan_input, gan_output)
    # discriminator.trainable = True
    print(id(gan_model), id(gan_model.layers[1]), id(gan_model.layers[2]))

    # # mix use
    # mix_model = Sequential()
    # mix_model.add(generator)
    # discriminator.trainable = False
    # mix_model.add(discriminator)

    # # ''''''
    # # # TODO: the difference between model.trainable and layer.trainable?
    # # generator.trainable = False
    # # discriminator.trainable = True
    # for i, layer in enumerate((gan_model.layers)[1].layers):
    #     # layer.trainable = False
    #     print(i, layer.name)
    # for i, layer in enumerate((gan_model.layers[2]).layers):
    #     # layer.trainable = False
    #     print(i, layer.name)
    # # ''''''

    generator.summary()
    discriminator.summary()
    gan_model.summary()
    # plot_model(gan_model, to_file='dcgan.png')

    return generator, discriminator, gan_model

def main(args):
    generator, discriminator, gan_model = build_models(args)

    # for i, model in enumerate(mix_model.layers):
    #     for j, layer in enumerate(model.layers):
    #         print(i, model.name, j, layer.name, layer.trainable)

    if args.init_weights_path is not None:
        gan_model.load_weights(args.init_weights_path, by_name=True)

    image_dict = get_image_lists(args)
    if args.mode == 'train':
        train(args, image_dict, generator, discriminator, gan_model)
    elif args.mode == 'infer':
        generate(args, image_dict, generator, discriminator, gan_model)
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

def train(args, image_dict, generator, discriminator, gan_model):
    save_model_path = os.path.join(args.logdir, "trained_models")
    if not os.path.exists(save_model_path): os.makedirs(save_model_path)

    # # define callbacks
    # # reduce_lr=ReduceLROnPlateau(monitor="val_loss", factor=args.lr_decay_mult"], patience=2)
    # csv_logger = CSVLogger(filename=os.path.join(args.logdir, "history.log"))
    # checkpointer = ModelCheckpoint(
    #     filepath=os.path.join(save_model_path, "epoch_{epoch:04d}--trainloss_{loss:.5f}--valloss_{val_loss:.5f}.hdf5"),
    #     period=1)
    # tensorboard = TensorBoard(log_dir=os.path.join(args.logdir, "tf_logs"), write_images=True)

    # train on generator
    train_generator = iter_mini_batches(args, image_dict['train'], batch_size=args.batch_size)
    valid_generator = iter_mini_batches(args, image_dict['valid'], batch_size=args.batch_size)

    # TODO: what model.compile() means ?
    g_optimizer = SGD(lr=1e-2, momentum=9e-1, decay=0.0, nesterov=False) # momentum = 0.0
    # generator.compile(optimizer=g_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    m_optimizer = SGD(lr=1e-2, momentum=9e-1, decay=0.0, nesterov=True) # lr = 5e-4
    # gan_model.compile(optimizer=m_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    # discriminator.trainable = True
    d_optimizer = SGD(lr=1e-2, momentum=9e-1, decay=0.0, nesterov=True) # lr = 5e-4
    # discriminator.compile(optimizer=d_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    for epoch in range(args.epoches):
        for total_batch_index, (cur_batch_index, raw_image_batch) in enumerate(train_generator):
            discriminator.trainable = True
            generator.trainable = False
            gan_model.compile(optimizer=m_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
            discriminator.compile(optimizer=d_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
            generator.compile(optimizer=g_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

            # generator.summary()
            # discriminator.summary()
            # gan_model.summary()

            # TODO: sigmoid
            # np.random.seed(0)
            raw_noise_batch = np.random.uniform(-1.0, 1.0, size=(args.batch_size, 1024))
            gen_image_batch = generator.predict(raw_noise_batch)

            if total_batch_index % 1 == 0:
                # TODO: SAVE IMAGES
                for i in range(gen_image_batch.shape[0]):
                    if i > 0: break
                    gen_image = gen_image_batch[i,...]
                    print(gen_image.shape, gen_image.min(), gen_image.max(), gen_image.sum())

                    # gen_image = (gen_image - gen_image.min()) / (gen_image.max() - gen_image.min())
                    # gen_image = np.round(gen_image * 255.0).astype(np.uint8)
                    # TODO: tanh | sigmoid | relu
                    gen_image = (gen_image * (255.0/2.0) + (255.0/2.0)).astype(np.uint8)
                    cv2.imwrite('../outputs/batch_%d.jpg'%(total_batch_index), gen_image)

            # # train discriminator model
            # discriminator.trainable = True
            # generator.trainable= False
            # for i, layer in enumerate(generator.layers):
            #     layer.trainable = False
            # for i, layer in enumerate(discriminator.layers):
            #     layer.trainable = True
            # for i, model in enumerate(mix_model.layers): # reference
            #     for j, layer in enumerate(model.layers):
            #         print(i, model.name, j, layer.name, layer.trainable)
            # optimizer = SGD(lr=5e-4, momentum=9e-1, nesterov=True)
            # discriminator.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_crossentropy])

            print('*' * 64)
            # print((discriminator.layers[-5].get_weights())[0][0][0][0][:5], ((discriminator.layers[-5].get_weights())[0][0][0][0][:5]).sum())
            # print((generator.layers[3].get_weights())[0][0][0][0][:5], ((generator.layers[3].get_weights())[0][0][0][0][:5]).sum())
            input_batch = np.concatenate((raw_image_batch, gen_image_batch), axis=0)
            label_batch = raw_image_batch.shape[0] * [1] + gen_image_batch.shape[0] * [0] # !!! tf.one_like() | tf.zero_like()
            d_loss, _ = discriminator.train_on_batch(input_batch, label_batch)
            # print((discriminator.layers[-5].get_weights())[0][0][0][0][:5], ((discriminator.layers[-5].get_weights())[0][0][0][0][:5]).sum())
            # print((generator.layers[3].get_weights())[0][0][0][0][:5], ((generator.layers[3].get_weights())[0][0][0][0][:5]).sum())

            # # train generator model, error propagates back through discriminator
            # discriminator.trainable = False
            # generator.trainable = True
            # for i, layer in enumerate(generator.layers):
            #     layer.trainable = True
            # for i, layer in enumerate(discriminator.layers):
            #     layer.trainable = False
            # for i, model in enumerate(mix_model.layers): # reference
            #     for j, layer in enumerate(model.layers):
            #         print(i, model.name, j, layer.name, layer.trainable)
            # optimizer = SGD(lr=5e-4, momentum=9e-1, nesterov=True)
            # mix_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_crossentropy])

            # TODO: WHY CAN'T GENERATOR BE TRAINED???
            discriminator.trainable = False
            generator.trainable = True
            gan_model.compile(optimizer=m_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
            discriminator.compile(optimizer=d_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
            generator.compile(optimizer=g_optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

            # generator.summary()
            # discriminator.summary()
            # gan_model.summary()

            g_loss = 0.0
            tmp = 1.0
            for i in range(2):
                # TODO: sigmoid
                np.random.seed(0)
                input_batch = np.random.uniform(-1.0, 1.0, size=(args.batch_size, 1024))
                print(input_batch.tolist()[0][:7])
                label_batch = input_batch.shape[0] * [1] # !!! tf.one_like() | tf.zero_like()
                # print((discriminator.layers[-5].get_weights())[0][0][0][0][:5], ((discriminator.layers[-5].get_weights())[0][0][0][0][:5]).sum())
                print((generator.layers[3].get_weights())[0][0][0][0][:5], ((generator.layers[3].get_weights())[0][0][0][0][:5]).sum())
                print((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5], ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum(), ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum()/tmp)
                tmp = ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum()/tmp
                loss, _ = gan_model.train_on_batch(input_batch, label_batch)
                # print((discriminator.layers[-5].get_weights())[0][0][0][0][:5], ((discriminator.layers[-5].get_weights())[0][0][0][0][:5]).sum())
                print((generator.layers[3].get_weights())[0][0][0][0][:5], ((generator.layers[3].get_weights())[0][0][0][0][:5]).sum())
                print((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5], ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum(), ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum()/tmp)
                tmp = ((gan_model.layers[1].layers[3].get_weights())[0][0][0][0][:5]).sum()/tmp
                # pred_label = mix_model.predict(input_batch)
                # print(pred_label.tolist(), loss)
                g_loss += loss
            g_loss = g_loss / 2
            # discriminator.trainable = True
            print('*' * 64)
            print('epoch: %d, batch: %d, d_loss: %.8f, g_loss_: %.8f\n' % (epoch, total_batch_index, d_loss, g_loss))

        # if epoch % 10 == 0:
        #     # TODO: SAVE WEIGHTS
        #     pass

if __name__ == '__main__': 
    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            main(args)
    else:
        main(args)
    # build_models(args)