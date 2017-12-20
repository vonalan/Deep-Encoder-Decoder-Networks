import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Reshape, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
# from keras.layers.normalization import BatchNormalization as BN
# from keras.layers.noise import GaussianNoise as GN
from keras.optimizers import SGD

# TODO: KERAS_DCGAN_ORIGIN

def generator_model(inputs):
    # replace unpool layer with stride convolutional layer

    # Input
    # inputs = Input(shape=(1,))
    # inputs = Input(shape=(1024,))
    x = inputs

    # FC
    # TODO: relu | tanh | sigmoid
    # x = Dense(1024, activation='tanh')(x)
    x = Dense((64 * 7 * 7), activation='relu')(x)
    x = Reshape((7,7,64))(x) # not (64,7,7)

    # FCN
    # TODO: is fc_8 necessary?
    # x = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same', name='fc_6')(x)
    # x = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', padding='same', name='fc_7')(x)
    x = Conv2DTranspose(64, (7, 7), strides=(1, 1), activation='relu', padding='same', name='fc_8')(x)

    # XBlock 5
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='deconv5_3')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv5_2')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv5_1')(x)

    # XBlock 4
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='deconv4_3')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv4_2')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv4_1')(x)

    # XBlock 3
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='deconv3_3')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv3_2')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv3_1')(x)

    # XBlock 2
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='deconv2_2')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv2_1')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv2_1')(x)

    # XBlock 1
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='deconv1_2')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv1_1')(x)
    x = Conv2D(3, (3, 3), strides=(1, 1), activation='relu', padding='same', name='deconv1_1')(x)

    # TODO: relu | tanh | sigmoid
    # x = Conv2D(3, (3, 3), strides=(1, 1), activation='sigmoid', padding='same', name='deconv1_1')(x)

    # Output
    outputs = x

    # Model
    # model = Model(inputs=[inputs], outputs=[outputs], name='generator')
    # return model
    return outputs

def discriminator_model(inputs):
    # replace pool layer with convolutional layer

    # Input
    # inputs = Input(shape=(224,224,3))
    x = inputs

    # Block 1
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv1_2')(x) # pool

    # Block 2
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv2_1')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv2_2')(x)

    # Block 3
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv3_3')(x)

    # Block 4
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv4_3')(x)

    # Block 5
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv5_3')(x)

    # FCN
    # TODO: is fc_8 necessary?
    x = Conv2D(64, (7,7), strides=(1,1), activation='relu', padding='same', name='fc_6')(x)
    # x = Conv2D(64, (1,1), strides=(1,1), activation='relu', padding='same', name='fc_7')(x)
    # x = Conv2DTranspose(64, (7,7), strides=(1,1), activation='relu', padding='same', name='fc_8')(x)

    # FC
    # TODO: relu | tanh | sigmoid
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    # outputs
    outputs = x
    # model = Model(inputs=[inputs], outputs=[outputs], name='discriminator')
    # return model
    return outputs

def deconvnet(inputs):
    # replace pool layer with convolutional layer
    # replace unpool layer with stride convolutional layer

    # Input
    # inputs = Input(shape=(224,224,3))
    x = inputs

    # Block 1
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv1_2')(x) # pool

    # Block 2
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv2_1')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv1_1')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv2_2')(x)

    # Block 3
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv3_3')(x)

    # Block 4
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv4_3')(x)

    # Block 5
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv5_3')(x)

    # FCN
    # TODO: is fc_8 necessary?
    x = Conv2D(64, (7,7), strides=(1,1), activation='relu', padding='same', name='fc_6')(x)
    x = Conv2D(64, (1,1), strides=(1,1), activation='relu', padding='same', name='fc_7')(x)
    x = Conv2DTranspose(64, (7,7), strides=(1,1), activation='relu', padding='same', name='fc_8')(x)

    # XBlock 5
    x = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv5_3')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv5_2')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv5_1')(x)

    # XBlock 4
    x = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv4_3')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv4_2')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv4_1')(x)

    # XBlock 3 
    x = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv3_3')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv3_2')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv3_1')(x)

    # XBlock 2
    x = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv2_2')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv2_1')(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv2_1')(x)

    # XBlock 1
    x = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv1_2')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv1_1')(x)
    x = Conv2D(3, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv1_1')(x)

    # Output
    outputs = x
    
    # Model 
    model = Model(inputs=[inputs], outputs=[outputs])
    return None, model

def test_generator(model):
    import numpy as np
    import cv2

    zs = np.random.random((1024,))
    zs = np.expand_dims(zs, axis=0)

    ys = model.predict(zs)
    print(ys.shape)

    ys = (ys - ys.min()) / (ys.max() - ys.min())
    ys = (ys * 255.0).astype(np.uint8)

    ys = ys[0]
    cv2.imwrite('../outputs/hhh.jpg', ys)

def test_deconvnet(model):
    import os
    import numpy as np
    import cv2

    image_dir = '../images/'
    for i, image in enumerate(os.listdir(image_dir)):
        if i > 10: break

        raw_image = cv2.imread(os.path.join(image_dir, image))
        raw_image = cv2.resize(raw_image, (224, 224))
        raw_image = np.expand_dims(raw_image, axis=0)

        pred_image = model.predict(raw_image)
        pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min())
        pred_image = (pred_image * 255.0).astype(np.uint8)

        pred_image = pred_image[0]
        cv2.imwrite('../outputs/hhh_%d.jpg'%(i), pred_image)

def test_discriminator(model): 
    import os 
    import numpy as np 
    import cv2

    image_dir = '../images/'
    for i, image in enumerate(os.listdir(image_dir)): 
        if i > 10: break

        image = cv2.imread(os.path.join(image_dir, image))
        image = cv2.resize(image, (224,224))
        image = np.expand_dims(image, axis=0)

        print(model.predict(image))

def generator_containing_discriminator_dev(generator, discriminator):
    pass

def generator_containing_discriminator(generator, discriminator):
    # TODO: substitute Sequential() with Model()
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

if __name__ == "__main__":
    input_image_shape = (224,224,3)
    input_noise_shape = (1024,)

    generator = generator_model(Input(shape=input_noise_shape))
    # for i, layer in enumerate(generator.layers):
    #     print(i, '---', layer.name, '---', layer.output_shape)
    # test_generator(model)

    discriminator = discriminator_model(Input(shape=input_image_shape))
    # for i, layer in enumerate(discriminator.layers):
    #     print(i, '---', layer.name, '---', layer.output_shape)
    # test_discriminator(model)

    model = generator_containing_discriminator(generator, discriminator)
    # for i, model in enumerate(model.layers):
    #     for j, layer in enumerate(model.layers):
    #         print(i, model.name, j, layer.name)
