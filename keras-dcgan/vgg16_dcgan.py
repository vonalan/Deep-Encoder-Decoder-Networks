import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Reshape, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
# from keras.layers.normalization import BatchNormalization as BN
# from keras.layers.noise import GaussianNoise as GN
from keras.optimizers import SGD

# TODO: KERAS_DCGAN_ORIGIN

def create_generator_model(input_noise_shape, output_image_shape):
    # replace unpool layer with stride convolutional layer

    # Input
    # inputs = Input(shape=(1,))
    inputs = Input(shape=input_noise_shape)
    x = inputs

    # FC
    # TODO: relu | tanh | sigmoid
    x = Dense((64 * 7 * 7), activation='relu')(x)
    x = Reshape((7, 7, 64))(x)  # not (4096,1,1)

    # TODO: where is fc_7?
    # FCN
    x = Conv2DTranspose(64, (1, 1), strides=(1, 1), activation='relu', padding='same', name='fc_7')(x)
    x = Conv2DTranspose(64, (7, 7), strides=(1, 1), activation='relu', padding='same', name='fc_6')(x)

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
    # x = Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same', name='deconv1_1')(x)
    # TODO: relu | tanh | sigmoid
    x = Conv2D(3, (3,3), strides=(1, 1), activation='tanh', padding='same', name='deconv1_1')(x)

    # XBlock 0 | Output
    outputs = x

    # Model
    model = Model(inputs=[inputs], outputs=[outputs], name='generator')
    return model
    # return outputs

def create_discriminator_model(input_image_shape, output_noise_shape):
    # replace pool layer with convolutional layer

    # Block 0 | Input
    assert input_image_shape == (224, 224, 3)
    inputs = Input(shape=input_image_shape)
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
    x = Conv2D(64, (7,7), strides=(1,1), activation='relu', padding='same', name='fc_6')(x)
    x = Conv2D(64, (1,1), strides=(1,1), activation='relu', padding='same', name='fc_7')(x)

    # FC
    # TODO: relu | tanh | sigmoid
    x = Flatten()(x)
    x = Dense(output_noise_shape[0], activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    # outputs
    outputs = x
    model = Model(inputs=[inputs], outputs=[outputs], name='discriminator')
    return model
    # return outputs

def deconvnet(input_image_shape):
    # replace pool layer with convolutional layer
    # replace unpool layer with stride convolutional layer

    # Block 0 | Input
    assert input_image_shape == (224,224,3)
    inputs = Input(shape=input_image_shape)
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
    # TODO: is fc_8 | d_fc_8 necessary?
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
    # x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='deconv1_1')(x)
    # TODO: relu | tanh | sigmoid
    x = Conv2D(3, (3,3), strides=(2,2), activation='relu', padding='same', name='deconv1_1')(x)

    # XBlock 0 | Output
    outputs = x
    
    # Model 
    model = Model(inputs=[inputs], outputs=[outputs])
    return None, model

def test_generator(model):
    import numpy as np
    import cv2

    zs = np.random.uniform(-1.0, 1.0, size=(10, 1024))
    ys = model.predict(zs)
    ys = (ys * (255.0 / 2.0) + (255.0 / 2.0)).astype(np.uint8)
    print(ys.shape, ys.min(), ys.max(), ys.mean())

    for i in range(ys.shape[0]):
        cv2.imwrite('../outputs/test_generator_%d.jpg'%(i), ys[i,...])

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

if __name__ == "__main__":
    input_image_shape = (224,224,3)
    input_noise_shape = (1024,)

    generator = create_generator_model((1024,),(224,224,3))
    for i, layer in enumerate(generator.layers):
        print(i, '---', layer.name, '---', layer.input_shape, '---', layer.output_shape)
    # test_generator(generator)

    discriminator = create_discriminator_model((224,224,3), (1024,))
    for i, layer in enumerate(discriminator.layers):
        print(i, '---', layer.name, '---', layer.input_shape, '---', layer.output_shape)
    # test_discriminator(discriminator)