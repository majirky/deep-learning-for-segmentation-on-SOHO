
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Input,
    concatenate,
    Activation,
)


def conv2_block(inputs, filters, batch_norm=True):
    """   Convolution block used in encoder (left-side) of neural network.

    Args:
        inputs (numpy.array): input tensor

        filters (int): how many filters to use in convolution

        batch_norm (bool, optional): to determine usage of batch noramlization. Defaults to True.

    Returns:
        numpy.array: output tensor
    """
    tensor = inputs
    for block in range(2):
        tensor = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(tensor)
        if batch_norm:
            tensor = BatchNormalization()(tensor)
        tensor = Activation("relu")(tensor)
    return tensor


def deconv2_block(inputs, filters, batch_norm=True):
    """   Deconvolution block used in decoder (right-side) of neural network.

    Args:
        inputs (numpy.array): input tensor

        filters (int): how many filters to use in DEconvolution

        batch_norm (bool, optional): to determine usage of batch noramlization. Defaults to True.

    Returns:
        numpy.array: output tensor
    """
    tensor = inputs
    for block in range(2):
        tensor = Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding="same")(tensor)
        if batch_norm:
            tensor = BatchNormalization()(tensor)
        tensor = Activation("relu")(tensor)
    return tensor


def scss_net(input_shape, filters, layers=4, batch_norm=True, drop_prob=0.0):
    """using tensorflow functional API, customizes U-Net[1] architecture for segmentation of solar active regions.
    [1]: https://arxiv.org/abs/1505.04597

    Args:
        input_shape (numpy.array): input shape of image (height, width, depth)

        filters (int): number of filter used in first layer (doubles with each layer)

        layers (int, optional): number of layers. Defaults to 4.

        batch_norm (bool, optional): whether to use batch normalization. Defaults to True.

        drop_prob (float, optional): dropout probability. Defaults to 0.0.

    Returns:
        tf.keras.Model: model of SCSS-Net. Ready to use with eg. model.compile() and model.fit() and model.predict() or model.evaluate()
    """
    dropout = drop_prob != 0.0
    inputs = Input(shape=input_shape, name="input_tensor")
    tensor = inputs
    encoder_layers = []
    
    # Encoder
    for layer in range(layers):
        if dropout and layer >= 2:
            tensor = Dropout(drop_prob)(tensor)
        tensor = conv2_block(tensor, filters, batch_norm)
        encoder_layers.append(tensor)
        tensor = MaxPooling2D((2, 2))(tensor)
        filters = filters * 2  # double filters for each layer
    if dropout:
        tensor = Dropout(drop_prob)(tensor)
    tensor = conv2_block(tensor, filters, batch_norm)  # Last encoder layer

    # Decoder
    for ii, layer in enumerate(reversed(encoder_layers)):
        if dropout and ii < 2:
            tensor = Dropout(drop_prob)(tensor)
        filters = filters // 2  # decrease filters for each layer
        tensor = UpSampling2D((2, 2))(tensor)
        tensor = concatenate([tensor, layer])  # skip connection
        tensor = deconv2_block(tensor, filters, batch_norm)

    outputs = Conv2D(filters=1, kernel_size=(1, 1), padding="same")(tensor) 
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model
