from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.losses import MeanSquaredError

def build_autoencoder(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    encoded = MaxPooling2D((2, 2), padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(encoded)
    d1 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2))(d1)
    d2 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2))(d2)
    d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d3)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=MeanSquaredError())

    return model
