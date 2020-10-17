from tensorflow.keras.layers import Conv2D, Add, UpSampling2D, MaxPool2D


def build_fpn(bottom_up_maps):
    top_down_size = 256
    C2, C3, C4, C5 = bottom_up_maps
    ############
    # TOP DOWN #
    ############
    M5 = Conv2D(top_down_size, kernel_size=(1, 1))(C5)
    M4 = Add()([UpSampling2D(size=(2, 2))(M5), Conv2D(top_down_size, kernel_size=1)(C4)])
    M3 = Add()([UpSampling2D(size=(2, 2))(M4), Conv2D(top_down_size, kernel_size=1)(C3)])
    M2 = Add()([UpSampling2D(size=(2, 2))(M3), Conv2D(top_down_size, kernel_size=1)(C2)])
    ##############
    # FPN OUTPUT #
    ##############
    P6 = MaxPool2D(pool_size=(1, 1), strides=1)(M5)
    P5 = M5
    P4 = Conv2D(top_down_size, kernel_size=(3, 3), padding='same')(M4)
    P3 = Conv2D(top_down_size, kernel_size=(3, 3), padding='same')(M3)
    P2 = Conv2D(top_down_size, kernel_size=(3, 3), padding='same')(M2)

    return P2, P3, P4, P5, P6
