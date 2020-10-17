from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, Add, ZeroPadding2D, Input
from tensorflow.keras.activations import swish
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.backend import image_data_format


def build_residual_block(input, filters, activation, bn_axis, stride_first=1, conv_shortcut=True):
    skip = input
    if conv_shortcut:
        skip = Conv2D(filters=4 * filters, kernel_size=1, strides=stride_first)(input)
        skip = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(skip)
    x = Conv2D(kernel_size=1, filters=filters, strides=stride_first, padding='same')(input)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=filters, strides=1, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * filters, strides=1, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    return x


def build_residual_stack(input, filters, num_blocks, activation, stride_first=1):
    bn_axis = 3 if image_data_format() == 'channels_last' else 1
    x = input
    x = build_residual_block(input=x, filters=filters, activation=activation, bn_axis=bn_axis,
                             stride_first=stride_first)
    for i in range(num_blocks - 1):
        x = build_residual_block(input=x, filters=filters, activation=activation, bn_axis=bn_axis, conv_shortcut=False)

    return x


def build_resnet_50():
    bn_axis = 3 if image_data_format() == 'channels_last' else 1

    activation = swish
    img_input = Input(shape=(224, 224, 3))
    ###########
    # STACK 1 #
    ###########
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(filters=64, kernel_size=7, strides=2, name='stack1_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='stack1_bn1')(x)
    x = Activation(activation, name='stack1_activation1')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    C1 = MaxPool2D(pool_size=3, strides=2, name='stack1_mpool1')(x)
    ##########
    # STACK 2 #
    ##########
    ##conv2_block1
    skip = C1
    skip = Conv2D(filters=4 * 64, kernel_size=1, strides=1, name='stack2_skip_conv1')(skip)
    skip = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='stack2_skip_bn1')(skip)
    x = Conv2D(kernel_size=1, filters=64, strides=1, padding='same', name='stack2_block1_conv1')(C1)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', name='stack2__block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 64, strides=1, padding='same', name='stack2_block1_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    ##conv2_block2
    skip = x
    x = Conv2D(kernel_size=1, filters=64, strides=1, padding='same', name='stack2_block2_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', name='stack2_block2_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 64, strides=1, padding='same', name='stack2_block2_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv2_block3
    skip = x
    x = Conv2D(kernel_size=1, filters=64, strides=1, padding='same', name='stack2_block3_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', name='stack2_block3_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 64, strides=1, padding='same', name='stack2_block3_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    C2 = Activation(activation)(x)
    ##########
    # STACK 3 #
    ##########
    # conv3_block1
    skip = C2
    skip = Conv2D(filters=4 * 128, kernel_size=1, strides=2, padding='same')(skip)
    skip = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(skip)
    x = Conv2D(kernel_size=1, filters=128, strides=2, padding='same', name='stack3_block1_conv1')(C2)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='same', name='stack3_block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 128, strides=1, padding='same', name='stack3_block1_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv3_block2
    skip = x
    x = Conv2D(kernel_size=1, filters=128, strides=1, padding='same', name='stack3_block2_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='same', name='stack3_block2_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 128, strides=1, padding='same', name='stack3_block2_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv3_block3
    skip = x
    x = Conv2D(kernel_size=1, filters=128, strides=1, padding='same', name='stack3_block3_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='same', name='stack3_block3_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 128, strides=1, padding='same', name='stack3_block3_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv3_block4
    skip = x
    x = Conv2D(kernel_size=1, filters=128, strides=1, padding='same', name='stack3_block4_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='same', name='stack3_block4_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 128, strides=1, padding='same', name='stack3_block4_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    C3 = Activation(activation)(x)
    ##########
    # STACK 4 #
    ##########
    # conv4_block1
    skip = C3
    skip = Conv2D(filters=4 * 256, kernel_size=1, strides=2, padding='same', name='stack4_skip_conv1')(skip)
    skip = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(skip)
    x = Conv2D(kernel_size=1, filters=256, strides=2, padding='same', name='stack4_block1_conv1')(C3)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block1_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv4_block2
    skip = x
    x = Conv2D(kernel_size=1, filters=256, strides=1, padding='same', name='stack4_block2_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block2_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block2_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv4_block3
    skip = x
    x = Conv2D(kernel_size=1, filters=256, strides=1, padding='same', name='stack4_block3_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block3_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block3_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv4_block4
    skip = x
    x = Conv2D(kernel_size=1, filters=256, strides=1, padding='same', name='stack4_block4_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block4_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block4_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv4_block5
    skip = x
    x = Conv2D(kernel_size=1, filters=256, strides=1, padding='same', name='stack4_block5_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block5_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block5_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # conv4_block6
    skip = x
    x = Conv2D(kernel_size=1, filters=256, strides=1, padding='same', name='stack4_block6_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=256, strides=1, padding='same', name='stack4_block6_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 256, strides=1, padding='same', name='stack4_block6_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    C4 = Activation(activation)(x)
    ##########
    # STACK 5 #
    ##########
    # 1
    skip = C4
    skip = Conv2D(filters=4 * 512, kernel_size=1, strides=2, padding='same', name='stack5_skip_conv1')(skip)
    skip = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(skip)
    x = Conv2D(kernel_size=1, filters=512, strides=2, padding='same', name='stack5_block1_conv1')(C4)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=512, strides=1, padding='same', name='stack5_block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 512, strides=1, padding='same', name='stack5_block1_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # 2
    skip = x
    x = Conv2D(kernel_size=1, filters=512, strides=1, padding='same', name='stack5_block2_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=512, strides=1, padding='same', name='stack5_block2_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 512, strides=1, padding='same', name='stack5_block2_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    x = Activation(activation)(x)
    # 3
    skip = x
    x = Conv2D(kernel_size=1, filters=512, strides=1, padding='same', name='stack5_block3_conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=3, filters=512, strides=1, padding='same', name='stack5_block3_conv2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation(activation)(x)
    x = Conv2D(kernel_size=1, filters=4 * 512, strides=1, padding='same', name='stack5_bloc3_conv3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Add()([skip, x])
    C5 = Activation(activation)(x)
    return [C2, C3, C4, C5]



def build_resnet_50_loop(img_input, activation):
    bn_axis = 3 if image_data_format() == 'channels_last' else 1
    C1 = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    C1 = Conv2D(filters=64, kernel_size=7, strides=2, name='stack1_conv1')(C1)
    C1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='stack1_bn1')(C1)
    C1 = Activation(activation, name='stack1_activation1')(C1)
    C1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(C1)
    C1 = MaxPool2D(pool_size=3, strides=2, name='stack1_mpool1')(C1)
    C2 = build_residual_stack(C1, filters=64, num_blocks=3, stride_first=1, activation=activation)
    C3 = build_residual_stack(C2, filters=128, num_blocks=4, stride_first=2, activation=activation)
    C4 = build_residual_stack(C3, filters=256, num_blocks=6, stride_first=2, activation=activation)
    C5 = build_residual_stack(C4, filters=512, num_blocks=3, stride_first=2, activation=activation)

    return [C2, C3, C4, C5]


def build_resnet_101_loop(img_input, activation):
    bn_axis = 3 if image_data_format() == 'channels_last' else 1
    C1 = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    C1 = Conv2D(filters=64, kernel_size=7, strides=2, name='stack1_conv1')(C1)
    C1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='stack1_bn1')(C1)
    C1 = Activation(activation, name='stack1_activation1')(C1)
    C1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(C1)
    C1 = MaxPool2D(pool_size=3, strides=2, name='stack1_mpool1')(C1)
    C2 = build_residual_stack(C1, filters=64, num_blocks=3, stride_first=1, activation=activation)
    C3 = build_residual_stack(C2, filters=128, num_blocks=4, stride_first=2, activation=activation)
    C4 = build_residual_stack(C3, filters=256, num_blocks=23, stride_first=2, activation=activation)
    C5 = build_residual_stack(C4, filters=512, num_blocks=3, stride_first=2, activation=activation)
    return [C2, C3, C4, C5]



def build_resnet_152_loop(img_input, activation):
    bn_axis = 3 if image_data_format() == 'channels_last' else 1

    C1 = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    C1 = Conv2D(filters=64, kernel_size=7, strides=2, name='stack1_conv1')(C1)
    C1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='stack1_bn1')(C1)
    C1 = Activation(activation, name='stack1_activation1')(C1)
    C1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(C1)
    C1 = MaxPool2D(pool_size=3, strides=2, name='stack1_mpool1')(C1)
    C2 = build_residual_stack(C1, filters=64, num_blocks=3, stride_first=1, activation=activation)
    C3 = build_residual_stack(C2, filters=128, num_blocks=8, stride_first=2, activation=activation)
    C4 = build_residual_stack(C3, filters=256, num_blocks=36, stride_first=2, activation=activation)
    C5 = build_residual_stack(C4, filters=512, num_blocks=3, stride_first=2, activation=activation)
    return [C2, C3, C4, C5]


