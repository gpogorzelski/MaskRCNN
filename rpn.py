from tensorflow.keras.layers import Conv2D, Activation, Lambda
from tensorflow.keras.activations import sigmoid
from tensorflow import reshape, shape


def build_rpn(len_aspect_ratios, shared_conv_activation, feature_maps):
    anchor_stride = 1

    def rpn(feature_map):
        # shared convolution of the RPN
        rpn_shared_conv = Conv2D(256, kernel_size=(3, 3), padding='same', activation=shared_conv_activation,
                                 strides=anchor_stride)(feature_map)

        rpn_box_convolution = Conv2D(4 * len_aspect_ratios, kernel_size=(1, 1), padding='valid', activation='linear')(
            rpn_shared_conv)
        # reshape box conv to [batch, anchors_per_location, 4]
        rpn_box_convolution = Lambda(lambda box: reshape(box, [shape(box)[0], -1, 4]))(rpn_box_convolution)

        rpn_class_logits = Conv2D(len_aspect_ratios, kernel_size=(1, 1), padding='valid')(rpn_shared_conv)
        # reshape class logits to [batch, anchors_per_location*map_width*map_height]
        rpn_class_convolution = Lambda(lambda cls: reshape(cls, [shape(cls)[0], -1]))(rpn_class_logits)
        rpn_class_convolution = Activation(activation=sigmoid)(rpn_class_convolution)

        return [rpn_class_logits, rpn_box_convolution, rpn_class_convolution]

    rpn_output_dict = dict()
    map_level = 2
    for feature_map in feature_maps:
        rpn_output_dict['p{}'.format(map_level)] = rpn(feature_map)
        map_level += 1
    return rpn_output_dict