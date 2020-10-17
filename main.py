from TF_model import backbone
from TF_model import fpn
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

activation = swish
img_input = Input(shape=(224, 224, 3))

def create_model():

    backbone_maps = backbone.build_resnet_101_loop(img_input=img_input, activation=activation)
    fpn_maps = fpn.build_fpn(backbone_maps)
    model = Model(inputs=img_input, outputs=fpn_maps)
    model.summary()

create_model()