import keras
from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPool2D, Dense

class NanoModel(object):

    def __init__(self, mode, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.keras_model = self.build()
    
    def build(self):
        input_image = Input(shape=self.config.IMAGE_SHAPE, name='input_image')

        backbone = VGG16(include_top=False, input_tensor=input_image)
        
        backbone.summary()
        feature = backbone.get_layer('block5_pool')

        class_branch = Conv2D()

        

