import keras
from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPool2D, Dense
import keras.layers as KL
import keras.backend as K

def class_loss_graph(self, class_preds, class_gt):
    #TODO

def bbox_loss_graph(self, class_gt, bbox_preds, bbox_gt):
    pass


class NanoModel(object):

    def __init__(self, mode, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.keras_model = self.build()
    
    def build(self):
        num_anchors = len(config.ANCHORS)
        num_classes = config.NUM_CLASSES

        channels_class_branch = num_anchors *  num_classes
        channels_bbox_branch = num_anchors * 4

        input_image = Input(shape=self.config.IMAGE_SHAPE, name='input_image')
        class_gt = Input(shape=(num_anchors*num_classes, 16, 16), name='class_gt')
        bbox_gt = Input(shape=(num_anchors*4, 16, 16), name='bbox_gt')

        backbone = VGG16(include_top=False, input_tensor=input_image)
        
        backbone.summary()
        feature = backbone.get_layer('block5_pool')

        class_branch = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='conv_cls')(feature)
        class_preds = Conv2D(filters=channels_class_branch, kernel_size=(3,3), padding='same', activation='relu', name='class_preds')(class_branch)

        bbox_branch = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='conv_bbox')(feature)
        bbox_preds = Conv2D(filters=channels_bbox_branch, kernel_size=(3,3), padding='same', activation='relu', name='bbox_preds')(bbox_branch)

        class_loss = KL.Lambda(lambda x: class_loss_graph(*x), name="class_loss")([class_preds, class_gt])
        bbox_loss = KL.Lambda(lambda x: bbox_loss_graph(*x), name="bbox_loss")([class_gt, bbox_preds, bbox_gt])


        inputs = [input_image, class_gt, bbox_gt]
        outputs = [class_loss, bbox_loss]
        keras_model = keras.models.Model(inputs=inputs, outputs=outputs, name='nano_object_detection')
        
        return keras_model
    
    def compile(self):
        
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "class_loss",  "bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = tf.reduce_mean(layer.output, keepdims=True)
            self.keras_model.add_loss(loss)


