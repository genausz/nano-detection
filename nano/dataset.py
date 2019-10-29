
import skimage

class Dataset():
    def __init__(self, dataset_path):
        self.image_info = []
        self.class_info = [{'class_id':0, "class_name":"BG"}]
        self.prepare(dataset_path)

    def add_class(self, class_id, class_name):
        for info in self.class_info:
            if class_id == info['class_id']:
                return

        self.class_info.append({
            "class_id": class_id,
            "class_name": class_name,
        })

    def add_image(self, image_name, image_path, object_infos):
        image_info = {
            "image_name":image_name,
            "image_path":image_path,
            "object_infos":object_infos,    # shape (N, 5) [[class_id, center_x, center_y, width, height],...]
        }
        self.image_info.append(image_info)

    def prepare(self, dataset_path):
        pass
        

    @property
    def num_images(self):
        return len(self.image_info)

    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

def data_generator(dataset, config, batch_size=1, shuffle=True):
    b = 0  # batch item index

    