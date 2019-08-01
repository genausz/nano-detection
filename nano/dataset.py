class Dataset():
    def __init__(self, dataset_path):
        self.image_info = []
        self.class_info = [{'class_id':0, "class_name":"BG"}]

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

    def prepare():
        

    @property
    def num_images(self):
        return len(self.image_info)

def data_generator(dataset, config, batch_size=1, shuffle=True):
    b = 0  # batch item index
    