import tensorflow as tf
import os
import albumentations as albu

class DatasetCreator:
    def __init__(self, preprocess,transforms: albu.Compose) -> None:
        self.preprocess = preprocess
        self.transforms = transforms

    def get_mask(self, image: tf.Tensor) -> tf.Tensor:
        return tf.strings.split(image, os.path.sep)[-1]

    def process_image_with_mask(self, file_path: tf.Tensor, maskspath: str) -> (tf.Tensor, tf.Tensor):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image)
        mask = maskspath + self.get_mask(file_path)
        mask = tf.io.read_file(mask)
        mask = tf.image.decode_image(mask)
        cond = tf.greater_equal(mask, 1)
        mask = tf.where(cond, 1, 0)
        return tf.cast(image, tf.uint8), tf.cast(mask, tf.uint8)

    def aug_fn(self, image: tf.Tensor, mask: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = image[:,:,:3]
        data = {"image": image, "mask": mask}
        aug_data = self.transforms(**data)
        image = aug_data["image"]
        mask = aug_data["mask"]
        return self.preprocess(tf.cast(image, tf.float32)), tf.cast(mask, tf.float32)

    def process_data(self, image: tf.Tensor, mask: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return tf.numpy_function(self.aug_fn, inp=(image, mask), Tout=(tf.float32, tf.float32))

    def __call__(self, imagepath: str, maskspath: str, shuffle_buffer_size: int = 100, batch_size: int = 4):
        dataset = tf.data.Dataset.list_files(imagepath)
        dataset = dataset.map(lambda x: self.process_image_with_mask(x, maskspath))
        dataset = dataset.map(lambda x, y: self.process_data(x, y))
        dataset = dataset.shuffle(shuffle_buffer_size)
        return dataset.batch(batch_size)
