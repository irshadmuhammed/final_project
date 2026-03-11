import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from skimage.transform import resize


class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, class_names, source_image_dir, tokenizer_wrapper, batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1, tags_column='Manual Tags', tags_list=None):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.tokenizer_wrapper = tokenizer_wrapper
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.tags_column = tags_column
        self.tags_list = tags_list
        # Create tag to index mapping
        if self.tags_list:
            self.tag_to_index = {tag: i for i, tag in enumerate(self.tags_list)}
        
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        
        batch_y_text = self.y_text[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_tags = self.y_tags[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Return format: (inputs, targets)
        # where inputs is usually just images
        # targets needs to match what the model expects/returns or what training loop expects output
        
        # Current train loop expects: img, target, _ = next(generator)
        # and target[:, 1:] is used for caption loss.
        # We need to change the yield signature.
        # Let's keep existing signature but pack tags into second argument?
        # Or better, return (batch_x, (batch_y_text, batch_y_tags), batch_x_path)
        
        return batch_x, (batch_y_text, batch_y_tags), batch_x_path

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y_text[:self.steps * self.batch_size, :]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path = df["Image Index"].values
        
        # Prepare captions
        if self.augmenter is not None:
             self.y_text = self.tokenizer_wrapper.GPT2_encode(df[self.class_names].values)
        else:
             self.y_text = self.tokenizer_wrapper.GPT2_encode(df[self.class_names].values, max_length=1000)

        # Prepare tags (Multi-hot encoding)
        if self.tags_list:
            raw_tags = df[self.tags_column].fillna("").values
            num_samples = len(raw_tags)
            num_classes = len(self.tags_list)
            self.y_tags = np.zeros((num_samples, num_classes), dtype=np.int32)
            
            for i, tags_str in enumerate(raw_tags):
                current_tags = [t.strip() for t in str(tags_str).split(',')]
                for tag in current_tags:
                    if tag in self.tag_to_index:
                        self.y_tags[i, self.tag_to_index[tag]] = 1
        else:
             self.y_tags = np.zeros((len(self.x_path), 0)) # Empty if no tags list provided

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
