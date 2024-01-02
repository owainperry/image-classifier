from keras.src.applications.xception import preprocess_input
from keras.src.utils import image_dataset_from_directory

from constant import RuntimeMode
from model import FoundationXceptionModelBase


class HotdogNotHotdogModel(FoundationXceptionModelBase):
    """
    This class encapsulates the model training and prediction logic for the Hotdog Not Hotdog app.
    It is trained based on the Xception model (CNN model) with transfer learning.
    """

    def load_datasets(self):
        """
        Load training and test datasets.
        :return: Training and test datasets tuple
        """
        # Load from directory
        label_mode = "categorical"
        batch_size = 32
        training_set = image_dataset_from_directory(
            self.training_data_path,
            shuffle=True,
            batch_size=batch_size,
            image_size=(self.img_height, self.img_width),
            label_mode=label_mode,
        )
        test_dataset = image_dataset_from_directory(
            self.test_data_path,
            shuffle=True,
            batch_size=batch_size,
            image_size=(self.img_height, self.img_width),
            label_mode=label_mode,
        )
        # Preprocess images for Xception model
        training_set = training_set.map(lambda x, y: (preprocess_input(x), y))
        test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))
        return training_set, test_dataset


if __name__ == '__main__':
    DATA_ROOT_PATH = 'data'
    model = HotdogNotHotdogModel(
        model_path="hotdog_not_hotdog_foundation_ai.keras",
        # model_path="xception_attempt81_seq.keras",
        classes=['hotdog', 'nothotdog'],
        training_data_path=f"{DATA_ROOT_PATH}/train",
        test_data_path=f"{DATA_ROOT_PATH}/test",
        train_epochs=20,
        fine_tune_epochs=10,
        patience=3,
    )
    # model.to_onnx()
    model.load_from_onnx()

    run_mode = RuntimeMode.ONNX
    # model.predict(f"{DATA_ROOT_PATH}/app/hotdog.png", mode=run_mode)
    # model.predict(f"{DATA_ROOT_PATH}/app/pizza.png", mode=run_mode)
    # model.predict(f"{DATA_ROOT_PATH}/app/HotDog.jpg", mode=run_mode)
    # model.predict(f"{DATA_ROOT_PATH}/app/Pizza.jpg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/train/hotdog/1.jpg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/train/nothotdog/1.jpg", mode=run_mode)