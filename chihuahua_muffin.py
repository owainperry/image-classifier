from keras.src.preprocessing.image import ImageDataGenerator

from constant import RuntimeMode
from model import FoundationXceptionModelBase


class ChihuahuaMuffinModel(FoundationXceptionModelBase):
    """
    This class encapsulates the model training and prediction logic for the Chihuahua Muffin app.
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
        train_datagen = ImageDataGenerator(
            rescale=1/127.5,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1/127.5)
        training_set = train_datagen.flow_from_directory(
            self.training_data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode=label_mode,
        )

        test_dataset = test_datagen.flow_from_directory(
            self.test_data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode=label_mode,
        )
        return training_set, test_dataset


if __name__ == '__main__':
    DATA_CM_ROOT_PATH = 'data/chihuahua-muffin'
    DATA_ROOT_PATH = 'data'
    model = ChihuahuaMuffinModel(
        model_path="chihuahua_muffin_foundation_ai.keras",
        # model_path="xception_attempt81_seq.keras",
        classes=["chihuahua", "muffin"],
        training_data_path=f"{DATA_CM_ROOT_PATH}/train",
        test_data_path=f"{DATA_CM_ROOT_PATH}/test",
        train_epochs=5,
        fine_tune_epochs=3,
        patience=3,
    )
    model.load()
    model.to_onnx()

    run_mode = RuntimeMode.ONNX
    model.predict(f"{DATA_CM_ROOT_PATH}/test/chihuahua/img_2_779.jpg", mode=run_mode)
    model.predict(f"{DATA_CM_ROOT_PATH}/test/muffin/img_3_749.jpg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/muffin.png", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/muffin2.jpeg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/muffin3.jpg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/muffin4.jpeg", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/chihuahua.png", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/chihuahua2.png", mode=run_mode)
    model.predict(f"{DATA_ROOT_PATH}/app/chihuahua3.png", mode=run_mode)
