import logging
import os

import keras
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.src.applications.xception import preprocess_input
from keras.src.layers import Flatten
from keras.src.losses import CategoricalCrossentropy
from keras.src.utils import image_dataset_from_directory

import preprocessing
from constant import RuntimeMode, ModelExtension
from util import timeit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoundationXceptionModelBase:
    """
    This class encapsulates the model training and prediction logic.
    It is trained based on the Xception model (CNN model) with transfer learning.
    """

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        classes: list[str],
        train_epochs: int = 3,
        fine_tune_epochs: int = 1,
        patience: int = 5,
        model_path: str = "hotdog_not_hotdog_foundation_ai.keras",
    ):
        """
        Initialize the model.

        :param model_path: Model file path
        :param training_data_path: Training data path
        :param test_data_path: Test data path
        :param train_epochs: Epochs for training
        :param fine_tune_epochs: Epochs for fine tuning
        :param patience: Patience for early stopping
        """
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.train_epochs = train_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.patience = patience
        self.img_width, self.img_height = 299, 299
        self.input_shape = (self.img_width, self.img_height, 3)
        self.model_path = model_path
        self.model = None
        self.classes = classes

        if not model_path:
            raise ValueError("model_path is required")

    def load(self):
        """
        Load the model from a file.
        :return: Model
        """
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path '{self.model_path}' does not exist, you have to train using '.train_and_save()'.")
        logger.info(f"Loading model from '{self.model_path}'...")
        self.model = keras.models.load_model(self.model_path)
        return self.model

    def train_and_save(self):
        """
        Train the model and save it to a file.
        :return: Model
        """
        logger.info(f"Model path '{self.model_path}' does not exist, initializing model training.")
        self.model = self.train()
        logger.info(f"Saving model to '{self.model_path}'...")
        self.save_model(self.model)
        return self.model

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

    def _get_base_model(self):
        """
        Get the base model (Xception) for transfer learning.
        :return: Base model
        """
        return keras.applications.Xception(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False,
        )

    @staticmethod
    def _freeze_layers(model):
        """
        Freeze all layers in the model.
        :param model: Model to freeze
        :return: Model with frozen layers
        """
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        return model

    @staticmethod
    def _unfreeze_layers(model):
        """
        Unfreeze all layers in the model.
        :param model: Model to unfreeze
        :return: Model with unfrozen layers
        """
        model.trainable = True
        for layer in model.layers:
            layer.trainable = True
        return model

    def _fine_tune(self, base_model, num_of_output_layers: int = None):
        """
        Create new layers for fine tuning.
        :param base_model: Base model
        :param num_of_output_layers: Number of output layers to add
        :return: Model with new layers
        """
        _num_of_output_layers = num_of_output_layers or len(self.classes)

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        # model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(_num_of_output_layers, activation="softmax"))
        return model

    @staticmethod
    def _compile(model, learning_rate: float = 0.001):
        """
        Compile the model.
        :param model: Model to compile
        :param learning_rate: Learning rate for Adam optimizer (default: 0.001)
        :return: Compiled model
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy', 'Precision', 'Recall'],
        )
        return model

    @timeit
    def train(self):
        """
        Train the model.
        :return: Trained model instance
        """
        training_set, test_dataset = self.load_datasets()

        # Create Xception base model and freeze layers
        base_model = self._get_base_model()
        self._freeze_layers(base_model)

        # Create new layers and compile model to train this use case
        model = self._fine_tune(base_model)
        self._compile(model, learning_rate=1e-5)
        model.fit(training_set, epochs=self.train_epochs, validation_data=test_dataset)

        # Unfreeze layers and compile model to fine tune
        # Define a callback to stop training early if validation loss stops improving
        self._unfreeze_layers(base_model)
        self._compile(model, learning_rate=1e-6)
        callbacks = [
            EarlyStopping(patience=self.patience),
        ]
        model.fit(training_set, epochs=self.fine_tune_epochs, validation_data=test_dataset, callbacks=callbacks)
        return model

    def save_model(self, model):
        """
        Save the model to a file.
        :param model: Model to save
        """
        model.save(self.model_path)

    @timeit
    def predict(self, image_path: str, verbose: bool = False, mode: RuntimeMode = RuntimeMode.KERAS):
        """
        Infer the class of an image.
        :param image_path: Image path
        :param verbose: Verbose flag
        :param mode: Runtime mode
        :return: Predicted class and probability
        """
        _image = self.preprocess_foundation(image_path)
        if mode == RuntimeMode.ONNX:
            if not isinstance(self.model, rt.InferenceSession):
                raise ValueError(f"Model is not loaded in ONNX format, you have to load using '.load_from_onnx()'.")
            # dense_1 is the output layer name
            prediction = self.model.run(["dense_1"], {"input": _image})
        else:
            # Keras model
            prediction = self.model.predict(_image, verbose=2 if verbose else 0)
        label, probability = self.decode_predictions(prediction)
        logger.info(f"Path: {image_path}: Predicted class '{label}' with probability {probability}%")
        return label, probability, prediction

    def decode_predictions(self, prediction):
        """
        Decode the prediction.
        :param prediction: Prediction
        :return: Decoded prediction
        """
        prediction_max = np.argmax(prediction)
        if isinstance(prediction, list):
            # Output from ONNX is a list
            prediction = prediction[0]
        result = [(self.classes[prediction_max], float(prediction[i][prediction_max]) * 100.0) for i in
                  range(len(prediction))]
        result.sort(reverse=True, key=lambda x: x[1])
        (class_name, prob) = result[0]
        return class_name, prob

    def preprocess(self, image_path: str):
        """
        Preprocess the image for the model.
        :param image_path: Image path
        :return: Preprocessed image
        """
        img = image.load_img(image_path, target_size=(self.img_height, self.img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def preprocess_foundation(self, image_path: str):
        img = preprocessing.load_img(image_path, target_size=(self.img_height, self.img_width))
        x = preprocessing.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocessing.preprocess_input(x, mode="tf")
        return x


    @timeit
    def load_from_onnx(self):
        """
        Load the model from ONNX format.
        :return: Model Inference Session
        """
        providers = ['CPUExecutionProvider']
        path_name = self.model_path.replace(ModelExtension.KERAS.value, ModelExtension.ONNX.value)
        self.model = rt.InferenceSession(path_name, providers=providers)
        return self.model

    def to_onnx(self):
        """
        Convert the model to ONNX format.
        https://github.com/onnx/tensorflow-onnx/issues/2262
        """
        if self.model is None:
            raise ValueError(f"Model is not loaded, you have to load using '.load()'.")
        logger.info(f"Converting model to ONNX format...")
        file_name = self.model_path.replace(".keras", ".onnx")
        input_signature = [tf.TensorSpec((None, self.img_width, self.img_height, 3), tf.float32, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=13)
        onnx.save(onnx_model, file_name)
        output_names = [n.name for n in onnx_model.graph.output]  # used in ONNX predict
        logger.info(f"Saved model to '{file_name}'")
        logger.info(f"Output names: {output_names}")
