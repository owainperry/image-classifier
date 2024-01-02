import enum


class RuntimeMode(enum.Enum):
    KERAS = "keras"
    ONNX = "onnx"


class ModelExtension(enum.Enum):
    KERAS = ".keras"
    ONNX = ".onnx"