from hotdog_not_hotdog import HotdogNotHotdogModel
from constant import RuntimeMode

DATA_ROOT_PATH = '/tmp/data'
model = HotdogNotHotdogModel(
    model_path="hotdog_not_hotdog_foundation_ai.keras",
    training_data_path=f"{DATA_ROOT_PATH}/train",
    classes=["hotdog", "not_hotdog"],
    test_data_path=f"{DATA_ROOT_PATH}/test",
    train_epochs=3,
    fine_tune_epochs=1,
    patience=3,
)

# Train model
model.train_and_save()
