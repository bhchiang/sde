from flax import serialization
import os

model_path = "models/"
os.makedirs(model_path, exist_ok=True)


def _save_model(model, fname):
    bytes_output = serialization.to_bytes(model)
    with open(os.path.join(model_path, fname), 'wb') as f:
        f.write(bytes_output)


# model = variables
def _load_model(model, fname):
    path = os.path.join(model_path, fname)
    ifile = open(pretrained_path, 'rb')
    bytes_input = ifile.read()
    ifile.close()
    variables = serialization.from_bytes(variables, bytes_input)
    return variables