import logging
import base64
import io
import os
import torch
import numpy as np
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from torch import nn, Tensor
import torchvision
import torchvision.ops as ops

############## Define helper compatibility classes ###############
##################################################################
class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Compose:
    def __init__(self, transformers):
        self.transformerss = transformers

    def __call__(self, image, target):
        for t in self.transformerss:
          image, target = t(image, target)
        return image, target

def get_transform():
    transformers = []
    transformers.append(PILToTensor())
    transformers.append(ConvertImageDtype(dtype=torch.float32))
    return Compose(transformers)


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the model path
    model_path = os.getenv("AZUREML_MODEL_DIR", "")
    model_path = os.path.join(model_path, "model_chad.pt")

    # Cache the model in memory
    model = torch.load(model_path)
    model.to(device)

    #Set model in evaluate mode such that dropout layers are disabled
    model.eval()
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    # Get the bytes of the image that have been encoded to base 64
    decoded_data = base64.b64decode(raw_data)
    logging.info("Converted encoded image to raw Pillow image")

    # Parse the raw bytes to a PIL image. Save a copy for later use when drawing the detection boxes
    pil_image = Image.open(io.BytesIO(decoded_data))
    pil_image_rgb = pil_image.convert("RGB")
    logging.info("Converted encoded image to raw Pillow image")

    # Convert the image to a tensor
    tensor_image, _ = get_transform()(pil_image_rgb, {})
    logging.info("Tensor image ready")

    # Make the prediction for the requested image
    prediction = model([tensor_image.to(device)])
    logging.info("Model prediction ready")

    # Draw boxes around high confidence predictions
    # Convert the image to a numpy array such that we can change the color of each channel more easily 
    boxes = prediction[0]['boxes'].data.cpu().numpy().astype(int)
    scors = prediction[0]['scores'].data.cpu().numpy()

    im_array = np.array(pil_image)

    idx = 0
    for box in boxes:
        if(scors[idx] >= 0.8):
            im_array[box[1]:box[3], box[0]-4:box[0], :] = 0
            im_array[box[1]:box[3], box[0]-4:box[0], 1] = 255
            im_array[box[1]:box[3], box[2]:box[2]+4, :] = 0
            im_array[box[1]:box[3], box[2]:box[2]+4, 1] = 255
            im_array[box[1]:box[1]+4, box[0]:box[2], :] = 0
            im_array[box[1]:box[1]+4, box[0]:box[2], 1] = 255
            im_array[box[3]-4:box[3], box[0]:box[2], :] = 0
            im_array[box[3]-4:box[3], box[0]:box[2], 1] = 255
        idx += 1

    # Reconstruct a PIL image from the numpy array
    ret_image = Image.fromarray(np.uint8(im_array))
    
    # Convert the image back to raw bytes
    imgByteArr = io.BytesIO()
    ret_image.save(imgByteArr, format="JPEG")
    logging.info("Finish request")

    # Encode the bytes in base64 decoded to utf8
    return base64.b64encode(imgByteArr.getvalue()).decode("utf-8")