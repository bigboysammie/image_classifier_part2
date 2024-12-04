# TODO: Process a PIL image for use in a PyTorch model
from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Define aspect ratio and required crop size
    aspect_ratio = 256
    crop_size = 224
    
    # Load the image
    pil_image = Image.open(image)
    
    #Resize the image to the set aspect ratio
    pil_image = pil_image.resize((aspect_ratio,aspect_ratio))
    
    #Set parameters for image crop
    width, height = pil_image.size
    new_width, new_height = crop_size, crop_size
    
    # Calculate crop box coordinates
    left = (width - new_width) // 2
    right = (width + new_width) // 2
    top = (height - new_height) // 2
    bottom = (height + new_height) // 2
    
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert cropped pil image to np array
    np_image = np.array(pil_image, dtype=np.float32)
    
    #Normalise NP array
    np_image /= 255
    
    # Normalize colour channels using the specified means and standard deviations
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    #Transpose np array to match expected input for Pytorch array
    np_image = np_image.transpose((2,0,1))
    
    #Convert numpy image to pytorch tensor
    image_pytorch = torch.from_numpy(np_image).float().unsqueeze(0)
    
    #Return pre-processed image
    return image_pytorch
    