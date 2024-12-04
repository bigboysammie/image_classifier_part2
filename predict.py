import argparse
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models
from time import time

from load_checkpoint import load_checkpoint
from process_image import process_image

# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):

    model = load_checkpoint()
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model = model.to(device)
    
    model.eval() # Set model to evaluation mode
    
    
    #Call 'process_image' to load image as pytorch tensor for mo
    image = process_image(image_path)
    
    image = image.to(device)

    
    with torch.no_grad():
        #Perform forward pass through the model
        logps = model(image)
    
    #Convert log probability distribution to regular probability distribution 
    ps = torch.exp(logps)
        
    #Get a tensor containing the top K probabilities and corresponding classes
    probs, classes = ps.topk(topk, dim=1)
    
    #Convert probabilities & classes into flattened numpy arrays
    probs = probs.cpu().numpy().flatten()
    classes =  classes.cpu().numpy().flatten()
    
    #Return probs and classes as tuple containing probs and converted class names
    return probs, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    
    # Load model
    model = load_checkpoint(args.checkpoint)

    # Pass image from pre-trained model and receive predictions
    probs, classes = predict(args.image_path,model, args.top_k)

    # Load category names if provided by user
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[str(cls.item())] for cls in top_classes[0]]

    # Print results
    for i in range(args.top_k):
        print(f"Class: {top_classes[i]}, Probability: {top_probs[0][i].item()}")

if __name__ == '__main__':
    main()

    