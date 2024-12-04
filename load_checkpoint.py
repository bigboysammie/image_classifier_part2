import torch
from model_build import model_build

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    #Rebuild model using 'model_build' function passing in arguments from checkpoint dictionary
    model = model_build(checkpoint.get('arch'), checkpoint.get('hidden_units'))
    
    #Load weights into rebuilt model
    model.load_state_dict(checkpoint.get('model_state_dict'))

    #Assign classes for mapping
    model.class_to_idx = checkpoint.get('class_to_idx',{})

    return model