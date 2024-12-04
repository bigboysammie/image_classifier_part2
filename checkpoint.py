def checkpoint(model,arch,input_size, hidden_units,epochs):

# Save the class_to_idx mapping
    model.class_to_idx = train_data.class_to_idx

# Create a checkpoint dictionary
    checkpoint = {
    'input_size': input_size,
    'arch': arch,
    'output_size': 102,  
    'hidden_units': hidden_units, 
    'epochs': epochs,  
    'model_state_dict': model.state_dict(),  
    'optimizer_state_dict': optimizer.state_dict(),  
    'class_to_idx': model.class_to_idx
}

    return checkpoint
