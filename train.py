#Import necessary modules
import argparse
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models
from time import time

from preprocess_data import preprocess_data
from model_build import model_build
from checkpoint import checkpoint

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs,use_gpu=True):
    '''Uses flower image database and user fed parameters
    to build and train a neural network.
    they want those printouts (use non-default values)
    Parameters:
      data_dir - File pathway to image database to be used for training
      save_dir -File pathway to save location of model
      arch - User input pre-trained architecture
      learning_rate = User specified model learning rate
      hidden_units = Number of hidden units within hidden layers
      epochs = Sets number of times the model will interate through the 
      entire dataset
      use_gpu = Specifies whether the use wishes to use GPU

    
    Returns:
           None - Prints information about epoch, training loss, validation loss and
           validation accuracy. The function will also save the trained neural network
           in the location specified by the user.

    '''
    # Call 'preprocess_data' function to load, preprocess and return datasets
    trainloader, testloader,validloader, train_data = preprocess_data(data_dir)

    # Call 'model_builder' to build a model of defined type and size
    model, inputs = model_build(arch, hidden_units)

    # Set loss function to be implemented to calculate errors
    criterion = nn.NLLLoss()

    # Set optmizer function for network to adjust weights
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = epochs
    print_every = 50
    steps = 0

    print("Model training in progress..")
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0
    
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()
                    
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    end_time = time.time()
    elapsed_time = int((end_time - start_time)/60)

    print(f"Completion time: {elapsed_time}mins")
    # Save the checkpoint
    checkpoint = checkpoint(model,arch,inputs, hidden_units,epochs, train_data)
    torch.save(checkpoint, 'model_checkpoint.pth')
    print("Checkpoint saved successfully.")
    
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset of flower images.')

    #Command line inputs from user

    #Argument 1: Image directory as '--data_dir' with default location of 'flowers/'
    parser.add_argument('--data_dir', type=str, default='flowers/' help='Directory of the training data')
    #Argument 2: Save directory as '--save_dir' with default location of ''
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    #Argument 3: Neural network pre-trained architecture as '--arch' with default  of 'vgg16'
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    #Argument 4: Model learning rate as '--learning_rate' with default value of 0.003
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    #Argument 5: Hidden units within classification layer as '--hidden_units' with default value of 512
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    #Argument 6: Model epochs as '--lepochs' with default value of 5
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    #Argument 4: GPU use specified as '--gpu' with no default set
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    train(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == '__main__':
    main()