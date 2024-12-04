from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def preprocess_data(data_dir):
    '''Performs necessary preprocessing on image dataset to prepare data
    to be processed through neural network
    Parameters:
      data_dir - File pathway to image database to be used for training


    
    Returns:
           trainloader - returns batches of processed image files into pytorch tensors for model training
           testloader - returns batches of processed image files into pytorch tensors for model testing
           validloader - returns batches of processed image files into pytorch tensors for model validation
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets

    # Set variable for crop size of image
    crop_size = 224

    # Degrees of rotation added to random images
    random_rotation = 30

    # Normalisation of colour channels to speed up training 
    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]

    # Resize image to prepare for cropping
    image_resize = 256

    #Set batch size for dataloader
    batch_size = 32

    # Define training transform including random rotation, cropping, and flipping
    train_transforms = transforms.Compose([transforms.RandomRotation(random_rotation),
                     transforms.RandomResizedCrop(crop_size),
                     transforms.RandomHorizontalFlip(), 
                     transforms.ToTensor(),
                     transforms.Normalize(mean_norm,std_norm)])

    # Define transform config for testing and validation
    common_transforms = ([transforms.Resize(image_resize),
                   transforms.CenterCrop(crop_size),
                   transforms.ToTensor(),
                      transforms.Normalize(mean_norm,std_norm)])

    # Define testing and validation transforms from 'common_transforms' template
    test_transforms = transforms.Compose(common_transforms)
    valid_transforms = transforms.Compose(common_transforms)

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader,validloader, train_data