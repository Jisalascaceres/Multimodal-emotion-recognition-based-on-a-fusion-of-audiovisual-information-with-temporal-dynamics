import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
from sklearn.model_selection import train_test_split
import random 
from sklearn.metrics import confusion_matrix

from dataset_gen import EmotionDataset,discard
from model_zoo import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device





Modality = 'Mix'
dataset = 'SAVEE'
model_arch = 'PCA' #['image','audio','concat','AE','PCA','Embrace']
train = False
n_frames = 32
n_frames_LSTM = 16
batch_size =  16
if Modality == 'Audio':
    n_frames = 3
    n_frames_LSTM = 2

image_size = 2048
audio_size = 512

if dataset == 'RAVDESS':
    path_images = '/media/old/Datasets/Speech/By_emotions/'
    path_audio ='/media/old/Datasets/Speech/Splitted_Audio_By_emotions/'
    path_models = '/home/jose/Documents/GitHub/Paper_NN_emotions/Models/RADVESS/'
    path_results = '/home/jose/Documents/GitHub/Paper_NN_emotions/Results/RAVDESS/'
    n_emotions = 8
    
if dataset =='SAVEE':
    path_images = '/media/old/Datasets/Generados/Embeddings/Faces/'
    path_audio = '/media/old/Datasets/Generados/Embeddings/Voices/'
    path_models = '/home/jose/Documents/GitHub/Paper_NN_emotions/Models/SAVEE/'
    path_results = '/home/jose/Documents/GitHub/Paper_NN_emotions/Results/SAVEE/'
    n_emotions = 7
    
    
if dataset == 'CREMA-D':
    path_images = '/media/old/Datasets/CREMA-D/Generados/Embeddings/Images/'
    path_models = '/home/jose/Documents/GitHub/Paper_NN_emotions/Models/CREMA-D/'
    path_audio = '/media/old/Datasets/CREMA-D/Generados/Embeddings/Audio/AudioWAV/'
    n_emotions = 6
    
        


data_dir = path_images
video_folders = []

# Make a list of every video, to shuffle and split into train and validation sets

for emotion in os.listdir(data_dir): 
    emotion_dir = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_dir):

        video_folders.extend([os.path.join(emotion_dir, video_folder) for video_folder in os.listdir(emotion_dir)])



data_dir = path_audio
audio_folders = []

# Make a list of every video, to shuffle and split into train and validation sets

for emotion in os.listdir(data_dir): 
    emotion_dir = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_dir):

        audio_folders.extend([os.path.join(emotion_dir, video_folder) for video_folder in os.listdir(emotion_dir)])

video_folders.sort()
audio_folders.sort()
if dataset == 'CREMA-D':
    video_folders,audio_folders = discard(video_folders,audio_folders)
folders = [[video_folders[i], audio_folders[i]] for i in range (len(video_folders))]



# Shuffle the video folders
random.shuffle(folders)

# Split video folders 

train_video_folders, val_video_folders = train_test_split(folders, test_size=0.2, random_state=0)
val_video_folders, test_video_folders = train_test_split(val_video_folders, test_size=0.5, random_state=0)


# Create the datasets and data loaders
transform = None  # We don't apply any transform to the data


train_dataset = EmotionDataset(train_video_folders, transform=transform,dataset=dataset,modality=Modality)
val_dataset = EmotionDataset(val_video_folders, transform=transform,dataset=dataset,modality=Modality)
test_dataset = EmotionDataset(test_video_folders, transform=transform ,dataset=dataset,modality=Modality)

batch_size =  16


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True) # It will shuffle the sets of 32 but not the frames itself
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
print ('length of the datasets')
print ('Train Set',len(train_dataset))
print ('Validation Set',len(val_dataset))
print ('Test Set',len(test_dataset))

print ('length of the loaders')
print ('Train Set',len(train_loader))
print ('Validation Set',len(val_loader))
print ('Test Set',len(test_loader))

# Create the model

assert model_arch in ['image','audio','concat','AE','PCA','Embrace'], 'The model has to be one of the following: images, audio, concat, AE, PCA, Embrace'

if model_arch == 'image':
    model = Simple_LSTM_image(num_frames_LSTM = n_frames_LSTM,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,image_size = image_size)
    path_model = path_models + 'best_model_image.pth'

if model_arch == 'audio':
   
    model = Simple_LSTM_audio(num_frames_LSTM = n_frames_LSTM,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,audio_size = audio_size)
    path_model = path_models + 'best_model_audio.pth'
    
if model_arch == 'concat':
    model = Concatenation_net(num_frames_LSTM = n_frames_LSTM,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,image_size = image_size,audio_size = audio_size)
    path_model = path_models + 'best_model_concatenation.pth'

if model_arch == 'AE':
    Autoencoder = Autoencoder(image_size = image_size,audio_size = audio_size)
    path_model_AE = path_models + 'best_model_AE.pth'
    if train:
        AE_training_loop(
        epochs = 50,
        optimizer = optim.Adam(Autoencoder.parameters(), lr=1e-4),
        loss_fcn = nn.MSELoss(),
        val_data = val_loader,
        train_data = train_loader,
        model = Autoencoder,
        device = device,
        path_models=path_model_AE
            )
        
    Autoencoder.load_state_dict(torch.load(path_model_AE))
    
    model = Multimodal_Autoencoder(num_frames_LSTM = n_frames_LSTM,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,
                                   image_size = image_size,audio_size = audio_size,AE = Autoencoder)
        
    path_model = path_models + 'best_model_multimodal_encoder.pth'
    
if model_arch == 'PCA':
    
    from sklearn.decomposition import PCA
    pca_size = 0.95
    
    pca = PCA(n_components = pca_size)
    
    train_loader_PCA_train = DataLoader(train_dataset,batch_size = 1,shuffle = False)
    
    concat_data = []
    
    print ('Calculating PCA')
    for i, (data_image,data_audio, target) in tqdm(enumerate(train_loader_PCA_train)):
    
        data = torch.cat((data_image,data_audio), dim=2)
        data = data.view(n_frames,image_size+audio_size)
        concat_data.append(data.numpy())
    
    concat_data = np.array(concat_data)

    pca = pca.fit(concat_data.reshape(-1,image_size+audio_size))

    PCA_shape = pca.n_components_
    
    train_dataset_PCA = EmotionDataset(train_video_folders, transform=transform, dataset=dataset, modality='Mix',PCA = pca)
    val_dataset_PCA = EmotionDataset(val_video_folders, transform=transform, dataset=dataset, modality='Mix',PCA = pca)
    test_dataset_PCA = EmotionDataset(test_video_folders, transform=transform, dataset=dataset, modality='Mix',PCA = pca)
    
    train_loader = DataLoader(train_dataset_PCA, batch_size=batch_size, shuffle=True,drop_last=True) # It will shuffle the sets of 32 but not the frames itself
    val_loader = DataLoader(val_dataset_PCA, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset_PCA, batch_size=batch_size, shuffle=False,drop_last=True)
        
    model = PCA_net(PCA_size=PCA_shape,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,num_frames_lstm=n_frames_LSTM)
    path_model = path_models + 'best_model_PCA.pth'

if model_arch == 'Embrace':
    
    embrace_size = 1024
    model = Embrace(embrace_size=embrace_size,n_emotions = n_emotions,num_frames = n_frames,batch_size = batch_size,image_size = image_size,audio_size = audio_size,device = device)
    path_model = path_models + 'best_model_embrace.pth'
    
    
    
model.to(device)

if train:
    
    if Modality == 'Images' or 'Audio' or model_arch == 'PCA':
        simple_training_loop(
            n_epochs = 20,
            optimizer = optim.Adam(model.parameters(), lr=1e-4),
            loss_fcn = nn.CrossEntropyLoss(),
            val_dataset=val_loader,
            train_dataset=train_loader,
            name_save = path_model,
            model = model,
            device = device
        )
        
    if Modality == 'Mix':
        Mix_training_loop(
        n_epochs = 8,
        optimizer = optim.Adam(model.parameters(), lr=1e-5), # RADVESS 1e-4, SAVEE 1e-4
        loss_fcn = nn.CrossEntropyLoss(),
        val_dataset = val_loader,
        train_dataset = train_loader,
        model = model,
        device = device,
        name_save = path_model
         )
        
        
model.load_state_dict(torch.load(path_model))
model.eval()

if Modality == 'Images' or 'Audio' or model_arch == 'PCA':
    accuracy = simple_get_accuracy(test_loader,model,device)
    print('Accuracy of the network on the test images: %.2f %%' % (accuracy))
else:
    accuracy = Mix_get_accuracy(test_loader,model,device)
    print('Accuracy of the network on the test images: %.2f %%' % (accuracy))