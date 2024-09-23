
import torch 
import torch.nn as nn

# Script with the models used in the paper: Multimodal Emotion Recognition based on a Fusion of Audiovisual Information with Temporal Dynamics
'''
The overall architectures of the model is the following:
1. The embedders of both audio and image. To the models it enters the embedding generated. The size of the embeddings will be x and y
2. LSTM layer with f(x,y) neurons and x layer. The goal is to capture the states of each video, in a way that the model learn the different states the face pass for the different emotions
3. Fully connected layer with 500 neurons and a dropout of 0.5
4. Fully connected layer with n_emotions neurons, one for each emotion

The paper we mention is tituled "Emotion recognition at a distance: The robustness of machine learning
based on hand-crafted facial features vs deep learning models" 

'''

class Concatenation_net(nn.Module):
    
    def __init__(self,num_frames_LSTM = 16,n_emotions = 8,num_frames = 32,batch_size = 16,image_size = 2048,audio_size = 512,num_layers_LSTM = 1):
        super(Concatenation_net, self).__init__()
        
        self.num_frames_lstm = num_frames_LSTM
        self.n_emotions = n_emotions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.image_size = image_size
        self.audio_size = audio_size
        
        
        self.lstm = nn.LSTM(input_size=self.image_size + self.audio_size, # Size of every single input
                            hidden_size= self.num_frames_lstm, # Number of frames the lstm will be able to see, it has to be less than the number of frames we insert in the lstm
                            num_layers=num_layers_LSTM,  # Number of layers stacked on the lstm
                            batch_first=True
                            )
        
        self.dense1 = nn.Sequential(
            nn.Linear(self.num_frames*self.num_frames_lstm, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        
        self.dense2= nn.Linear(500, self.n_emotions)
        
        
    def forward(self, x, y):
        
        embed_image = x#
        embed_audio = y
        
        
        x = torch.cat((embed_image, embed_audio), dim=2)
        
       
        
        x = x.view(self.batch_size, self.num_frames,self.image_size+self.audio_size) 
        
        x = self.lstm(x) # we pass the data to the lstm layer
        
        #print (x[0].shape)
        #x = x[0] # we get the output of the lstm layer
        
        x = x[0].reshape(self.batch_size, self.num_frames*self.num_frames_lstm) # we reshape the data to have batch_size, 8
        x = self.dense1(x) 
        
        x = self.dense2(x)
        x = x.view(self.batch_size, -1) # we reshape the data to have batch_size, 8
        return x
    

class PCA_net(nn.Module):
    
    def __init__(self,PCA_size,num_frames_lstm = 16,n_emotions = 8,num_frames = 32,batch_size = 16,num_layers_LSTM = 1):
        super(PCA_net, self).__init__()
    
        
        self.PCA_size = PCA_size
        self.num_frames_lstm = num_frames_lstm
        self.n_emotions = n_emotions
        self.num_frames = num_frames
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_size=self.PCA_size, # Size of every single input
                            hidden_size= self.num_frames_lstm, # Number of frames the lstm will be able to see, it has to be less than the number of frames we insert in the lstm
                            num_layers=num_layers_LSTM,  # Number of layers stacked on the lstm
                            batch_first=True
                            )
        
        self.dense1 = nn.Sequential(
            nn.Linear(self.num_frames*self.num_frames_lstm, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        
        self.dense2= nn.Linear(500, n_emotions)
        
        
    def forward(self, x):
        
        x = x.view(self.batch_size, self.num_frames, self.PCA_size)
        x = self.lstm(x) # we pass the data to the lstm layer
        
        #print (x[0].shape)
        #x = x[0] # we get the output of the lstm layer
        
        x = x[0].reshape(self.batch_size, self.num_frames*self.num_frames_lstm) # we reshape the data to have batch_size, 8
        x = self.dense1(x) 
        
        x = self.dense2(x)
        x = x.view(self.batch_size, -1) # we reshape the data to have batch_size, 8
        return x
    
    
class Autoencoder(nn.Module):
    
    def __init__(self,image_size = 2048, audio_size = 512, num_frames = 32, batch_size = 16):
        super(Autoencoder, self).__init__()
    
        self.image_size = image_size
        self.audio_size = audio_size
        self.num_frames = num_frames
        self.batch_size = batch_size
        
        self.encoder = nn.Sequential(
            nn.Linear(self.image_size+self.audio_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size+audio_size),
            nn.ReLU())
    
        
        
    def forward(self, x, y):
        
        embed_image = x#
        embed_audio = y
        
        
        x = torch.cat((embed_image, embed_audio), dim=2)
        x = x.view(self.batch_size, self.num_frames,self.image_size+self.audio_size)
        x = self.encoder(x) 
        x = self.decoder(x)
        return x
    

class Multimodal_Autoencoder(nn.Module):
        
    def __init__(self,image_size = 2048, audio_size = 512, num_frames = 32, batch_size = 16, num_frames_LSTM = 16, n_emotions = 8, AE = Autoencoder(),
                 num_layers_LSTM = 1):
        super(Multimodal_Autoencoder, self).__init__()
            
            
         # use the encoder of the autoencoder, freesze the weights
        self.encoder = AE.encoder
        
        self.image_size = image_size
        self.audio_size = audio_size
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_frames_lstm = num_frames_LSTM
        self.n_emotions = n_emotions
        
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(input_size=512, # Size of every single input
                        hidden_size= self.num_frames_lstm, # Number of frames the lstm will be able to see, it has to be less than the number of frames we insert in the lstm
                        num_layers=num_layers_LSTM,  # Number of layers stacked on the lstm
                        batch_first=True
                        )
    
        self.dense1 = nn.Sequential(
            nn.Linear(self.num_frames*self.num_frames_lstm, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
    
        self.dense2= nn.Linear(500, n_emotions)
        
        
    def forward(self, x, y):
        
        embed_image = x#
        embed_audio = y
        x = torch.cat((embed_image, embed_audio), dim=2)
        #x = x.view(batch_size, num_frames,512+2048)
        x = self.encoder(x)
        x = self.lstm(x)
        x = x[0].reshape(self.batch_size, self.num_frames*self.num_frames_lstm) # we reshape the data to have batch_size, 8
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(self.batch_size, -1) # we reshape the data to have batch_size, 8
        return x    
    
    
# The embracenet class is the extracted from the paper "EmbraceNet: A robust deep learning architecture for multimodal classification"  
# taken from the github repository https://github.com/idearibosome/embracenet

class EmbraceNet(nn.Module):
  
  def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
    """
    Initialize an EmbraceNet module.
    Args:
      device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
      input_size_list: A list of input sizes.
      embracement_size: The length of the output of the embracement layer ("c" in the paper).
      bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
    """
    super(EmbraceNet, self).__init__()

    self.device = device
    self.input_size_list = input_size_list
    self.embracement_size = embracement_size
    self.bypass_docking = bypass_docking

    if (not bypass_docking):
      for i, input_size in enumerate(input_size_list):
        setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))


  def forward(self, input_list, availabilities=None, selection_probabilities=None):
    """
    Forward input data to the EmbraceNet module.
    Args:
      input_list: A list of input data. Each input data should have a size as in input_size_list.
      availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
      selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
    Returns:
      A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
    """

    # check input data

    
    assert len(input_list) == len(self.input_size_list)
    num_modalities = len(input_list)
    batch_size = input_list[0].shape[0]
    

    # docking layer
    docking_output_list = []
    if (self.bypass_docking):
      docking_output_list = input_list
    else:
      for i, input_data in enumerate(input_list):
        x = getattr(self, 'docking_%d' % (i))(input_data)
        x = nn.functional.relu(x)
        docking_output_list.append(x)
    

    # check availabilities
    if (availabilities is None):
      availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
    else:
      availabilities = availabilities.float()
    

    # adjust selection probabilities
    if (selection_probabilities is None):
      selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
    selection_probabilities = torch.mul(selection_probabilities, availabilities)

    probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
    selection_probabilities = torch.div(selection_probabilities, probability_sum)


    # stack docking outputs
    docking_output_stack = torch.stack(docking_output_list, dim=-1)  # [batch_size, embracement_size, num_modalities]


    # embrace
    modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
    modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

    embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
    embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

    return embracement_output

    
    
class Embrace (nn.Module):
    
    def __init__(self,device, embrace_size = 256, num_frames_LSTM = 16, n_emotions = 8, num_frames = 32, batch_size = 16,image_size = 2048, audio_size = 512, num_layers_LSTM = 1):
        super(Embrace, self).__init__()
        
        self.device = device
        self.embrace_size = embrace_size
        self.num_frames_lstm = num_frames_LSTM
        self.n_emotions = n_emotions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.image_size = image_size
        self.audio_size = audio_size
        
        
        self.embracenet = EmbraceNet(device= device, input_size_list=[ self.image_size,  self.audio_size],
                                    embracement_size=self.embrace_size)
        
        self.lstm = nn.LSTM(input_size=self.embrace_size, # Size of every single input
                        hidden_size= self.num_frames_lstm, # Number of frames the lstm will be able to see, it has to be less than the number of frames we insert in the lstm
                        num_layers=num_layers_LSTM,  # Number of layers stacked on the lstm
                        batch_first=True
                        )
        
        self.dense1 = nn.Sequential(
            nn.Linear(self.num_frames*self.num_frames_lstm, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        
    
        
        self.dense2= nn.Linear(256, n_emotions)
        
        
    def forward(self,x,y):

        x = x.view(1,self.batch_size, self.num_frames, 2048)
        y = y.view(1,self.batch_size, self.num_frames, 512)

        embrace = self.embracenet(input_list=[x,y])
        

        x = embrace.reshape(self.batch_size, self.num_frames, self.embrace_size)
 
        x = self.lstm(x)
        x = x[0].reshape(self.batch_size, self.num_frames*self.num_frames_lstm) # we reshape the data to have batch_size
 
        x = self.dense1(x)
 
        x = self.dense2(x)
        
        
        return x
 
 
class Simple_LSTM_image (nn.Module): # Create the model class
    
    

    
    
    def __init__(self,num_frames_LSTM = 16,n_emotions = 8,num_frames = 32,batch_size = 16,image_size = 2048,num_layers_LSTM = 1):
        super(Simple_LSTM_image, self).__init__()
        
        self.num_frames_LSTM = num_frames_LSTM
        self.n_emotions = n_emotions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.lstm = nn.LSTM(
            input_size = self.image_size, # length of the embeddings of the frames 
            hidden_size = self.num_frames_LSTM, # number of cells in the LSTM layer, number of frames the lstm will be able to see at once
            num_layers = num_layers_LSTM,
            batch_first = True
        )
        self.dense1 = nn.Sequential(
            nn.Linear(num_frames*num_frames_LSTM, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        
        self.dense2= nn.Linear(500, n_emotions)
        
    def forward(self, x):
        
        batch_size,frames, C, H, W = x.shape
        x = x.view(batch_size, self.num_frames,self.image_size) # we reshape the data to have,,1,2*2*512
        x = self.lstm(x) # we pass the data through the lstm layer
        

        x = x[0].reshape(batch_size, self.num_frames*self.num_frames_LSTM) # we reshape the data to have batch_size
        
        x = self.dense1(x) 
        
        x = self.dense2(x)
        x = x.view(batch_size, -1) # we reshape the data to have batch_size, n_emotions
        return x
    
    
class Simple_LSTM_audio (nn.Module): # Create the model class
    
    

    
    
    def __init__(self,num_frames_LSTM = 16,n_emotions = 8,num_frames = 32,batch_size = 16,audio_size = 512,num_layers_LSTM = 1):
        super(Simple_LSTM_audio, self).__init__()
        
        self.num_frames_lstm = num_frames_LSTM
        self.n_emotions = n_emotions
        self.num_frames = num_frames    
        self.batch_size = batch_size
        self.audio_size = audio_size
        
        self.lstm = nn.LSTM(input_size=self.audio_size, # Size of every single input
                            hidden_size= self.num_frames_lstm, # Number of frames the lstm will be able to see, it has to be less than the number of frames we insert in the lstm
                            num_layers=num_layers_LSTM,  # Number of layers stacked on the lstm
                            batch_first=True
                            )
        
        self.dense1 = nn.Sequential(
            nn.Linear(self.num_frames*self.num_frames_lstm, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        
        self.dense2= nn.Linear(500, self.n_emotions)
        
        
    def forward(self, x):
        batch_size,frames,columns = x.shape # we get the shape of the data
       
        
        x = x.view(self.batch_size, self.num_frames,512) 
        x = self.lstm(x) 
        
        x = x[0].reshape(self.batch_size, self.num_frames*self.num_frames_lstm) 
        x = self.dense1(x) 
        
        x = self.dense2(x)
        x = x.view(self.batch_size, -1) 
        return x