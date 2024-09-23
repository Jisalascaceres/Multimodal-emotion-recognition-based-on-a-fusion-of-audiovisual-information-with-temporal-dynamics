import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm



def get_index (filepath,dataset):
    if dataset == 'RAVDESS':
        index = filepath.split('.npy')[0].split('-')[-1]
    if dataset == 'SAVEE':
        index = filepath.split('.npy')[0].split('/')[-1]
    if dataset == 'CREMA-D':
        index = filepath.split('.npy')[0].split('/')[-1]
    return int(index)


def get_emotion (filepath,dataset):
    
    if dataset == 'RAVDESS':
        # RAVDESS has 8 emotions.
        emotion = os.path.basename(os.path.dirname(os.path.dirname(filepath))) # Get the emotion of the video, is the name of the folder that contains the video folder
        
        '''
        Savee has 7 emotions:
        - a: anger
        - d: disgust
        - f: fear
        - h: happiness
        - n: neutral
        - sa: sadness
        - su: surprise
        '''
        return emotion        
        
    if dataset == 'SAVEE':
        
        emotion = filepath.split('/')[-2]
        # The memotion is the name an the number. split the name and the number, is not separated by nothing, is just "sa23"
        aux = emotion[0] # Get the first letter of the emotion, that is the emotion
        if aux == 's':
            aux = emotion[0:2] # If the emotion is sadness or surprise, get the first two letters
        
        emotion = aux
        
        dict_emotions = {'a':0,'d':1,'f':2,'h':3,'n':4,'sa':5,'su':6}
        
        emotion = dict_emotions[emotion]
        return emotion
        
        
    if dataset == 'CREMA-D':
        emotion = filepath.split('/')[-2]
        emotion = emotion.split('_')[2]
        dict_emotions = {'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
        emotion = dict_emotions[emotion]
        return emotion    



# The data is in .npy and is organized in 8 folders that correspond to the 8 emotions
# Each folder contains different folder representing diferent videos. Each video folder contains the npy files in order of the frames
# Each npy file contains the embeddings of the frames of the video
# THe audios are organized in the same way, but each video folder contains 3 audio files.

# Load the data


# The train and test data is in the same folder, so we need to separate it
# The train data will be the first 80% of the data and the test data will be the last 20% of the data


class EmotionDataset(Dataset):
    def __init__(self, folders,frames_number=32,aud_number=3, transform=None,PCA=None,dataset='RAVDESS',modality = 'Mix',image_shape = 2048, audio_shape = 512):
        self.video_folders = folders # List of video folders
        self.transform = transform # Transform to be applied to each video, none
        self.frames_number = frames_number # Number of frames per video
        self.dataset = dataset
        self.aud_number = aud_number # Number of audio files per video
        self.PCA = PCA
        self.Modality = modality
        self.image_shape = image_shape
        self.audio_shape = audio_shape
        assert self.Modality in ['Image','Audio','Mix'], 'The modality must be Image, Audio or Mix'
        assert self.dataset in ['RAVDESS','SAVEE','CREMA-D'], 'The dataset must be RAVDESS, SAVEE or CREMA-D'


        self.filepaths = self._get_filepaths() # List of filepaths for each video
        
        
    def _get_filepaths(self): # Get the paths of the 32 frames sequence of each video and the audio files. 

        filepaths_image = []    
        frames_video = []
        filepaths_audio = []
        auds = []
        filepaths_mix = [] 
        
        # Return the path of the frames of each video
        
        filepaths_image = []    
        frames_video = []
        filepaths_audio = []
        auds = []
        filepaths_mix = [] 

        # Return the path of the frames of each video

        for video_folder in self.video_folders: # for each video folder, get the filepaths of the embeddings
            npy_files = [file for file in os.listdir(video_folder[0]) if file.endswith('.npy')]  
            npy_files.sort() # We sort them to have them in order of appearance in the video
            frames_video.extend([os.path.join(video_folder[0], npy_file) for npy_file in npy_files])
            filepaths_image.append (frames_video)
            frames_video = []
            
            npy_files = [file for file in os.listdir(video_folder[1]) if file.endswith('.npy')]
            npy_files.sort() # We sort them to have them in order of appearance in the video
            auds.extend([os.path.join(video_folder[1], npy_file) for npy_file in npy_files])
            filepaths_audio.append (auds)
            auds = [] 
       


        # remove the even frames from the filepaths
        filepaths_no_even = []
        for folder in filepaths_image:
            filepaths_no_even.append([filepath for filepath in folder if get_index(filepath,self.dataset) % 2 != 0])
            
       
        filepaths_image = filepaths_no_even
        Sequence = []
       
        if self.Modality == 'Image':
            for i in range (len(filepaths_image)):
                num_frames = len(filepaths_image[i])

                for idx in range (self.frames_number):
                        
                    start_idx = idx # The index of the first frame of the video
                    end_idx = idx + (self.frames_number-1) # The index of the last frame of the video
                    
                    if end_idx >= num_frames:
                        break
                    
                    sequence_filepaths = filepaths_image[i][start_idx:end_idx + 1]
                    
                    Sequence.append(sequence_filepaths)
                
            return Sequence
        # Now we got the filepaths of the frames and the audio files, we will associate the audio to the frames
        # In a way that the first 33 % of the frames have the first audio file, the next 33% have the second audio file and the last 33% have the third audio file
        
        if self.Modality == 'Audio':
            for i in range (len(filepaths_audio)):
                num_frames = len(filepaths_audio[i])

                for idx in range (num_frames):
                        
                    start_idx = idx # The index of the first frame of the video
                    end_idx = idx + (self.aud_number-1) # The index of the last frame of the video
                    
                    if end_idx >= num_frames:
                        break
                    
                    sequence_filepaths = filepaths_audio[i][start_idx:end_idx + 1]
                    
                    Sequence.append(sequence_filepaths)
                    return Sequence
            
        if self.Modality == 'Mix':
        
            for i in range (len(filepaths_image)): # For each video
                num_frames = len(filepaths_image[i]) # Get the number of frames
                num_aud = len(filepaths_audio[i]) # Get the number of audio files
                
                
                frames_per_aud = int(num_frames/num_aud) # Get the number of frames per audio file
                
                for j in range (num_aud): # For each audio file
                    start_idx = j*frames_per_aud # The index of the first frame of the video
                    end_idx = (j+1)*frames_per_aud # The index of the last frame of the video
                    
                    if j == num_aud-1: # If it is the last audio file, we add the rest of the frames
                        end_idx = num_frames
                        
                    # we want a list of pairs, where list[i][0] is the filepath of the frame and list[i][1] is the filepath of the audio file
                    
                    for k in range (start_idx, end_idx):
                        frames_video.append([filepaths_image[i][k], filepaths_audio[i][j]])
                        
                filepaths_mix.append(frames_video)
                frames_video = []
                
            

            
            # For each video, get 32 frames sequences until the end of the video. 
            for i in range (len(filepaths_mix)):
                num_frames = len(filepaths_mix[i])

                for idx in range (self.frames_number):
                        
                    start_idx = idx # The index of the first frame of the video
                    end_idx = idx + (self.frames_number-1) # The index of the last frame of the video
                    
                    if end_idx >= num_frames:
                        break
                    
                    sequence_filepaths = filepaths_mix[i][start_idx:end_idx + 1]
                    
                    Sequence.append(sequence_filepaths)
        
        # if we put the PCA transform here, it will be applied in the data when the dataset construction, securing that the data is transformed only 1 time before the training
            if self.PCA is not None:
                        
                Data_Sequence = []
                
                for sequence_filepaths in tqdm(Sequence):
                    
                    data_sequence_image = []
                    for i in range (len(sequence_filepaths)):
                        filepath_image = sequence_filepaths[i][0] # Get the filepath of the embedding
                        filepath_audio = sequence_filepaths[i][1] # Get the filepath of the audio file
                            
                        emotion = get_emotion(filepath_image,self.dataset) # Get the emotion of the video
                        data_image = np.load(filepath_image) # Load the data
                        data_audio = np.load(filepath_audio)
                        if self.transform: # apply the transform if it exists
                            data_image = self.transform(data_image)
                            data_audio = self.transform(data_audio)
                
                    
                
                        data_image = data_image.reshape(1,self.image_shape)
                        data_audio = data_audio.reshape(1,self.audio_shape)    
                        # to tensor
                        data_image= torch.tensor(data_image)
                        data_audio = torch.tensor(data_audio)
                        pair = torch.cat((data_image,data_audio),dim=1)
                        data = self.PCA.transform(pair.numpy())
                        
                        data_sequence_image.append(data)
                        
                        
                                
                    data_sequence_image = np.array(data_sequence_image)
                    data_sequence_image = data_sequence_image.reshape(self.frames_number,self.PCA.n_components_)
                    Data_Sequence.append([data_sequence_image,int(emotion)])
                
                return Data_Sequence
                

                
            return Sequence
    

                    

    def __len__(self): # Returns the number of videos
        return len(self.filepaths)

    def __getitem__(self, idx): # Returns the data and the label of the video in the index idx
        # the data is a list of 32 elements. Each elements have 2 np.arrays, the embedding of the image and the embedding of the audio
        
        
        #since the data is already preprocessed, we just need to load it and return it
        if self.PCA is not None:
            sequence,emotion = self.filepaths[idx]
            return torch.tensor(sequence),torch.tensor(int(emotion))
               
        if self.PCA is None:
            
            
            sequence_filepaths = self.filepaths[idx] # Get the filepaths of the embeddings of the video in the index idx
            data_sequence_image = []
            data_sequence_audio = []
            
            emotion = None
        
        if self.Modality == 'Image':
            sequence_filepaths = self.filepaths[idx] # Get the filepaths of the embeddings of the video in the index idx
            data_sequence = []
            
            emotion = None
                    
            for i in range (len(sequence_filepaths)):
                
                filepath = sequence_filepaths[i] # Get the filepath of the embedding
                
                    
                emotion = get_emotion(filepath,self.dataset) # Get the emotion of the video
                
                data = np.load(filepath) # Load the data
                
                if self.transform: # apply the transform if it exists
                    data = self.transform(data)
                    
                data = np.squeeze(data, axis=0) # The dimension of the data is 1,2,2,512. So we remove that first dimension to make it 2,2,512
                data_sequence.append(data)
            
            data_sequence = np.array(data_sequence)
            
            return torch.from_numpy(data_sequence), torch.tensor(int(emotion)) 
        
        if self.Modality == 'Audio':
            sequence_filepaths = self.filepaths[idx] # Get the filepaths of the embeddings of the video in the index idx
            data_sequence = []
            
            emotion = None
                    
            for i in range (len(sequence_filepaths)):
                
                filepath = sequence_filepaths[i] # Get the filepath of the embedding
                
                    
                emotion = get_emotion(filepath,self.dataset) # Get the emotion of the video
                data = np.load(filepath) # Load the data
                
                if self.transform: # apply the transform if it exists
                    data = self.transform(data)
                    
                data = np.squeeze(data, axis=0) # The dimension of the data is 1,2,2,512. So we remove that first dimension to make it 2,2,512
                data_sequence.append(data)
            
            data_sequence = np.array(data_sequence)
            
            return torch.from_numpy(data_sequence), torch.tensor(int(emotion)) 
        if self.Modality == 'Mix':
            
            for i in range (len(sequence_filepaths)):
                
                filepath_image = sequence_filepaths[i][0] # Get the filepath of the embedding
                filepath_audio = sequence_filepaths[i][1] # Get the filepath of the audio file
                    
                emotion = get_emotion(filepath_image,self.dataset) # Get the emotion of the video
                data_image = np.load(filepath_image) # Load the data
                data_audio = np.load(filepath_audio)
                if self.transform: # apply the transform if it exists
                    data_image = self.transform(data_image)
                    data_audio = self.transform(data_audio)
                    
                data_image = np.squeeze(data_image, axis=0) # The dimension of the data is 1,2,2,512. So we remove that first dimension to make it 2,2,512
                data_audio = np.squeeze(data_audio, axis=0)
                
                data_sequence_image.append(data_image)
                data_sequence_audio.append(data_audio)
                
            
            data_sequence_image = np.array(data_sequence_image)
            data_sequence_image = data_sequence_image.reshape(self.frames_number,self.image_shape)
            data_sequence_audio = np.array(data_sequence_audio)
            return torch.tensor(data_sequence_image),torch.tensor(data_sequence_audio), torch.tensor(int(emotion))    
        
        
    # function to discard the folders with no video or audio, or with less than 32 frames or with less than 3 audio files
    
def discard(videos,audios):
    mal = []
    
    for video in videos:
        
        folder = video.split('/')[-1]
        
        # search for the folder in the audio list
        for audio in audios:
            if folder in audio:
                pair = [video,audio]
                
        if len(os.listdir(pair[0])) < 32:
            
            mal.append(pair)  
            
        
                
    for pair in mal:
        videos.remove(pair[0])
        audios.remove(pair[1])
        
    for audio in audios:
        folder = audio.split('/')[-1]
        
        
        for video in videos:
            if folder in video:
                pair = [video,audio]
                
                if len(os.listdir(pair[1])) != 3:
                    videos.remove(pair[0])
                    audios.remove(pair[1])
                                        

    # If I dont make the loop twice, it doesnt work. Dont know why
    
    for audio in audios:
        folder = audio.split('/')[-1]
        if not any(folder in video for video in videos):
            audios.remove(audio)
            
    for audio in audios:
        folder = audio.split('/')[-1]
        if not any(folder in video for video in videos):
            audios.remove(audio)

        
            
            
    print (len(videos))
    print (len(audios))
    assert len(videos) == len(audios)
    
    videos.sort()
    audios.sort()
    return videos,audios