import cv2
import os
import shutil
import PIL as pil
import numpy as np
import math
import wave 
import matplotlib.pyplot as plt
import torch
import datetime
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
# For the face embeddings

#from keras_vggface.vggface import VGGFace


# https://github.com/rcmalli/keras-vggface

#vggface = VGGFace(model='vgg16', include_top=False, input_shape=(64, 64, 3),weights='vggface')



# For the audio embeddings


# from pyannote.audio import Inference, Model
# token_acceso = "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"

# voice_model = Model.from_pretrained('pyannote/embedding', use_auth_token=token_acceso)
# voice_model.eval()
# inference = Inference(voice_model, window='whole')





# Functions


# Extract the frames of the video
def extract_frames (path_input,path_output= None):

    '''
    Input:
    - path_input = path of the video
    - path_output = path where the frames will be saved (Optional)
    Output:
    - frames of the video saved in the path_output
    - frames = list of the frames of the video
    '''
    
    video = cv2.VideoCapture(path_input)
    frameno = 0
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frameno += 1
        
        frames.append(frame)
        if path_output != None:
            number = '{:03}'.format(frameno)
            cv2.imwrite(path_output+str(number)+'.jpg', frame)
        
    video.release()
    return frames
    

# Crop the face of the image
def cut_faces (path_input,path_output = None):
    '''
    Input: 
    - path_input = path of the image
    - path_output = path where the image will be saved. (Optional)
    Output: 
    
    image with the face cutted
    
    
    '''
    
    image = cv2.imread(path_input)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(image_gray, 1.3, 5)

    roi_color = None
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_color = image[y:y + h, x:x + w]
        # cut the border of the image
        roi_color = image[y+2:y + h-2, x+2:x + w-2]    

        if roi_color is None:
            print ("No face detected")
        if roi_color is not None:
            if path_output != None:
                cv2.imwrite(path_output, roi_color)
    
    return roi_color





def split_audio (x,output_folder):
# Create the output folder if it doesn't exist
    '''

    Input:
    - x = path of the audio
    - output_folder = path where the audio will be saved

    Output:
    - output_files = list of the three audios generated

    '''

    #print (x)
    #print (output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Open the input audio file
    with wave.open(x, 'rb') as wav_file:
        # Get the audio file's properties
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        # Calculate the duration of one-third of the audio
        one_third_duration = math.ceil(num_frames / 3)



        name = x.split('/')[-1]
        name_base = name.split('.wav')[0]
        
        # Create three output audio files
        output_files = []
        
        
        for i in range(3):
            name = name_base + '_' + str(i+1) + '.wav'
            output_file = os.path.join(output_folder, name)
            output_files.append(output_file)
            with wave.open(output_file, 'wb') as out_wav_file:
                out_wav_file.setnchannels(num_channels)
                out_wav_file.setsampwidth(sample_width)
                out_wav_file.setframerate(frame_rate)

                # Set the number of frames to write based on the one-third duration
                num_frames_to_write = min(one_third_duration, num_frames - (i * one_third_duration))
                frames = wav_file.readframes(num_frames_to_write)
                out_wav_file.writeframes(frames)

    return output_files

# Generate the embeddings of the audio
def embed_audio (x):
    
    '''
    Input:
    - x = path of the audio
    Output:
    - embedding = embedding of the audio with X-vectors
    '''
    embedding = inference(x)
    embedding = embedding/np.linalg.norm(embedding,2)
    embedding = embedding.reshape(1,-1)
    return embedding

# Generate the embeddings of the face
def embed_face (path_input):
    
    '''
    Input:
    - path_input = path of the image
    Output:
    - embed = embedding of the face with VGGFace
    '''
    
    img = pil.Image.open(path_input)
    img = img.resize((64,64))
    img = np.array(img)
    img = img.reshape((1,64,64,3))
    img = img.astype('float64')
    img = img/255
    embed = vggface.predict(img)
    embed = np.array(embed)
    return embed


#  Show the set of 32 frames of the video
def show_images_32(images,title=None):
    
    '''
    Input:
    - images = list of the frames of the video
    - title = title of the plot (Optional)
    Output:
    - Plot of the 32 frames of the video    
    '''
    
    fig = plt.figure(figsize=(12, 7))
    columns = 8
    rows = 4
    for i in range(1, columns * rows + 1):
        img = plt.imread(images[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')

    
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    if title != None:
        plt.title(title)
        
    plt.show()

    
    
# predict emotion of the set of frames
def predict_images(model,frames):
    
    '''
    Input:
    - model = model to predict the emotion
    - frames = list of the frames of the video
    Output:
    - emotion = emotion predicted by the model
    '''
    
    embeddings = []
    for frame in frames:
        embedding = embed_face(frame)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    embeddings = torch.from_numpy(embeddings)
    embeddings = embeddings.view(1,32,2,2,512)
    outputs = model(embeddings)
    #_, predicted = torch.max(outputs.data, 1)
    #emotion = predicted.item()
    return outputs

        


def simple_get_accuracy(set,model,device):
    
    from  sklearn.metrics import classification_report 
    
    correct = 0
    total = 0
    
    annotations = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(set):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            annotations.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum()
            
    print(classification_report(annotations,predictions,digits=4))
    
    return 100 * correct / total



def simple_check_accuracy(train_set,val_set,model,device):
    train_accuracy = simple_get_accuracy(train_set,model,device)
            
    print('Accuracy of the network on the train images: %.2f %%' % (train_accuracy))
    
    val_accuracy = simple_get_accuracy(val_set,model,device)
            
    print('Accuracy of the network on the val images: %.2f %%' % (val_accuracy))


def Mix_get_accuracy(set,model,device):
    
    from  sklearn.metrics import classification_report 
    
    correct = 0
    total = 0

    with torch.no_grad():
        annotations = []
        predictions = []
        
        for data in tqdm(set):
                
            images,aud, labels = data
            images = images.to(device)
            aud = aud.to(device)
            labels = labels.to(device)
            outputs = model(images,aud)
            _, predicted = torch.max(outputs.data, 1)
            
            
            annotations.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
    print(classification_report(annotations,predictions,digits=4))
    
    return 100 * correct / total

def Mix_check_accuracy(train_set,val_set,model,device):

    train_accuracy = Mix_get_accuracy(train_set,model,device)
    
    print('Accuracy of the network on the train images: %.2f %%' % (train_accuracy))
    
    val_accuracy = Mix_get_accuracy(val_set,model,device)
    
    print('Accuracy of the network on the val images: %.2f %%' % (val_accuracy))
    
    
    
    





def simple_training_loop(n_epochs,optimizer,loss_fcn,val_dataset,train_dataset,model,device,name_save = None):
        best_val_ac = float('inf')
        for epoch in range(1,n_epochs+1):
            loss_train = 0.0
            

            # train_dataset
            for i , (imgs,labels) in enumerate(tqdm(train_dataset)):
 
                imgs = imgs.to(device)
                
                labels = labels.to(device)
                outputs = model(imgs)
                loss = loss_fcn(outputs,labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_train += loss.item()
            
            

            loss_train /= len(train_dataset)
            
            
            # val_dataset
            loss_val = 0.0
            for imgs,labels in val_dataset:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = loss_fcn(outputs,labels)
                loss_val += loss.item()
                
            loss_val /= len(val_dataset)
            
            
            
               
            if name_save != None:
                if loss_val < best_val_ac:
                    best_val_ac = loss_val
                torch.save(model.state_dict(), name_save)
                
            # print the results with 2 decimals
            print('Epoch: %d/%d, Train loss: %.2f, Val loss: %.2f' % (epoch,n_epochs,loss_train,loss_val))
    
        
        
def Mix_training_loop(n_epochs,optimizer,loss_fcn,val_dataset,train_dataset,model,device,name_save = None):
        best_val_ac = float('inf')
        for epoch in range(1,n_epochs+1):
            loss_train = 0.0
            

            # train_dataset
            for i , (imgs,aud,labels) in enumerate(tqdm(train_dataset)):
 
    
                imgs = imgs.to(device)
                aud = aud.to(device)
                
                labels = labels.to(device)
                outputs = model(imgs,aud)
                loss = loss_fcn(outputs,labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_train += loss.item()
            
            

            loss_train /= len(train_dataset)
            
            
            # val_dataset
            loss_val = 0.0
            for imgs,aud,labels in val_dataset:
                imgs = imgs.to(device)
                aud = aud.to(device)
    
                labels = labels.to(device)
                outputs = model(imgs,aud)
                loss = loss_fcn(outputs,labels)
                loss_val += loss.item()
                
            loss_val /= len(val_dataset)
            
            
            
            if name_save != None:
                if loss_val < best_val_ac:
                    best_val_ac = loss_val
                    torch.save(model.state_dict(), name_save)
                    
            print('Epoch: %d/%d, Train loss: %.2f, Val loss: %.2f' % (epoch,n_epochs,loss_train,loss_val))
    
    

def simple_plot_confusion_matrix(data_loader, model, device,title = 'Confusion Matrix',class_names = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']):
    """
    Plots the confusion matrix for the given model and data.
    """
    
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # make the confusion matrix with percentage values instead of raw counts 
    # (i.e. each row sums to 1)
    # T
    
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    
    sorted_indices = np.argsort(class_names)
    sorted_class_names = np.array(class_names)[sorted_indices]
    confusion_mat = confusion_mat[sorted_indices, :][:, sorted_indices]
    confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)  # Adjust the font size for labels
    # Create a heatmap using Seaborn
    # Change label orientation 

    ax = sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Blues",
                 xticklabels=sorted_class_names, yticklabels=sorted_class_names,cbar = False)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')

    
    plt.xlabel('Predicted Labels')
    # labels in the vertical axis

    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()
   
    
    

def mix_plot_confusion_matrix(data_loader, model, device,title = 'Confusion Matrix',class_names = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for img,aud, labels in data_loader:
            img = img.to(device)
            aud = aud.to(device)
            labels = labels.to(device)

            outputs = model(img,aud)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

# PLot a confusion matrix that have len(labels) classes
    

        
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    
    sorted_indices = np.argsort(class_names)
    sorted_class_names = np.array(class_names)[sorted_indices]
    confusion_mat = confusion_mat[sorted_indices, :][:, sorted_indices]
    confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    # plot the confusion mat in alphanumerical order

            
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)  # Adjust the font size for labels
    
    # Create a heatmap using Seaborn
    # Change label orientation 

    ax = sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sorted_class_names, yticklabels=sorted_class_names,cbar = False)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')

    
    plt.xlabel('Predicted Labels')
    # labels in the vertical axis

    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()
    
def AE_training_loop(epochs,optimizer,loss_fcn,val_data,train_data,model,device,path_models):
    
    best_loss = np.inf
    for epoch in range(1,epochs+1):
        loss_train = 0.0
        loss_val = 0.0
        for image,audio,_ in tqdm(train_data):
            image = image.to(device)
            audio = audio.to(device)
            optimizer.zero_grad()
            outputs = model(image,audio)
            loss = loss_fcn(outputs,torch.cat((image,audio),dim=2))
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        loss_train /= len(train_data)
            
        for image,audio,_ in val_data:
            image = image.to(device)
            audio = audio.to(device)
            outputs = model(image,audio)
            loss = loss_fcn(outputs,torch.cat((image,audio),dim=2))
            loss_val += loss.item()
            
        loss_val /= len(val_data)
            
        print("Epoch {}, Training loss {}, Validation loss {}".format(epoch, loss_train, loss_val))
        
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), path_models+'best_model_AE.pth')