import os
import pandas as pd

#Create the csv file with the paths of the examples

#ds_path: Path of the original test csv
#samples: int number corresponding to the number of frames to sample for each video
#return: Path of the created csv

def getFFPath(ds_path,samples=10):
    #Create the name of the new csv
    save_path = ds_path.split('/')
    save_path[-1] = save_path[-1].split('.')[0] + '_final.csv'
    save_path = '/'.join(save_path)

    #Return the path if the csv already exists
    if(os.path.isfile(save_path)):
        return save_path

    #Read the original test csv
    df = pd.read_csv(ds_path, sep=' ')

    #Create a dataframe with only the deepfake examples
    df = df[~df['relative_path'].str.startswith('o')]

    #Create a dataframe with all of the videos
    unique_videos=lambda x: '/'.join((x.split('/')[:-1]))
    df_unique=df['relative_path'].apply(unique_videos).drop_duplicates(ignore_index=True)

    #Produce the dataframe with the sampled video frames
    df_final=pd.DataFrame()
    #For every video
    for video in df_unique:
        #Count the number of video frames
        count=df['relative_path'].str.contains(video).sum()
        #Compute the step based on the number of frames and the sampling rate (minimum step=1)
        step=max(round(count/samples),1)
        #Compute the indexes (keep only the number of samples we want)
        rows_index=[*range(0,count,step)][:samples]
        #If the number of resulting samples is smaller than desired
        if(len(rows_index)<samples and count>=samples):
            #Sample more starting from the beginning of the indexes and shifting them by one
            for i in range(samples-len(rows_index)):
                rows_index.append(rows_index[i]+1)
            rows_index.sort()
        #Find the indexes of the sampled frames in the dataframe and concatenate them to the final dataframe
        indexes = df.index[df['relative_path'].str.contains(video)].tolist()
        df_final=pd.concat([df_final,df.iloc[[indexes[i] for i in rows_index]]])

    #Save the produced dataframe and return the path
    df_final.to_csv(save_path,sep=' ',index=False)
    return save_path