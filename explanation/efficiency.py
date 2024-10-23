import os
import time
import torch
import contextlib
from torchvision.transforms import v2
import sys
sys.path.append('../src')
sys.path.append('./methods')
from model.frame import FrameModel
import numpy as np
from data.datasets import DeepfakeDataset
from methods.lime_xai import explain as LIME
from methods.shap_xai import explain as SHAP
from methods.sobol_xai import explain as SOBOL
from methods.rise_xai import explain as RISE
from evaluation.generate_ff_test_data import getFFPath



# Function to suppress output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


#Add inference counter into the ff_attribution model
def addInferenceCounter(model):
    from types import MethodType

    #Modify the model's forward to advance the counter each time it is called
    def forward(self, x):
        self.inference_counter+=1
        return self.act(self.model(x))  # type: ignore

    #Create a function to reset the model's counter
    def reset_counter(self):
        self.inference_counter=0

    #Create a function to get the model's counter
    def get_counter(self):
        return self.inference_counter

    #Set the functions and initialize the counter
    model.reset_counter = MethodType(reset_counter, model)
    model.get_counter = MethodType(get_counter, model)
    model.forward = MethodType(forward, model)
    model.reset_counter()
    #Return the modified model
    return model


#Load the model
rs_size = 224
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt",map_location='cuda').eval()
task = "multiclass"

#Create the transforms for inference and visualization purposes
interpolation = 3
inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(rs_size, interpolation=interpolation, antialias=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
visualize_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(rs_size, interpolation=interpolation, antialias=False),
    v2.ToDtype(torch.float32, scale=True),
])

#Create the test examples and load the dataset
ds_path = getFFPath("../data/csvs/ff_test.csv")

#Dataset with inference transformations
target_transforms = lambda x: torch.tensor(x, dtype=torch.float32)
ds = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=inference_transforms,
    target_transforms=target_transforms,
    task=task
)
#Dataset with visualization transformations
ds_visualize = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=visualize_transforms,
    target_transforms=target_transforms,
    task=task
)

#Modify the model to add an inference counter
model=addInferenceCounter(model)

lime_time=[]; lime_inf=[]; lime_adv_time=[]; lime_adv_inf=[]
shap_time=[]; shap_inf=[]; shap_adv_time=[]; shap_adv_inf=[]
sobol_time=[]; sobol_inf=[]; sobol_adv_time=[]; sobol_adv_inf=[]
rise_time=[]; rise_inf=[]; rise_adv_time=[]; rise_adv_inf=[]

start_index = 0
#Load the saved results from the file if it exists, print the current statistics and set the correct index
if(os.path.isfile("./results/efficiency.npy")):
    scores = np.load("./results/efficiency.npy")
    lime_time = list(scores[0]); lime_inf = list(scores[1].astype(int)); lime_adv_time = list(scores[2]); lime_adv_inf = list(scores[3].astype(int))
    shap_time = list(scores[4]);shap_inf = list(scores[5].astype(int));shap_adv_time = list(scores[6]);shap_adv_inf = list(scores[7].astype(int))
    sobol_time = list(scores[8]);sobol_inf = list(scores[9].astype(int));sobol_adv_time = list(scores[10]);sobol_adv_inf = list(scores[11].astype(int))
    rise_time = list(scores[12]);rise_inf = list(scores[13].astype(int));rise_adv_time = list(scores[14]);rise_adv_inf = list(scores[15].astype(int))
    start_index = len(lime_time)

    #Compute the mean time and number of inferences for each method for the current number of images tested
    lime_mean_time = round(np.mean(lime_time), 1)
    lime_mean_inf = int(np.mean(lime_inf))
    lime_adv_mean_time = round(np.mean(lime_adv_time), 1)
    lime_adv_mean_inf = int(np.mean(lime_adv_inf))
    shap_mean_time = round(np.mean(shap_time), 1)
    shap_mean_inf = int(np.mean(shap_inf))
    shap_adv_mean_time = round(np.mean(shap_adv_time), 1)
    shap_adv_mean_inf = int(np.mean(shap_adv_inf))
    sobol_mean_time = round(np.mean(sobol_time), 1)
    sobol_mean_inf = int(np.mean(sobol_inf))
    sobol_adv_mean_time = round(np.mean(sobol_adv_time), 1)
    sobol_adv_mean_inf = int(np.mean(sobol_adv_inf))
    rise_mean_time = round(np.mean(rise_time), 1)
    rise_mean_inf = int(np.mean(rise_inf))
    rise_adv_mean_time = round(np.mean(rise_adv_time), 1)
    rise_adv_mean_inf = int(np.mean(rise_adv_inf))

    #Print the number of images tested
    print(start_index)
    #Print the current mean statistics
    print("Time Original: LIME " + str(lime_mean_time) + "s, SHAP " + str(shap_mean_time) + "s, SOBOL " + str(sobol_mean_time) + "s, RISE " + str(rise_mean_time) + "s")
    print("Time Adversarial: LIME " + str(lime_adv_mean_time) + "s, SHAP " + str(shap_adv_mean_time) + "s, SOBOL " + str(sobol_adv_mean_time) + "s, RISE " + str(rise_adv_mean_time) + "s")
    print("Inferences Original: LIME " + str(lime_mean_inf) + ", SHAP " + str(shap_mean_inf) + ", SOBOL " + str(sobol_mean_inf) + ", RISE " + str(rise_mean_inf))
    print("Inferences Adversarial: LIME " + str(lime_adv_mean_inf) + ", SHAP " + str(shap_adv_mean_inf) + ", SOBOL " + str(sobol_adv_mean_inf) + ", RISE " + str(rise_adv_mean_inf))

#For every image of the dataset
for idx in range(start_index, len(ds.df)):

    #Print the number of images tested
    print(idx+1)

    #Get the images
    inference_image, label_real = ds[idx]
    visualize_image, _ = ds_visualize[idx]

    #Compute the inference scores
    with torch.no_grad():
        frame = inference_image.to(model.device)
        output = model(frame.unsqueeze(0))

    output = output.cpu().reshape(-1, ).numpy()
    explanation_label_index = np.argmax(output)

    #Call the corresponding explanation method to calculate the explanation and get the statistics
    with suppress_output():
        model.reset_counter()
        start_time = time.time()
        LIME(visualize_image.permute(1, 2, 0).numpy(), inference_transforms, explanation_label_index, model, visualize=False)
        lime_time.append(time.time()-start_time)
        lime_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        LIME(visualize_image.permute(1, 2, 0).numpy(), inference_transforms, explanation_label_index, model, adv_mask=True, visualize=False)
        lime_adv_time.append(time.time() - start_time)
        lime_adv_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        SHAP(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model, visualize=False)
        shap_time.append(time.time() - start_time)
        shap_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        SHAP(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model, adv_mask=True, visualize=False)
        shap_adv_time.append(time.time() - start_time)
        shap_adv_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        SOBOL(inference_image, visualize_image, explanation_label_index, model, visualize=False)
        sobol_time.append(time.time() - start_time)
        sobol_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        SOBOL(inference_image, visualize_image, explanation_label_index, model, adv_mask=True, visualize=False)
        sobol_adv_time.append(time.time() - start_time)
        sobol_adv_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        RISE(inference_image, visualize_image.unsqueeze(0), explanation_label_index, model, visualize=False)
        rise_time.append(time.time() - start_time)
        rise_inf.append(model.get_counter())

        model.reset_counter()
        start_time = time.time()
        RISE(inference_image, visualize_image.unsqueeze(0), explanation_label_index, model, adv_mask=True, visualize=False)
        rise_adv_time.append(time.time() - start_time)
        rise_adv_inf.append(model.get_counter())

    #Compute the mean time and number of inferences for each method for the current number of images tested
    lime_mean_time=round(np.mean(lime_time),1)
    lime_mean_inf=int(np.mean(lime_inf))
    lime_adv_mean_time = round(np.mean(lime_adv_time),1)
    lime_adv_mean_inf = int(np.mean(lime_adv_inf))
    shap_mean_time=round(np.mean(shap_time),1)
    shap_mean_inf=int(np.mean(shap_inf))
    shap_adv_mean_time = round(np.mean(shap_adv_time),1)
    shap_adv_mean_inf = int(np.mean(shap_adv_inf))
    sobol_mean_time=round(np.mean(sobol_time),1)
    sobol_mean_inf=int(np.mean(sobol_inf))
    sobol_adv_mean_time = round(np.mean(sobol_adv_time),1)
    sobol_adv_mean_inf = int(np.mean(sobol_adv_inf))
    rise_mean_time=round(np.mean(rise_time),1)
    rise_mean_inf=int(np.mean(rise_inf))
    rise_adv_mean_time = round(np.mean(rise_adv_time),1)
    rise_adv_mean_inf = int(np.mean(rise_adv_inf))

    #Print the current mean of statistics
    print("Time Original: LIME "+str(lime_mean_time)+"s, SHAP "+str(shap_mean_time)+"s, SOBOL "+str(sobol_mean_time)+"s, RISE "+str(rise_mean_time)+"s")
    print("Time Adversarial: LIME "+str(lime_adv_mean_time)+"s, SHAP "+str(shap_adv_mean_time)+"s, SOBOL "+str(sobol_adv_mean_time)+"s, RISE "+str(rise_adv_mean_time)+"s")
    print("Inferences Original: LIME "+str(lime_mean_inf)+", SHAP "+str(shap_mean_inf)+", SOBOL "+str(sobol_mean_inf)+", RISE "+str(rise_mean_inf))
    print("Inferences Adversarial: LIME "+str(lime_adv_mean_inf)+", SHAP "+str(shap_adv_mean_inf)+", SOBOL "+str(sobol_adv_mean_inf)+", RISE "+str(rise_adv_mean_inf))

    #Save the results in a file to avoid having to restart the whole procedure in case of the program crashing or something else going wrong
    scores=np.array((lime_time,lime_inf,lime_adv_time,lime_adv_inf,shap_time,shap_inf,shap_adv_time,shap_adv_inf,
                     sobol_time,sobol_inf,sobol_adv_time,sobol_adv_inf,rise_time,rise_inf,rise_adv_time,rise_adv_inf))
    np.save("./results/efficiency.npy", scores)