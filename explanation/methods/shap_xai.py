import sys
import numpy as np
import torch
from explanation.evaluation.adversarial_image_generation import AdversarialImageGeneration
sys.path.append('../../src')
from model.frame import FrameModel
from torchvision.transforms import v2
from PIL import Image
import shap

def inverseNormalization(img):
    invTrans = v2.Compose([v2.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                           v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    return invTrans(img)

def explain(inference_image,visualize_image,label,model,adv_mask=False,visualize=True):

    #The function used by the explainer to predict the model's output
    def predict(img):
        img = torch.Tensor(img).permute(0, 3, 1, 2)
        img = img.to(model.device)
        output = model(img)
        return output

    batch_size = 50
    n_evals = 2000

    if (adv_mask):
        #Create the AdversarialImageGeneration object
        evaluation = AdversarialImageGeneration(inference_image, model, 0.001, 40, 80, 16 / 255, 1 / 255)
        #Generate the adversarial variant by attacking the whole image
        mask = evaluation.generateAdversarialImage(np.ones((inference_image.shape[1], inference_image.shape[2]), dtype=int))
        mask = mask.permute(1, 2, 0).numpy()
        #Define a masker that is used to mask out partitions of the input image with the adversarial variant
        masker = shap.maskers.Image(mask)
    else:
        #Define a masker that is used to blur out partitions of the input image
        masker = shap.maskers.Image("blur(128,128)", inference_image.permute(1, 2, 0).shape)

    #Create an explainer with the model and image masker
    explainer = shap.Explainer(predict, masker)

    #Compute the explanation
    shap_values = explainer(
        inference_image.permute(1, 2, 0).unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=[label]
    )
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

    #If selected visualize the result
    if(visualize):
        shap.image_plot(shap_values=shap_values.values,pixel_values=visualize_image)

    #Create a saliency map from the shap values
    heatmap = (shap_values.values[0]-shap_values.values[0].min()) / (shap_values.values[0].max()-shap_values.values[0].min())
    heatmap = np.mean(heatmap, axis=2)

    #Return the saliency map
    return heatmap


if __name__ == "__main__":
    #Load the model
    rs_size = 224
    model = FrameModel.load_from_checkpoint("../../model/checkpoint/ff_attribution.ckpt", map_location='cuda').eval()

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

    #Open the image
    image = Image.open('test.jpg')
    #Apply the transformations
    inference_image = inference_transforms(image)
    visualize_image = visualize_transforms(image).permute(1,2,0).numpy()
    #Select the explanation label
    label = 0

    #Call the explanation method
    explain(inference_image, visualize_image, label, model)
    explain(inference_image, visualize_image, label, model, adv_mask=True)



