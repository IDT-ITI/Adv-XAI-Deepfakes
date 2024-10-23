import sys
import numpy as np
from explanation.evaluation.adversarial_image_generation import AdversarialImageGeneration
sys.path.append('../../src')
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torch
from torchvision.transforms import v2
from lime import lime_image
from model.frame import FrameModel
import torch.nn.functional as F

def inverseNormalization(img):
    invTrans = v2.Compose([v2.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                           v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    return invTrans(img)

def explain(image,transforms,label,model,adv_mask=False,custom_seg=None,visualize=True):

    #The function used by the explainer to predict the model's output
    def predict(images):
        model.eval()
        batch = torch.stack(tuple(transforms(i) for i in images), dim=0)
        logits = model(batch.to(model.device))
        if(logits.shape[1]>1):
            probs = F.softmax(logits, dim=1)
        else:
            probs=logits
        return probs.detach().cpu().numpy()

    #Create an explainer
    explainer = lime_image.LimeImageExplainer()

    if(adv_mask):
        #Create the AdversarialImageGeneration object
        evaluation = AdversarialImageGeneration(transforms(image), model, 0.001, 40, 80, 16 / 255, 1 / 255)
        #Generate the adversarial variant by attacking the whole image
        masker = evaluation.generateAdversarialImage(np.ones((image.shape[0],image.shape[0]),dtype=int))
        #Set the masker
        masker = inverseNormalization(masker)
        masker = masker.permute(1, 2, 0).numpy()
    else:
        #Otherwise use the default masker of LIME
        masker=None

    #Set the segmentation function if it exists and compute the explanation
    if(custom_seg==None):
        explanation = explainer.explain_instance(image, predict, hide_color=masker, num_samples=2000)
    else:
        explanation = explainer.explain_instance(image, predict, hide_color=masker, segmentation_fn=custom_seg, num_samples=2000)

    #Get the image and the explanation mask
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=2, hide_rest=False)

    #If selected visualize the result
    if(visualize):
        img_boundries = mark_boundaries(temp, mask)
        plt.imshow(img_boundries)
        plt.show()

    #Return the the explanation mask and segments and the weights of the linear regressor for each segment
    return [mask,explanation.segments,explanation.local_exp[label]]

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
    visualize_image = visualize_transforms(image).permute(1,2,0).numpy()
    #Select the explanation label
    label = 0

    #Call the explanation method
    explain(visualize_image, inference_transforms, label, model)
    explain(visualize_image, inference_transforms, label, model, adv_mask=True)