import numpy as np
from scipy.stats import pearsonr

class ExplanationMetrics():
    def __init__(self, model):
        self.model=model

    #Compute the comprehensiveness between original and adversarially attacked examples

    #x: Example with applied inference transformations
    #x_adv: Adversarially attacked example with applied inference transformations
    #label: The label to compute the metric
    #return: The difference between the scores of the original and the adversarially attacked example for label

    def disc(self, x, x_adv, label):
        score_original=self.model(x.unsqueeze(0).to(self.model.device)).cpu().detach().numpy()[0][label]
        score_adv=self.model(x_adv.unsqueeze(0).to(self.model.device)).cpu().detach().numpy()[0][label]
        return score_original-score_adv