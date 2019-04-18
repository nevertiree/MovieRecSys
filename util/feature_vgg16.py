import numpy as np
from PIL import Image

import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms


class Feature_Extract_Poster():
    """ use this module to get the feature of pics with the special imdb_id """

    def __init__(self):
        self.model = models.vgg16(pretrained=True).features.eval()
        self.poster_path = 'posters/'  # the path of the input pic
        self.target_img_size = 224  # the size of VGG input pic

    def extract_feature(self, imdb_id):
        """
        use the vgg model to extract from the posters path
        """
        try:
            img = Image.open(self.poster_path + imdb_id +
                             '.jpg')  # get the posters
        except FileNotFoundError:
            return None  # there is no  movie according to this imdbid
        tran = transforms.Compose([
            transforms.Resize((self.target_img_size, self.target_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tran(img).unsqueeze_(dim=0)  # get the four dim tensor
        result = torch.flatten(
            self.model(img))  # get the flatten feature vector
        result = result.data.cpu().numpy()  # return the cpu data
        return result


if __name__ == '__main__':
    fep = Feature_Extract_Poster()
    print(fep.extract_feature('1'))
