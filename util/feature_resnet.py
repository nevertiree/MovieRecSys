import numpy as np
from PIL import Image

import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class Feature_Extract_Poster():
    """ use this module to get the feature of pics with the special imdb_id """

    def __init__(self, poster_path="posters/"):
        self.model = models.resnet18(pretrained=True).eval()
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.poster_path = poster_path  # the path of the input pic
        self.target_img_size = 256  # the size of VGG input pic
        self.svd_size = (512, 256)  # the multiply must equal to 128*64
        self.pool_3d = torch.nn.MaxPool3d((4, 2, 2), stride=4)  # 3-d pool

    def extract_feature(self, imdb_id):
        """
        use the vgg model to extract from the posters path
        :par
        |imdb_id:str, the imdb id you want to get
        """
        try:
            if not imdb_id.endswith("jpg"):
                img_path = os.path.join(self.poster_path, imdb_id+".jpg")
            else:
                img_path = os.path.join(self.poster_path, imdb_id)
            print(img_path)
            img = Image.open(img_path)  # get the posters
        except FileNotFoundError:
            return None  # there is no  movie according to this imdbid

        """ image transform """
        tran = transforms.Compose([
            transforms.Resize((self.target_img_size, self.target_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tran(img).unsqueeze_(dim=0)  # get the four dim tensor

        """ deal with model output """
        result = self.pool_3d(self.model(img)) 
        result = torch.flatten(result)  # get the flatten feature vector
        result = result.data.cpu().numpy()  # return the cpu data
        
        result_str = ''
        for data_ in result:  # concat the data
           result_str += str(round(data_, 4)) + '|'
        #print(result_str[:-1], 'the size is')
        #print(result.size, 'the size is')

        return result_str[:-1]

    def correlation_cal_matrix(self):
        """
        cal the correlation between two movies
        arg:
        |imdb_id: str, the imdb of movie you want to cal
        :not used for now, but reserve for furture
        """
        similarity_deep = self.matrix_vgg_features.dot(
            self.matrix_vgg_features.T)  # matrix * matrix.T
        norms = np.array([np.sqrt(np.diagonal(similarity_deep))
                          ])  # get the norm values
        similarity_deep = similarity_deep / norms / norms.T  # get the similarity of all movies
        return similarity_deep

    def correlation_cal_single(self, feature_1, feature_2):
        """
        cal the correlation between two movies
        arg:
        |feature_n: array with shape:[1, 25508], the result of the 30th layer in VGG16 
        """
        similarity_deep = feature_1.dot(feature_2.T)  # matrix * matrix.T
        norms = np.array([
            np.sqrt(np.linalg.norm(feature_1) * np.linalg.norm(feature_2))
        ])  # get the norm values
        similarity_deep = similarity_deep / norms / norms.T  # get the similarity of all movies
        return similarity_deep


if __name__ == '__main__':
    fep = Feature_Extract_Poster("../../data/image")  # for test

    """ get two feature to test the correlation func """
    feature_1 = fep.extract_feature('1')
    feature_2 = fep.extract_feature('2')
    feature_3 = fep.extract_feature('3')
