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
        :par
        |imdb_id:str, the imdb id you want to get
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
    fep = Feature_Extract_Poster()  # for test
    """ get two feature to test the correlation func """
    feature_1 = fep.extract_feature('1')
    feature_3 = fep.extract_feature('3')
    correlation = fep.correlation_cal_single(feature_1, feature_3)
