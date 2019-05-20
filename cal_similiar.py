# -*- coding: utf-8 -*-

import os
from util.feature_resnet import Feature_Extract_Poster
import numpy as np
import random


def str_to_array(feature_str: str, split="|"):
    return np.asarray([float(i) for i in feature_str.split(split)])


def cal_similar(target_id, feature_file="data/image_feature.txt"):
    similar_dict = {}

    target_feature = []
    with open(feature_file, "r", encoding="utf8") as f:
        for line in f:
            if line.split(",")[0] == target_id:
                target_feature = str_to_array(line.split(",")[1])
                break

    with open(feature_file, "r", encoding="utf8") as f:
        for line in f:
            if line.split(",")[0] != target_id:
                image_id = line.split(",")[0]
                image_feature = str_to_array(line.split(",")[1])

                similar = correlation_cal_single(target_feature, image_feature)
                similar_dict[image_id] = similar

    print(sorted(similar_dict.items(), key=lambda item: item[1], reverse=True)[:5])


def correlation_cal_single(feature_1, feature_2):
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
    return similarity_deep[0]


def save_feature():
    feature_extractor = Feature_Extract_Poster("data/image")

    poster_list = os.listdir("data/image")

    with open("data/image_feature.txt", "a+", encoding="utf8") as f:
        for poster_id in poster_list:
            poster_feature = feature_extractor.extract_feature(poster_id)
            f.write(poster_id + "," + poster_feature + "\n")
            print(poster_id)


if __name__ == '__main__':
    image_list = os.listdir("data/image")
    for _ in range(5):
        random_image = random.choice(image_list)
        print(random_image)
        cal_similar(random_image)
