import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from sklearn.model_selection import train_test_split

genres_list = [
    'Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance',
    'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi',
    'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir',
    '(no genres listed)'
]


def seperate_genres(genres_str, genres_list):
    return [
        1 / (genres_str.split("|").index(genre) + 1)
        if genre in genres_str.split("|") else 0 for genre in genres_list
    ]


class MovieLensDataset(Dataset):
    def __init__(self, data, root_path, transform=None):
        self.root_path = root_path
        # self.file_path = os.path.join(self.root_path, file_name)
        self.item_path = os.path.join(self.root_path, 'movies.csv')
        self.transform = transform

        self.data = data
        self.item = pd.read_csv(self.item_path)
        self.genres_list = genres_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        genres_str = self.data['genres'].iloc[idx]
        if self.transform is not None:
            movie_genres_feature = self.transform(genres_str, self.genres_list)
        movie_genres_feature = np.array(movie_genres_feature, dtype=np.float32)

        preference_list = self.data['user_preference_topk'].iloc[idx]
        preference_list = [
            int(idx)
            for idx in preference_list.strip('[').strip(']').split(',')
        ]
        preference_genres_feature_list = []

        for preference_idx in preference_list:
            preference_genres_str = self.item['genres'][
                self.item.movieId == preference_idx].values[0]

            if self.transform is not None:
                preference_genres_feature = self.transform(
                    preference_genres_str, self.genres_list)
            preference_genres_feature_list.append(preference_genres_feature)
        preference_genres_feature = np.array(preference_genres_feature_list,
                                             dtype=np.float32)
        preference_genres_feature = preference_genres_feature.mean(0)

        y = self.data['rating'].iloc[idx]

        return torch.LongTensor([idx for idx in range(len(genres_list))]+[idx for idx in range(len(genres_list))]),\
               torch.from_numpy(np.concatenate((movie_genres_feature, preference_genres_feature), axis=-1)),\
               torch.tensor(y, dtype=torch.float32)


class DeepFM_FeatureExtraction(nn.Module):
    def __init__(self, embed_feat_size, feat_size, feat_dim=10):
        super(DeepFM_FeatureExtraction, self).__init__()
        self.embed_feat_size = embed_feat_size
        self.feat_size = feat_size
        self.feat_dim = feat_dim
        self.feat_embedding = nn.Embedding(self.feat_size, self.feat_dim)

        self.feat_bias = nn.Embedding(self.feat_size, 1)

        self.deep_layer1 = nn.Sequential(
            nn.Linear(self.feat_size * self.feat_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
        )
        self.deep_layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
        )
        self.deep_layer3 = nn.Linear(32, 16)

        self.out = nn.Linear(self.feat_size + self.feat_dim + 16, 1)

    def forward(self, feat_index, feat_value):
        '''
        :param feat_index: N,F
        :param feat_value: N,F
        :return:
        '''
        feat_embedding = self.feat_embedding(feat_index)  # N,F,K
        feat_bias = self.feat_bias(feat_index)  # N,F,1

        feat_value = feat_value.reshape(-1, self.feat_size, 1)
        feat_first_order = torch.mul(feat_bias, feat_value)  # batch, F,1
        feat_embed_value = torch.mul(feat_embedding, feat_value)  # batch, F, K

        feat_first_order = feat_first_order.reshape(-1,
                                                    self.feat_size)  # batch, F

        sum_feat_square = torch.sum(feat_embed_value, 1)  # N, K
        sum_feat_square = sum_feat_square**2

        square_feat_sum = feat_embed_value**2  # batch, F, K
        square_feat_sum = torch.sum(square_feat_sum, 1)  # batch, K

        feat_second_order = 0.5 * (sum_feat_square - square_feat_sum)

        #  deep layer
        #  batch,F*K
        feat_embed_value = feat_embed_value.reshape(
            -1, self.feat_size * self.feat_dim)

        deep_layer1 = self.deep_layer1(feat_embed_value)
        deep_layer2 = self.deep_layer2(deep_layer1)
        feat_deep = self.deep_layer3(deep_layer2)

        feat_conc = torch.cat([feat_first_order, feat_second_order, feat_deep],
                              -1)

        out = self.out(feat_conc)

        return out


data = pd.read_csv(os.path.join('./', 'feature.csv'))

train, test = train_test_split(data)

trainDataset = MovieLensDataset(train, './', transform=seperate_genres)
trainDataLoader = DataLoader(dataset=trainDataset,
                             batch_size=512,
                             shuffle=True)

num_epoch = 30
for epoch in range(num_epoch):
    print('start {} epoch'.format(epoch))
    avg_loss = 0.0
    for step, data in enumerate(trainDataLoader):
        feat_index, feat_value, target = data
        # print(feat_index.shape)
        # print(feat_value.shape)
        deepFM_model = DeepFM_FeatureExtraction(
            embed_feat_size=len(genres_list),
            feat_size=2 * len(genres_list),
            feat_dim=10)
        out = deepFM_model(feat_index, feat_value)

        deepFM_model.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            deepFM_model.parameters()),
                                     lr=1e-5)
        loss = loss_fn(out, target)
        # print('loss={}'.format(loss))
        avg_loss += loss.item() / len(trainDataLoader)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        deepFM_model.eval()

        # if step>=3:
        #     break
    print('epoch={}, avg_train_loss={}'.format(epoch, avg_loss))
