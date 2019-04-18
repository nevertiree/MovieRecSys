# -*- coding: utf-8 -*-

import requests
import os

import tmdbsimple as tmdb

base_url = 'https://image.tmdb.org/t/p/w500'
tmdb.API_KEY = '75b3949a04006bebf98471a9410d4309'


def poster_path_get(save_path, start_num, range_num):
    """ use this func to get the movies that have image
        :param save_path
        :param start_num
        :param range_num
    """
    global base_url
    poster_set = set()  # store the image id we have get

    with open(save_path, 'w') as f:
        """ get the image from the imdb """
        for idx in range(start_num, range_num):
            print(idx)
            try:
                movie = tmdb.Movies(idx).info()
                if (not movie['poster_path'] is None) and (
                        not movie['imdb_id'] in poster_set):
                    str_content = str(
                        movie['imdb_id']
                    ) + ',' + base_url + movie['poster_path'] + '\n'
                    f.write(str_content)

                    # add the id to the set. so there is no copy
                    poster_set.add(movie['id'])
            except:
                continue


def _download_poster(poster_name, poster_url, poster_path):
    print(poster_name, poster_url)
    with open(os.path.join(poster_path,
                           poster_name + '.jpg'),
              'wb') as f:
        request = requests.get(url=poster_url)
        f.write(request.content)


def start_download(url_file_path, poster_store_path):
    """ Download movie image.
        :param url_file_path: The path of file which stores the movie image URLs.
        :param poster_store_path: The path to store the downloaded movie posters.
    """

    with open(url_file_path, 'r') as url_file:
        while True:
            line = url_file.readline()
            if not line:
                break
            # Get the image name and url
            poster_name, poster_url = line.split(sep=',')
            _download_poster(poster_name, poster_url, poster_store_path)
