# -*- coding: utf-8 -*-
import requests
import os
import time
import asyncio
import tmdbsimple as tmdb

from concurrent import futures
from multiprocessing import cpu_count

MAX_WORKER = cpu_count()

# API_KEY
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
    with open(os.path.join(poster_path,
                           poster_name + '.jpg'),
              'wb') as f:
        request = requests.get(url=poster_url)
        f.write(request.content)


async def _async_download(url_file_path, poster_store_path):
    """ Download movie poster image (async).
        :param url_file_path: The path of file which stores the movie image URLs.
        :param poster_store_path: The path to store the downloaded movie posters.
    """
    print("Start async downloading...")

    with open(url_file_path, 'r') as url_file:
        """ Read each line in file, eg:
            tt0067753,https://image.tmdb.org/t/p/w500/l3swEgsxYSBhx2a4xm6PWA757lH.jpg
        """
        with futures.ThreadPoolExecutor(MAX_WORKER) as executor:
            loop = asyncio.get_event_loop()
            future_list = (
                loop.run_in_executor(
                    executor,  # concurrent executor
                    _download_poster,  # downloader
                    *line.split(","), poster_store_path  # downloader parameter
                )
                for line in url_file  # movie poster <name, url>
            )
            for _ in await asyncio.gather(*future_list):
                pass


def _sync_download(url_file_path, poster_store_path):
    """ Download movie poster image (sync).
        :param url_file_path: The path of file which stores the movie image URLs.
        :param poster_store_path: The path to store the downloaded movie posters.
    """
    print("Start sync downloading...")

    with open(url_file_path, 'r') as url_file:
        for line in url_file:
            """ Read each line in file, eg:
                tt0067753,https://image.tmdb.org/t/p/w500/l3swEgsxYSBhx2a4xm6PWA757lH.jpg
            """
            poster_name, poster_url = line.split(sep=',')
            _download_poster(poster_name, poster_url, poster_store_path)


def start_download(url_file_path, poster_store_path, is_sync=False):

    begin_time = time.time()

    if is_sync:
        _sync_download(url_file_path, poster_store_path)
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_async_download(url_file_path, poster_store_path))

    end_time = time.time()
    print("Time: " + str(end_time-begin_time))
