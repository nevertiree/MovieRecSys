# -*- coding: utf-8 -*-

from util.download import start_download
import os
import shutil

if __name__ == '__main__':

    """ Step 1: Download movie image """

    # Set movie image download directory
    poster_dir_path = os.path.join(os.getcwd(), "data", "image")

    # Clean image directory
    if os.path.exists(poster_dir_path):
        shutil.rmtree(poster_dir_path)
    os.mkdir(poster_dir_path)

    # Start downloading posters.
    start_download(os.path.join(os.getcwd(), "data", "poster_info.txt"),
                   poster_dir_path,
                   is_sync=False)
