import requests
import json
import tmdbsimple as tmdb
import urllib3


base_url = 'https://image.tmdb.org/t/p/w500'
tmdb.API_KEY = '75b3949a04006bebf98471a9410d4309'

def id2pic(id):
    """ the func usage is to change the id to pic
    :par
    |id: int, the idx of the movie in the dataset
    """
    global base_url
    movie = tmdb.Movies(id).info()
    URL = base_url+movie['belongs_to_collection']['poster_path']
    print(movie)
    f = open(str(id)+'.jpg', 'wb')
    request = requests.get(URL)
    f.write(request.content)
    f.close()

def poster_path_get(save_path, start_num, range_num):
    """ use this func to get the movies that have poster
    """
    global base_url
    poster_set = set() # store the poster id we have get
    f = open(save_path, 'w')

    """ get the poster from the imdb """
    for idx in range(start_num, range_num):
        print(idx)
        try:
            movie = tmdb.Movies(idx).info()
            if (not movie['poster_path'] == None) and (not movie['imdb_id'] in poster_set):
                str_content = str(movie['imdb_id']) + ',' + base_url + movie['poster_path'] + '\n'
                f.write(str_content)
                poster_set.add(movie['id']) # add the id to the set. so there is no copy
        except:
            continue

    """ close the file """
    f.close()

def poster_get(poster_path_file, poster_store_path):
        cnt = 0 # for cnt   
        for line in open(poster_path_file, 'r'):
            cnt += 1 
            print(cnt)
            try:
                line_str = line.split(sep=',') # get the poster name and url
                request = requests.get(line_str[1][:-1])
                f = open(poster_store_path+line_str[0] + '.jpg', 'wb')
                f.write(request.content)
                f.close()
            except:
                continue

if __name__ == "__main__":
    #poster_path_get('poster_data85000_120000.txt', 85000, 120000)
    #poster_path_get('poster_data_new.txt', 5000, 15000)
    poster_get('poster_data_new.txt', "./posters/")
    #poster_check('poster_data_final_.txt', "poster_checked.txt")
