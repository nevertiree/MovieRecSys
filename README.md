# Movie Rec Sys
package insatll:  
    torch  
    PIL  
    tmdbsimple  
    urlib3  
    json  
    requests  

# Steps for get the feature of posters

## step 1: Prepare training data

    1. Download movie posters to /data/posters/

    `python get_poster.py`

## step 2:

    Use the feature_vgg16.py to extract the feature of poster with identical imdb_id,you should write code like this:
    example : you want to get the imdb_id='1111' movie poster's feature  
    code example:  
    import feature_vgg  
    fep = Feature_Extract_Poster()
    feature_for_imdbid = fep.extract_feature('1111')  
    
    you will get two results, one for real feature of the poster, another is None which shows that no poster in in the folder
