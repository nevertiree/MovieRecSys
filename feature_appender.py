
import os
import pandas as pd
from util.feature_resnet import Feature_Extract_Poster

if __name__ == '__main__':

    image_dir = os.path.join(os.getcwd(), "data", "image")
    fep = Feature_Extract_Poster(image_dir)

    user_data_file = os.path.join(os.getcwd(), "data", "movies.csv")
    result_data_file = os.path.join(os.getcwd(), "data", "movies_result.csv")

    movie_data = pd.read_csv(user_data_file)
    movie_df = pd.DataFrame(movie_data)
    movie_id_col = movie_df

    print("Load origin file successfully.")

    with open(result_data_file, "a+") as of:
        print("Create output file successfully.")
        for line in f_csv:
            image_name = line.split(",")[6]
            try:
                full_movie_id = "tt" + str(int(float(image_name))).zfill(7)
            except ValueError:
                print(line.split(",")[0]+"-"+image_name)
                continue

            if image_name:
                # print(full_movie_id)
                feature_str = fep.extract_feature(full_movie_id)
                if feature_str:
                    # split (lld)
                    feature_str = ",".join(feature_str.split("|"))   # Addition
                    feature_str = "," + feature_str
                    line = line.rstrip() + feature_str + "\n"
                    # print(image_name)

            of.write(line)
