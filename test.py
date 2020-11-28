#importing libraries
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import collections

import surprise
from surprise import Dataset, Reader, AlgoBase, PredictionImpossible
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split

from RS_main import RS_algorithm

#building parser
def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", help = "input filename", dest='input')
    parser.add_argument("--output", help = "output filename", dest='output')
    return parser

def get_top_n(predictions, n=5):
    #Return the top-N recommendation for each user from a set of predictions.
   
    # First map the predictions to each user.
    top_n = collections.defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    #recommended movie_ids for user
    movie_recom={}
    for uid, user_ratings in top_n.items():
        movie_recom[uid]=[iid for (iid, _) in user_ratings]
        
    return movie_recom

# returns movie prediction with ratings in a dictionary
def predicted_movies(algo, user_id, movie_recom, movies):
  recom = {}
  m_ids=movie_recom[user_id]
  title=[]
  rate=[]
  for i in m_ids:
    title.append((movies.loc[movies["movieId"] == i].values)[0][1])
    rate.append(algo.predict((user_id),(i), verbose=False).est)

  recom["Predicted_Movie"]=title
  recom["Predicted_Movie_Rating"]=rate

  return recom
  
# returns a dataframe with user id, top 5 predicted movies with their ratings, top 5 movies rated by user (ratings > 3)  
def get_prediction(algo, user_id, dataset, movie_recom, movies):
  pred_dict=(predicted_movies(algo, user_id, movie_recom, movies))
  given=pd.DataFrame.from_dict(pred_dict)

  user_data=dataset.loc[dataset["userId"] == user_id]
  rr=user_data.loc[user_data["rating"]>3]
  past=rr[rr["movieId"].isin(movie_recom[user_id])][:5][["title","rating"]]    #(use ~rr to get different movie from past)
  past.reset_index(drop=True, inplace=True)
  past.columns=["Movie_seen_in_past","Rating_of_the_movie_seen_in_past>3"]
  df = pd.DataFrame()
  df['Test User'] = [user_id] * 5

  return (pd.concat([df,given,past],axis=1))

def main():

    #defining variables
    parser = build_parser()
    arguments = parser.parse_args()
    
    n_similar_users = 3
    folds = 5
    sim_options = {'name': 'pearson'}
    input_file = arguments.input
    output_file = arguments.output

    #defining algorithm
    algo = RS_algorithm(n = n_similar_users, sim_options=sim_options)
    
    input_file = arguments.input
    output_file = arguments.output
    
    original_ratings = pd.read_csv('ori_ratings.csv')
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv("movies.csv",usecols=["movieId","title"])
    dataset = pd.merge(original_ratings,movies, on='movieId')
    
    # reading txt file for user ids
    users = []

    with open(input_file, 'r') as filehandle:
        for line in filehandle:
            currentuser = line[:-1]
            users.append(int(currentuser))
            
    test_data=original_ratings[original_ratings["userId"].isin(users)]
    
    # concatinating ratings and test_data, concatination is required to calculate similarities of users
    data = pd.concat([ratings, test_data], axis = 0)
    
    #defining reader with minimum and maximum rating values
    reader = Reader(rating_scale=(min(original_ratings['rating']),max(original_ratings['rating'])))
    #defining datasets for surprise
    train_set = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    test_set = Dataset.load_from_df(test_data[['userId', 'movieId', 'rating']], reader)
    
    trainset = train_set.build_full_trainset()
    # shortcut method to create a testing set
    NA, test = train_test_split(test_set, test_size=1.0)
    
    #fitting the algorithm
    algo.fit(trainset)
    #getting predictions
    predictions = algo.test(test) 
    
    #top 5 movies for test_users
    movies_recom = get_top_n(predictions, n=5)
    
    output = pd.DataFrame()
    for user_id in users:
        # getting top 5 predicted movies with ratings as well as the actual top 5 movies for each user
        x = get_prediction(algo, user_id, dataset, movies_recom, movies)
        output = pd.concat([output, x], axis = 0)
    
    #saving output    
    output = output.reset_index(drop = True)
    output.to_csv(output_file, index = False)
    print("Succesfully saved {}".format(output_file))
    
if __name__ == '__main__':
    main()                        