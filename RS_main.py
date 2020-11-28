#importing libraries
from argparse import ArgumentParser
import time

import pandas as pd
import numpy as np

import surprise
from surprise import Dataset, Reader, AlgoBase, PredictionImpossible
from surprise.model_selection import KFold

#building parser
def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", help = "input filename", dest='input')
    parser.add_argument("--output", help = "output filename", dest='output')
    return parser

#building RS algorithm
class RS_algorithm(AlgoBase):

    def __init__(self, n, epsilon = 1e-5, sim_options={}):

        AlgoBase.__init__(self, sim_options=sim_options)
        
        self.n = n
        self.epsilon = epsilon

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Compute similarities
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
            
                
        numerator = 0
        
        # ratings of item i by all users
        item_rating_by_user = self.trainset.ir[i]
        
        # average rating by user u
        avg_rating_by_user_u = sum(y[1] for y in self.trainset.ur[u])/len(self.trainset.ur[u])

        # Compute similarities between u and v, where v describes all other
        # users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in item_rating_by_user]
        
        # getting top k similar users for a user u
        top_k = sorted(neighbors, key=lambda x: x[1], reverse=True)[:self.n]
        
        # calculating numerator part of the resnick prediction function
        for j,(v, _) in enumerate(top_k):
            avg_rating_by_user_v = sum(y[1] for y in self.trainset.ur[v])/len(self.trainset.ur[v])
            numerator += top_k[j][1] * (list(filter(lambda x:v in x, item_rating_by_user))[0][1] - avg_rating_by_user_v)
            
        # calculating denominator part of the resnick prediction function
        denominator = sum(abs(y[1]) for y in top_k)
            
        # getting prediction from the resnick prediction function
        # adding a small value epsilon to denominator to avoid division by 0
        prediction = avg_rating_by_user_u + (numerator/(denominator + self.epsilon))

        return prediction

def Kfold_validation(k, algo, data):
    # determining number of folds of splitting
    kf = KFold(n_splits=k)
    # dictionary to hold folds with their MAE values
    fold_dict = {}
    # list of folds numbers
    folds = []
    # list of errors
    error = []

    for j,(trainset, testset) in enumerate(kf.split(data)):
        start_time = time.time()
        #append fold number in folds list
        folds.append('FOLD ' + str(j))
        #fitting algorithm on training set
        algo.fit(trainset)
        #predicting on test set
        predictions = algo.test(testset)
        #appending error in errors list
        error.append(surprise.accuracy.mae(predictions, verbose = False))
        end_time = time.time()
        print('Fold {}, MAE: {:.3f}, Time Elapsed: {:.3f} seconds'.format(j, error[j], end_time-start_time))
    #making key value pairs in dictionary
    #FOLD as key and folds list as value
    fold_dict['FOLD'] = folds
    #MAE as key and error list as value
    fold_dict['MAE'] = error

    return fold_dict

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
    #reading csv file
    df = pd.read_csv(input_file)
    #defining reader with minimum and maximum rating values
    reader = Reader(rating_scale=(min(df['rating']),max(df['rating'])))
    #making dataset relevant to AlgoBase class of surprise library
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)
    #K fold cross validation
    fold_dict = Kfold_validation(k = folds, algo = algo, data = data)
    #saving results in csv
    pd.DataFrame.from_dict(fold_dict).to_csv(output_file, index = False)
    print("Evaluation Complete! Results successfully saved to {}".format(output_file))

if __name__ == '__main__':
    main()