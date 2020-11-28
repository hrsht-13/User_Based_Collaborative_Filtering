# Recommendation System
A **USER BASED COLLABORATIVE FILTERING** approach for recommending movies the users. This project uses the [MovieLens](http://movielens.org) dataset for evaluation and testing purposes.

# Dependencies
The following dependencies are required to be installed for proper functioning of the code:
>  - python 3.8.3
>  - pandas 1.1.4
>  - numpy 1.19.4
>  - scikit learn 0.23.2
>  - suprise 1.1.1 

# File Description
- *ori_ratings.csv* - the original ratings file from the movielens dataset
- *ratings.csv* - contains all the ratings except the users listed in test_user.txt, used in evaluation
- *movies.csv* - contains informations about the movie ie movieId, title, genres
- *RS_main.py* - contains the algorithm used in the recommendation system, used for evaluation
- *test.py* - for testing of recommendation system
- *test_user.py* - contains the randomly selected users list for testing, creates test_user.txt, remove the users in the list from the original ratings file and saves in ratings.csv
- *eval.csv* - output result after evaluation
- *output.csv* - output result after testing

# Creating test_user.txt
Accordingly change the users list defined in the test_user.py, save and run
> `python test_user.py`

This will create the test_user.txt and removes those users from the ratings.csv

# Evaluating
**RS_main.py** takes input the **ratings.csv** file that contains ratings given by the user to a particular movie and outputs a csv with the MAE error for each in during k-fold cross validation (default k = 5)
> `python RS_main.py --input ratings.csv --output eval.csv`

**Make sure ratings.csv is in the same folder of RS_main.py**

# Testing
**test.py** takes input **test_user.txt** which contains 10 randomly selected user ids and predicts the ratings for each movie for each users individually. The output contains the top movie recommendations as well as the actual the top 5 movies rated by the user.
> `python test.py --input test_user.txt --output output.csv`

**Make sure movies.csv is in the same folder of test.py**