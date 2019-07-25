import eel
import pandas as pd
from scipy import stats
from scipy.stats.stats import pearsonr
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

movies_df=pd.read_csv('movies.csv')
ratings_df=pd.read_csv('ratings.csv')

movies_df['year']=movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title']=movies_df['title'].apply(lambda x: x.strip())
movies_df=movies_df.drop('genres',1)
ratings_df=ratings_df.drop('timestamp',1)
arr=[]
user_input={}
print("Started")
eel.init('web')
@eel.expose
def data_options():
    arr=str(movies_df.title.tolist())
    return arr
@eel.expose
def ml(user_movies,user_ratings):
    print("Running Algo")
    for i in range(len(user_movies)):
        print(user_movies[i])
    userInput=[
            {'title':''+user_movies[0]+'', 'rating':float(user_ratings[0])},
            {'title':''+user_movies[1]+'', 'rating':float(user_ratings[1])},
            {'title':''+user_movies[2]+'', 'rating':float(user_ratings[2])},
            {'title':''+user_movies[3]+'', 'rating':float(user_ratings[3])},
            {'title':''+user_movies[4]+'', 'rating':float(user_ratings[4])}
          ]
    print(userInput)
    inputMovies=pd.DataFrame(userInput)
    inputId=movies_df[movies_df.title.isin(inputMovies.title.tolist())]
    inputMovies=pd.merge(inputId,inputMovies)
    inputMovies=inputMovies.drop('year',1)
    print(inputMovies.head())
    userSubset=ratings_df[ratings_df.movieId.isin(inputMovies.movieId).tolist()]
    userSubsetGroup=userSubset.groupby(['userId'])
    userSubsetGroup=sorted(userSubsetGroup, key=lambda x: len(x[1]),reverse=True)
    userSubsetGroup=userSubsetGroup[0:100]
    pearsonCorrelationDict = {}
    for name, group in userSubsetGroup:
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        nRatings = len(group)
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        tempRatingList = temp_df['rating'].tolist()
        tempGroupList = group['rating'].tolist()
        pearson_coeff,p_value=pearsonr(tempRatingList,tempGroupList)
        pearsonCorrelationDict[name] = pearson_coeff
    

    pearson_df=pd.DataFrame.from_dict(pearsonCorrelationDict,orient='index')
    pearson_df.columns=['similarityIndex']
    pearson_df['userId']=pearson_df.index
    pearson_df.index=range(len(pearson_df))
    topUsers=pearson_df.sort_values(by='similarityIndex',ascending=False)
    topUsersRating=topUsers.merge(ratings_df,left_on='userId',right_on='userId',how='inner')
    topUsersRating['weightedRating']=topUsersRating['similarityIndex']*topUsersRating['rating']
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    recommendation_df = pd.DataFrame()
    #Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    temp_rec_mov=movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
    temp_rec_mov=temp_rec_mov[:5]
    rec_mov=temp_rec_mov.title.tolist()
    print(rec_mov)
    return rec_mov

eel.start('main.html')
