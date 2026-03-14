import pandas as pd
import numpy as np

def get_items_rated_by_user(rate_matrix,user_id):
    y_user=rate_matrix[:,0]
    ids=np.where(y_user==user_id)[0]
    items_id=rate_matrix[ids,1]
    scores=rate_matrix[ids,2]
    return (items_id,scores)

def load_movie_mapping(filepath='../data/processed/movie_mapping.csv'):
    return pd.read_csv(filepath,header=0)

def get_movie_title_by_id(mapping,id):
    try:
        return mapping.loc[id,'movie title']
    except:
        return 'Unknown Movie'