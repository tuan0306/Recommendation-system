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
    
def search_all_movies_by_title(mapping,title):
    try:
        matches=mapping[mapping['movie title'].str.contains(title,case=False,na=False)]
        if not matches.empty:
            return matches['movie title'].to_dict()
        else:
            return {}
    except Exception:
        return {}
    
def get_top_rated_movies(ratings_data,min_ratings=50,top=10):
    df=pd.DataFrame(ratings_data,columns=['user_id','item_id','rating','timestamp'])
    stats=df.groupby('item_id').agg(
        rating_mean=('rating','mean'),
        rating_count=('rating','count')
    ).reset_index()
    top_items=stats[stats['rating_count']>=min_ratings]
    top_items=top_items.sort_values('rating_mean',ascending=False).head(top)
    return top_items
    