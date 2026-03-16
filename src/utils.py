import pandas as pd
import numpy as np
import requests
import streamlit as st

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

def get_ratings_of_a_movie(ratings_df,m_id):
    df=pd.DataFrame(ratings_df,columns=['user_id','item_id','rating','timestamp'])
    return df[df['item_id']==m_id]['rating']

@st.cache_data(show_spinner=False)
def fetch_poster(movie_title):
    search_title = movie_title.split(' (')[0]
    API_KEY = "107e36cceba118779a580cef51c2327b"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={search_title}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['results'] and data['results'][0]['poster_path']:
            poster_path = data['results'][0]['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
    except Exception:
        pass
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/500px-No-Image-Placeholder.svg.png"

def render_movie_grid(recommendation_list, mapping_df, has_subtitle=False):
    """Hàm hỗ trợ in danh sách phim thành lưới 5 cột"""
    for i in range(0, len(recommendation_list), 5):
        cols = st.columns(5)
        row_movies = recommendation_list[i : i+5]
        for col_idx, movie in enumerate(row_movies):
            if has_subtitle:
                m_id=movie[0]
                subtitle=movie[1]
            else:
                m_id=movie
            title = str(get_movie_title_by_id(mapping_df, m_id)).strip()
            poster_url = fetch_poster(title)
            with cols[col_idx]:
                st.markdown(
                    f'''
                    <img src="{poster_url}" style="width: 100%; height: 300px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;">
                    ''', unsafe_allow_html=True
                )
                st.markdown(f"**{title}**")
                if has_subtitle:
                    st.markdown(subtitle,unsafe_allow_html=True)