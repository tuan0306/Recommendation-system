import streamlit as st
import requests
import concurrent.futures

from client_api import get_movie_title_by_id

@st.cache_data(show_spinner=False)
def fetch_poster(movie_title,_api_key):
    search_title = movie_title.split(' (')[0]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={_api_key}&query={search_title}"
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


def fetch_multiple_posters(movie_titles,_api_key):
    """
    Hàm này nhận vào một List các ID phim và tải tất cả Poster cùng một lúc.
    """
    posters = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_title = {executor.submit(fetch_poster, title, _api_key): title for title in movie_titles}
        for future in concurrent.futures.as_completed(future_to_title):
            title = future_to_title[future]
            try:
                posters[title] = future.result()
            except Exception:
                posters[title] = "https://via.placeholder.com/342x513?text=No+Poster"
                
    return [posters[title] for title in movie_titles]

def render_movie_grid(recommendation_list, has_subtitle=False):
    movie_titles = []
    movie_details = [] 
    
    for movie in recommendation_list:
        if has_subtitle:
            m_id = movie[0]
            subtitle = movie[1]
        else:
            m_id = movie
            subtitle = None
        
        title = str(get_movie_title_by_id(m_id)).strip()
        
        movie_titles.append(title)
        movie_details.append({"title": title, "subtitle": subtitle})
    
    API_KEY= st.secrets['TMDB_KEY']
    all_posters = fetch_multiple_posters(movie_titles,API_KEY)
    
    for i in range(0, len(movie_details), 5):
        cols = st.columns(5)
    
        row_details = movie_details[i : i+5]
        row_posters = all_posters[i : i+5]

        for col_idx, (detail, poster_url) in enumerate(zip(row_details, row_posters)):
            with cols[col_idx]:
                st.markdown(
                    f'''
                    <img src="{poster_url}" style="width: 100%; height: 300px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;">
                    ''', unsafe_allow_html=True
                )
                st.markdown(f"**{detail['title']}**")
                
                if has_subtitle:
                    st.markdown(detail['subtitle'], unsafe_allow_html=True)
