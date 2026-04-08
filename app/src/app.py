import streamlit as st
import pandas as pd
import numpy as np
import os
from client_api import search_all_movies_by_title, get_movie_title_by_id, get_items_rated_by_user, get_top_rated_movies,get_ratings_of_a_movie,recommend_similar_items,recommend_for_user
from utils_ui import render_movie_grid,fetch_poster


# Cau hinh trang
st.set_page_config(page_title="Hệ Thống Gợi Ý Phim", page_icon="🎬", layout="wide")
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login' # Cac trang thai: 'login', 'logged_in', 'guest'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# giao dien tim kiem phim
def render_search_movie_ui():
    st.markdown("### 🔍 Khám phá Phim Tương Tự")
    movie_query = st.text_input("Nhập tên phim bạn yêu thích và ấn Enter (VD: Toy Story, Star...):")
    
    if movie_query:
        matches = search_all_movies_by_title(movie_query)
        
        if not matches.empty:
            st.success(f"🎉 Tuyệt vời! Tìm thấy {len(matches)} phim khớp với từ khóa '{movie_query}'.")
            options = [f"{m_id} - {title}" for m_id, title in zip(matches['movie_id'], matches['title'])]
            selected_option = st.selectbox("👇 Vui lòng chọn một bộ phim chính xác từ danh sách dưới đây:", options)
            
            if selected_option:
                st.divider()
                parts = selected_option.split(" - ", 1)
                movie_id = int(parts[0])
                movie_title = parts[1]

                col_img, col_info = st.columns([1, 3])
                selected_title = str(get_movie_title_by_id(movie_id)).strip()
                _api_key=os.getenv('TMDB_KEY')
                selected_poster = fetch_poster(selected_title,_api_key)
                with col_img:
                    st.image(selected_poster, use_container_width=True)
                with col_info:
                    st.markdown(f"## {selected_title}")
                    try:
                        movie_reviews = get_ratings_of_a_movie(movie_id)
                        if len(movie_reviews) > 0:
                            avg_score = round(movie_reviews.mean(), 1)
                            total_votes = len(movie_reviews)
                            st.markdown(f"**⭐ Điểm đánh giá:** {avg_score}/5.0 *(từ {total_votes} người dùng)*")
                        else:
                            st.markdown("**⭐ Điểm đánh giá:** 0 lượt đánh giá")
                    except Exception as e:
                        st.error(f"Lỗi chi tiết: {e}")
                        st.markdown("**⭐ Điểm đánh giá:** Chưa có dữ liệu")
                st.markdown("---")
                    
                tab_cb, tab_cf = st.tabs(["🤖 Dựa trên Nội dung (Content-Based)", "👥 Dựa trên Cộng đồng (Collaborative)"])

                with tab_cb:
                    st.markdown("#### 🎬 Các phim có cùng chủ đề, thể loại")
                    with st.spinner("Đang phân tích nội dung phim..."):
                        cb_recommendations = recommend_similar_items(movie_id, top=10,model_type='cb')
                        render_movie_grid(cb_recommendations)

                with tab_cf:
                    st.markdown("#### 🍿 Những người thích phim này cũng xem")
                    with st.spinner("Đang tổng hợp dữ liệu cộng đồng..."):
                        cf_recommendations = recommend_similar_items(movie_id, top=10,model_type='cf')
                        render_movie_grid(cf_recommendations)
        else:
            st.error(f"Rất tiếc, không có bộ phim nào chứa từ khóa '{movie_query}'. Vui lòng thử lại!")


# man hinh dang nhap
if st.session_state.current_view == 'login':
    st.title("🍿 Chào mừng đến với Hệ Thống Gợi Ý Phim")
    st.markdown("Vui lòng định danh để nhận được những gợi ý cá nhân hóa tốt nhất.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("👤 Dành cho thành viên")
        user_input = st.number_input("Nhập User ID của bạn (Từ 0 đến 942):", min_value=0, max_value=942, step=1)
        if st.button("Đăng nhập vào hệ thống", type="primary"):
            st.session_state.user_id = user_input
            st.session_state.current_view = 'logged_in'
            st.rerun()
            
    with col2:
        st.success("🌍 Dành cho khách")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Tiếp tục mà không đăng nhập"):
            st.session_state.current_view = 'guest'
            st.rerun()

# man hinh khi da dang nhap
elif st.session_state.current_view == 'logged_in':
    u_id = st.session_state.user_id
    st.sidebar.title(f"👤 Xin chào, User {u_id}")
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.current_view = 'login'
        st.session_state.user_id = None
        st.rerun()
        
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Lựa chọn chức năng:", ["🎯 Gợi ý cho tôi", "🔍 Tìm phim tương tự"])
    
    st.title("Bảng Điều Khiển Cá Nhân")
    
    rated_items, rated_scores = get_items_rated_by_user(u_id)
    if len(rated_items) > 0:
        st.markdown("### 🕒 Phim bạn đánh giá cao nhất")
        with st.spinner('Đang tìm kiếm lịch sử đánh giá phim...'):
            top_rated_idx = np.argsort(rated_scores)[-5:][::-1]
            user_top_movies = []
            for idx in top_rated_idx:
                m_id = int(rated_items[idx])
                score = rated_scores[idx]
                user_top_movies.append((m_id, f"⭐ **{score}/5**"))
            
            render_movie_grid(user_top_movies, has_subtitle=True)
    
    st.markdown("---")
    
    if app_mode == "🎯 Gợi ý cho tôi":
        st.markdown(f"### ✨ Top 10 Đề xuất dành riêng cho bạn")
        if st.button("🚀 Bấm vào đây để AI tìm phim cho bạn"):
            with st.spinner("Hệ thống đang phân tích sở thích và tải hình ảnh..."):
                recommendations = recommend_for_user(u_id, top=10)
                render_movie_grid(recommendations, has_subtitle=False)
            
    elif app_mode == "🔍 Tìm phim tương tự":
        render_search_movie_ui()

# guest
elif st.session_state.current_view == 'guest':
    st.sidebar.title("🌍 Xin chào, Khách vãng lai")
    if st.sidebar.button("🔑 Trở lại Đăng nhập"):
        st.session_state.current_view = 'login'
        st.rerun()
        
    st.title("Khám Phá Thế Giới Điện Ảnh")
    
    st.markdown("### 🏆 Top Phim Được Đánh Giá Cao Nhất Mọi Thời Đại")
    st.caption("Dựa trên bình chọn của cộng đồng (yêu cầu trên 50 lượt đánh giá)")
    
    with st.spinner("Đang tải danh sách phim..."):
        top_movies_df = get_top_rated_movies()
        top_movies_list = []
        for index, row in top_movies_df.iterrows():
            item_id = int(row['movie_id'])
            score = row['rating_mean']
            count = int(row['rating_count'])
            subtitle_text = f"⭐ {score:.1f} ({count} rate)"
            top_movies_list.append((item_id,subtitle_text))
        render_movie_grid(top_movies_list, has_subtitle=True)
            
    st.markdown("---")
    
    # luon luon hien
    render_search_movie_ui()