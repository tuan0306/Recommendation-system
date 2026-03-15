import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

# Import các module từ thư mục src/
sys.path.append('.')
from src.utils import load_movie_mapping, search_all_movies_by_title, get_movie_title_by_id, get_items_rated_by_user, get_top_rated_movies
from src.recommender import ContentBasedFiltering, NBCF, HybridRecommender

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Gợi Ý Phim", page_icon="🎬", layout="wide")
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login' # Các view: 'login', 'logged_in', 'guest'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# --- HÀM LOAD DỮ LIỆU (Cache để không phải load lại mỗi lần click) ---
@st.cache_resource
def load_system():
    # 1. Load Mapping Data
    mapping_df = load_movie_mapping('data/processed/movie_mapping.csv')
    ratings_train = pd.read_csv('data/processed/ratings_a_train.csv').values
    
    # 2. Load Models
    cb_model = joblib.load('models/cb_model.pkl')
    cf_model = joblib.load('models/cf_model.pkl')
    hybrid_system = joblib.load('models/hb_model.pkl')
    
    return mapping_df, ratings_train, cb_model, cf_model, hybrid_system

# Khởi chạy hàm load
with st.spinner('Đang tải mô hình hệ thống...'):
    mapping_df, ratings_train, cb_model, cf_model, hybrid_system = load_system()

@st.cache_data
def cached_top_rated_movies(_ratings_data, _mapping_df, min_ratings=50, top_n=10):
    """Lấy top phim có điểm trung bình cao nhất (điều kiện: số người đánh giá >= 50)"""
    return get_top_rated_movies(_ratings_data,min_ratings,top_n)

# --- COMPONENT: GIAO DIỆN TÌM KIẾM PHIM ---
# Đóng gói giao diện tìm kiếm vào hàm để dùng chung cho cả User và Guest
def render_search_movie_ui():
    st.markdown("### 🔍 Khám phá Phim Tương Tự")
    
    # Giao diện tự động phản ứng khi có người nhập từ khóa (không cần nút bấm)
    movie_query = st.text_input("Nhập tên phim bạn yêu thích và ấn Enter (VD: Toy Story, Star...):")
    
    if movie_query:
        matches = search_all_movies_by_title(mapping_df, movie_query)
        
        if matches:
            st.success(f"🎉 Tuyệt vời! Tìm thấy {len(matches)} phim khớp với từ khóa '{movie_query}'.")
            
            # Tạo danh sách lựa chọn cho Selectbox (Format: "ID - Tên phim")
            options = [f"{m_id} - {title}" for m_id, title in matches.items()]
            selected_option = st.selectbox("👇 Vui lòng chọn một bộ phim chính xác từ danh sách dưới đây:", options)
            
            # Xử lý khi người dùng chọn 1 phim từ dropdown
            if selected_option:
                st.divider()
                
                # Tách lấy phần ID (nằm trước dấu "-")
                movie_id = int(selected_option.split(" - ")[0])
                movie_title = matches[movie_id]
                
                st.markdown(f"#### Đang hiển thị mạng lưới gợi ý cho: **{movie_title}**")
                
                # Hiển thị 2 cột thuật toán như cũ
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💡 Phim cùng thể loại (CB)")
                    try:
                        cb_sim_items = cb_model.recommend_similar_items(movie_id, top=10)
                        for i, m_id in enumerate(cb_sim_items, 1):
                            st.write(f"{i}. {get_movie_title_by_id(mapping_df, m_id)}")
                    except:
                        st.warning("Mô hình CB chưa hỗ trợ tính năng này.")

                with col2:
                    st.subheader("👥 Người khác cũng xem (CF)")
                    try:
                        cf_sim_items = cf_model.recommend_similar_items(movie_id, top=10)
                        for i, m_id in enumerate(cf_sim_items, 1):
                            st.write(f"{i}. {get_movie_title_by_id(mapping_df, m_id)}")
                    except:
                        st.warning("Mô hình CF chưa hỗ trợ tính năng này.")
        else:
            st.error(f"Rất tiếc, không có bộ phim nào chứa từ khóa '{movie_query}'. Vui lòng thử lại!")


# ==========================================
# MÀN HÌNH 1: ĐĂNG NHẬP (LOGIN SCREEN)
# ==========================================
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
            st.rerun() # Tải lại trang ngay lập tức để chuyển màn hình
            
    with col2:
        st.success("🌍 Dành cho khách")
        st.markdown("<br>", unsafe_allow_html=True) # Tạo khoảng trống cho cân đối
        if st.button("Tiếp tục mà không đăng nhập"):
            st.session_state.current_view = 'guest'
            st.rerun()

# ==========================================
# MÀN HÌNH 2: LOGGED-IN USER (ĐÃ ĐĂNG NHẬP)
# ==========================================
elif st.session_state.current_view == 'logged_in':
    u_id = st.session_state.user_id
    
    # --- Sidebar điều hướng ---
    st.sidebar.title(f"👤 Xin chào, User {u_id}")
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.current_view = 'login'
        st.session_state.user_id = None
        st.rerun()
        
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Lựa chọn chức năng:", ["🎯 Gợi ý cho tôi", "🔍 Tìm phim tương tự"])
    
    st.title("Bảng Điều Khiển Cá Nhân")
    
    # 1. Hiển thị lịch sử (Phim rate gần đây/cao nhất)
    rated_items, rated_scores = get_items_rated_by_user(ratings_train, u_id)
    if len(rated_items) > 0:
        st.markdown("### 🕒 Phim bạn đánh giá cao nhất")
        top_rated_idx = np.argsort(rated_scores)[-5:][::-1]
        history_list = [f"**{get_movie_title_by_id(mapping_df, rated_items[i])}** (⭐ {rated_scores[i]})" for i in top_rated_idx]
        st.info(" | ".join(history_list))
    
    st.divider() # Đường kẻ ngang
    
    # 2. Xử lý chức năng được chọn
    if app_mode == "🎯 Gợi ý cho tôi":
        st.markdown(f"### ✨ Top 10 Đề xuất dành riêng cho bạn (Hybrid System)")
        with st.spinner("Hệ thống đang phân tích sở thích..."):
            recommendations = hybrid_system.recommend_for_user(u_id, top=10)
            for i, m_id in enumerate(recommendations, 1):
                st.write(f"**{i}. {get_movie_title_by_id(mapping_df, m_id)}**")
            
    elif app_mode == "🔍 Tìm phim tương tự":
        render_search_movie_ui()

# ==========================================
# MÀN HÌNH 3: GUEST (CHƯA ĐĂNG NHẬP)
# ==========================================
elif st.session_state.current_view == 'guest':
    # --- Sidebar điều hướng ---
    st.sidebar.title("🌍 Xin chào, Khách vãng lai")
    if st.sidebar.button("🔑 Trở lại Đăng nhập"):
        st.session_state.current_view = 'login'
        st.rerun()
        
    st.title("Khám Phá Thế Giới Điện Ảnh")
    
    # 1. Hiển thị Top phim Rate cao nhất
    st.markdown("### 🏆 Top Phim Được Đánh Giá Cao Nhất Mọi Thời Đại")
    st.caption("Dựa trên bình chọn của cộng đồng (yêu cầu trên 50 lượt đánh giá)")
    
    top_movies_df = cached_top_rated_movies(ratings_train, mapping_df)
    
    # Trình bày dạng các thẻ (columns) cho đẹp mắt
    cols = st.columns(5)
    for idx, (index, row) in enumerate(top_movies_df.iterrows()):
        col_idx = idx % 5
        with cols[col_idx]:
            item_id = int(row['item_id'])
            title = get_movie_title_by_id(mapping_df, item_id)
            score = row['rating_mean']
            count = int(row['rating_count'])
            st.markdown(f"**{title}**<br>⭐ {score:.1f} ({count} rate)", unsafe_allow_html=True)
            
    st.divider()
    
    # 2. Luôn hiển thị tính năng Tìm phim
    render_search_movie_ui()