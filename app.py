import streamlit as st
import pandas as pd
import requests
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ================= PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) =================
st.set_page_config(
    page_title="📚 Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS FOR PROFESSIONAL LOOK =================
st.markdown("""
<style>
	/* Import sophisticated fonts */
	@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Lato:wght@300;400;700&display=swap');

	/* Main background - Old money cream/ivory */
	.stApp {
		background: #f8f6f0;
	}

	/* Content container */
	.main .block-container {
		background: #ffffff;
		border-radius: 0;
		padding: 3rem 2.5rem;
		max-width: 1400px;
		margin: 0 auto;
		box-shadow: none;
		border-left: 1px solid #e8e3d5;
		border-right: 1px solid #e8e3d5;
	}

	/* Title styling - Classic serif */
	h1 {
		font-family: 'Cormorant Garamond', serif !important;
		color: #2c3e2e !important;
		text-align: center;
		font-size: 2.5rem !important;
		font-weight: 600 !important;
		margin-bottom: 0.3rem !important;
		letter-spacing: 2px;
		text-transform: uppercase;
	}

	/* Subtitle */
	.subtitle {
		font-family: 'Lato', sans-serif;
		color: #6b7f6a;
		text-align: center;
		font-weight: 300;
		font-size: 0.95rem;
		margin-bottom: 2rem;
		letter-spacing: 1px;
	}

	/* Decorative line under title */
	.title-divider {
		width: 150px;
		height: 1px;
		background: linear-gradient(to right, transparent, #8b7355, transparent);
		margin: 1rem auto 1.5rem auto;
	}

	/* Category section title */
	.category-title {
		font-family: 'Cormorant Garamond', serif;
		color: #2c3e2e;
		font-size: 2rem;
		font-weight: 600;
		text-align: center;
		margin: 2rem 0 0.5rem 0;
		letter-spacing: 1px;
	}

	.category-count {
		font-family: 'Lato', sans-serif;
		color: #8b7355;
		text-align: center;
		font-size: 0.85rem;
		margin-bottom: 1.5rem;
		font-weight: 400;
		letter-spacing: 0.5px;
	}

	/* Select box styling */
	.stSelectbox {
		margin: 1.5rem auto;
		max-width: 500px;
	}

	.stSelectbox label {
		font-family: 'Lato', sans-serif !important;
		color: #2c3e2e !important;
		font-size: 1rem !important;
		font-weight: 400 !important;
		letter-spacing: 0.5px;
	}

	.stSelectbox > div > div {
		background: #fefdfb;
		border: 1.5px solid #d4cab8;
		border-radius: 0;
		font-family: 'Lato', sans-serif;
		font-size: 1rem;
		transition: all 0.3s ease;
		color: #2c3e2e;
	}

	.stSelectbox > div > div:hover {
		border-color: #8b7355;
		background: #ffffff;
	}

	.stSelectbox > div > div:focus-within {
		border-color: #8b7355;
		box-shadow: 0 0 0 1px #8b7355;
	}

	/* Book card container */
	.book-card {
		background: #fefdfb;
		border: 1px solid #e8e3d5;
		padding: 1.5rem 1rem;
		text-align: center;
		transition: all 0.35s ease;
		height: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
		margin-bottom: 1.5rem;
	}

	.book-card:hover {
		transform: translateY(-8px);
		box-shadow: 0 12px 30px rgba(44, 62, 46, 0.12);
		border-color: #8b7355;
		background: #ffffff;
	}

	/* Book cover image */
	.book-cover {
		width: 140px;
		height: 200px;
		object-fit: cover;
		border-radius: 2px;
		box-shadow: 0 6px 20px rgba(44, 62, 46, 0.15);
		transition: all 0.35s ease;
		margin-bottom: 1rem;
		border: 1px solid #e8e3d5;
	}

	.book-card:hover .book-cover {
		box-shadow: 0 10px 35px rgba(44, 62, 46, 0.25);
		transform: scale(1.03);
	}

	/* Book title */
	.book-title {
		font-family: 'Cormorant Garamond', serif;
		font-weight: 600;
		font-size: 1.1rem;
		color: #2c3e2e;
		margin-top: 0.8rem;
		margin-bottom: 0.4rem;
		line-height: 1.4;
		min-height: 2.5rem;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	/* Book author */
	.book-author {
		font-family: 'Lato', sans-serif;
		font-size: 0.85rem;
		color: #8b7355;
		font-weight: 400;
		font-style: italic;
		letter-spacing: 0.3px;
	}

	/* Info box styling */
	.stInfo {
		background: #f8f6f0;
		border: 1px solid #d4cab8;
		border-left: 3px solid #8b7355;
		border-radius: 0;
		font-family: 'Lato', sans-serif;
		color: #2c3e2e;
	}

	/* Warning box styling */
	.stWarning {
		background: #fef9f0;
		border: 1px solid #e8d9b8;
		border-left: 3px solid #c9a85c;
		border-radius: 0;
		font-family: 'Lato', sans-serif;
		color: #2c3e2e;
	}

	/* Error box styling */
	.stError {
		background: #fef5f5;
		border: 1px solid #e8d5d5;
		border-left: 3px solid #a85c5c;
		border-radius: 0;
		font-family: 'Lato', sans-serif;
		color: #2c3e2e;
	}

	/* Divider */
	hr {
		margin: 3rem 0;
		border: none;
		height: 1px;
		background: linear-gradient(to right, transparent, #d4cab8, transparent);
	}

	/* Sidebar styling */
	section[data-testid="stSidebar"] {
		background: #2c3e2e;
	}

	section[data-testid="stSidebar"] * {
		color: #f8f6f0 !important;
	}

	/* Remove Streamlit branding */
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	header {visibility: hidden;}

	/* Custom header decoration */
	.header-decoration {
		text-align: center;
		margin: 1rem auto 1.5rem auto;
	}

	.ornament {
		display: inline-block;
		width: 30px;
		height: 1px;
		background: #8b7355;
		margin: 0 0.8rem;
		vertical-align: middle;
	}

	.ornament-circle {
		display: inline-block;
		width: 5px;
		height: 5px;
		background: #8b7355;
		border-radius: 50%;
		vertical-align: middle;
	}

	/* Footer styling */
	.custom-footer {
		text-align: center;
		color: #6b7f6a;
		font-family: 'Lato', sans-serif;
		font-size: 0.9rem;
		padding: 3rem 0 2rem 0;
		border-top: 1px solid #e8e3d5;
		margin-top: 4rem;
		letter-spacing: 0.5px;
	}

	/* Responsive adjustments */
	@media (max-width: 768px) {
		h1 {
			font-size: 2.8rem !important;
		}

		.book-cover {
			width: 150px;
			height: 220px;
		}

		.main .block-container {
			padding: 2rem 1.5rem;
		}
	}
</style>
""", unsafe_allow_html=True)

# ================= CONSTANTS =================
PLACEHOLDER = "https://via.placeholder.com/140x200.png?text=No+Cover"


# ================= LOAD DATA AND MODELS =================
@st.cache_data
def load_data(path: str = "books.xlsx"):
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        st.error("❌ Could not find books.xlsx. Please place it in the project folder.")
        return pd.DataFrame(columns=["Title", "Author", "Category"])

    for col in ["Title", "Author", "Category"]:
        if col not in df.columns:
            df[col] = ""

    return df


@st.cache_data
def load_ml_models():
    try:
        # Load TF-IDF matrix and vectorizer from models folder
        tfidf_matrix = load_npz("models/tfidf_matrix.npz")
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        return tfidf_matrix, tfidf_vectorizer
    except FileNotFoundError:
        st.error("❌ ML model files not found in 'models' folder.")
        return None, None


df = load_data()
tfidf_matrix, tfidf_vectorizer = load_ml_models()


# ================= COVER FETCHER =================
@st.cache_data(show_spinner=False)
def fetch_cover(title: str, author: str = "") -> str:
    title = str(title).strip()
    author = str(author).strip()
    query = f"{title} {author}".strip()

    if not query:
        return PLACEHOLDER

    # ---------- Google Books API ----------
    try:
        params = {"q": query, "maxResults": 1}
        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params=params,
            timeout=6
        )

        if resp.status_code == 200:
            items = resp.json().get("items")
            if items:
                info = items[0].get("volumeInfo", {})
                images = info.get("imageLinks", {})
                for key in ["extraLarge", "large", "medium", "thumbnail", "smallThumbnail"]:
                    if images.get(key):
                        return images[key].replace("http://", "https://")
    except Exception:
        pass

    # ---------- Open Library API ----------
    try:
        ol_resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"title": title, "author": author, "limit": 1},
            timeout=6
        )

        if ol_resp.status_code == 200:
            docs = ol_resp.json().get("docs")
            if docs:
                doc = docs[0]

                if "cover_i" in doc:
                    return f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-L.jpg"

                if "isbn" in doc and len(doc["isbn"]) > 0:
                    return f"https://covers.openlibrary.org/b/isbn/{doc['isbn'][0]}-L.jpg"

                if "key" in doc:
                    olid = doc["key"].split("/")[-1]
                    return f"https://covers.openlibrary.org/b/olid/{olid}-L.jpg"
    except Exception:
        pass

    return PLACEHOLDER


# ================= ML RECOMMENDATION FUNCTION =================
def get_recommendations(book_title: str, num_recommendations: int = 8):
    """Get book recommendations based on TF-IDF similarity"""

    if tfidf_matrix is None or tfidf_vectorizer is None:
        return pd.DataFrame()

    # Find the book index
    try:
        idx = df[df['Title'].str.lower() == book_title.lower()].index[0]
    except IndexError:
        return pd.DataFrame()

    # Calculate cosine similarity
    book_vector = tfidf_matrix[idx]
    similarity_scores = cosine_similarity(book_vector, tfidf_matrix).flatten()

    # Get top similar books (excluding the book itself)
    similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]

    # Return recommended books
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = similarity_scores[similar_indices]

    return recommendations


# ================= DISPLAY BOOK GRID =================
def display_books(books_df, cols_per_row=4):
    """Display books in a grid layout"""
    for i in range(0, len(books_df), cols_per_row):
        cols = st.columns(cols_per_row, gap="large")

        for j in range(cols_per_row):
            if i + j < len(books_df):
                book = books_df.iloc[i + j]

                title = str(book.get("Title", "Unknown"))
                author = str(book.get("Author", "Unknown"))

                image_url = None
                if "Image_URL" in book.index:
                    img = book.get("Image_URL")
                    if pd.notna(img) and str(img).strip():
                        image_url = img

                if not image_url:
                    image_url = fetch_cover(title, author)

                with cols[j]:
                    st.markdown(
                        f"""
						<div class="book-card">
							<img src="{image_url}" class="book-cover" alt="{title}" />
							<div class="book-title">{title}</div>
							<div class="book-author">{author}</div>
						</div>
						""",
                        unsafe_allow_html=True
                    )


# ================= UI =================

# Header with decorative elements
st.markdown("""
	<div class="header-decoration">
		<span class="ornament"></span>
		<span class="ornament-circle"></span>
		<span class="ornament"></span>
	</div>
""", unsafe_allow_html=True)

st.title("BOOK COLLECTION")

st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">INTELLIGENT RECOMMENDATIONS POWERED BY MACHINE LEARNING</p>', unsafe_allow_html=True)

if df.empty:
    st.warning("📭 No dataset loaded. Please place books.xlsx in the project directory.")
elif tfidf_matrix is None:
    st.error(
        "❌ ML models not loaded. Please ensure tfidf_matrix.npz and tfidf_vectorizer.joblib are in the 'models' folder.")
else:
    # Create tabs for different browsing modes
    tab1, tab2 = st.tabs(["🎯 Get Recommendations", "📚 Browse by Category"])

    # ================= TAB 1: ML RECOMMENDATIONS =================
    with tab1:
        st.markdown("### Find Similar Books")

        # Book selection
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            book_titles = df['Title'].tolist()
            selected_book = st.selectbox(
                "SELECT A BOOK YOU LIKE",
                ["— SELECT A BOOK —"] + book_titles
            )

        if selected_book != "— SELECT A BOOK —":
            # Get recommendations
            recommendations = get_recommendations(selected_book, num_recommendations=8)

            if not recommendations.empty:
                st.markdown(f'<h2 class="category-title">Books Similar to "{selected_book}"</h2>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<p class="category-count">{len(recommendations)} Recommendations Based on Content Similarity</p>',
                    unsafe_allow_html=True)

                display_books(recommendations)
            else:
                st.warning("Could not find recommendations for this book.")
        else:
            st.info("👆 Select a book to get intelligent recommendations based on content similarity")

    # ================= TAB 2: CATEGORY BROWSING =================
    with tab2:
        st.markdown("### Browse by Genre")

        categories = (
            df["Category"]
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )

        # Center the selectbox
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected_category = st.selectbox(
                "SELECT CATEGORY",
                ["— SELECT A CATEGORY —"] + sorted(categories),
                key="category_select"
            )

        if selected_category != "— SELECT A CATEGORY —":
            books = df[df["Category"].astype(str) == selected_category].reset_index(drop=True)

            # Category title
            st.markdown(f'<h2 class="category-title">{selected_category}</h2>', unsafe_allow_html=True)
            st.markdown(f'<p class="category-count">{len(books)} Volumes Available</p>', unsafe_allow_html=True)

            display_books(books)
        else:
            st.info("👆 Select a category to browse our collection")

# ================= FOOTER =================
st.markdown("""
	<div class="custom-footer">
		<p>EST. 2024 · CRAFTED FOR BIBLIOPHILES · POWERED BY ML</p>
	</div>
""", unsafe_allow_html=True)