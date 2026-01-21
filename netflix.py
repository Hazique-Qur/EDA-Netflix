import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Netflix Titles EDA", page_icon="🎥", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    # Data cleaning
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df.fillna({'director': 'Unknown', 'cast': 'Unknown', 'country': 'Unknown'}, inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
    df['duration_unit'] = df['duration'].str.extract('([a-zA-Z\s]+)')
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("🎬 Netflix EDA Filters")
type_filter = st.sidebar.multiselect("Select Type", options=df['type'].unique(), default=df['type'].unique())
country_filter = st.sidebar.multiselect("Select Country", options=df['country'].unique(), default=[])
rating_filter = st.sidebar.multiselect("Select Rating", options=df['rating'].unique(), default=[])

# Apply filters
filtered_df = df[df['type'].isin(type_filter)]
if country_filter:
    filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
if rating_filter:
    filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]

# Main content
st.title("🎥 Netflix Titles Exploratory Data Analysis")
st.markdown("Dive into the world of Netflix content with interactive visualizations and insights.")

st.header("📊 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Titles", len(filtered_df))
    st.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
    st.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))
with col2:
    st.write("**Data Types:**")
    st.write(filtered_df.dtypes)

st.header("🔍 Sample Data")
st.dataframe(filtered_df.head(10), use_container_width=True)

st.header("📈 Categorical Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Type Distribution")
    type_counts = filtered_df['type'].value_counts()
    st.write(type_counts)
    fig, ax = plt.subplots()
    type_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#FF6B6B', '#4ECDC4'])
    st.pyplot(fig)

with col2:
    st.subheader("Top 10 Countries")
    top_countries = filtered_df['country'].value_counts().head(10)
    st.write(top_countries)
    fig, ax = plt.subplots()
    top_countries.plot(kind='barh', ax=ax, color='#45B7D1')
    st.pyplot(fig)

st.subheader("Rating Distribution")
rating_counts = filtered_df['rating'].value_counts()
st.write(rating_counts)
fig, ax = plt.subplots()
rating_counts.plot(kind='bar', ax=ax, color='#96CEB4')
st.pyplot(fig)

st.header("⏱️ Duration Insights")
movies = filtered_df[filtered_df['type'] == 'Movie']
tv_shows = filtered_df[filtered_df['type'] == 'TV Show']

col1, col2 = st.columns(2)
with col1:
    st.subheader("Movies Duration")
    if not movies.empty:
        st.write(movies['duration_num'].describe())
        fig, ax = plt.subplots()
        movies['duration_num'].hist(bins=20, ax=ax, color='#FECA57')
        ax.set_title('Movie Durations (min)')
        st.pyplot(fig)

with col2:
    st.subheader("TV Shows Duration")
    if not tv_shows.empty:
        st.write(tv_shows['duration_num'].describe())
        fig, ax = plt.subplots()
        tv_shows['duration_num'].hist(bins=10, ax=ax, color='#FF9FF3')
        ax.set_title('TV Show Durations (seasons)')
        st.pyplot(fig)

st.header("🌟 Top Contributors")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Directors")
    top_directors = filtered_df[filtered_df['director'] != 'Unknown']['director'].value_counts().head(10)
    st.write(top_directors)
    fig, ax = plt.subplots()
    top_directors.plot(kind='bar', ax=ax, color='#54A0FF')
    st.pyplot(fig)

with col2:
    st.subheader("Top 10 Cast Members")
    top_cast = filtered_df[filtered_df['cast'] != 'Unknown']['cast'].str.split(', ').explode().value_counts().head(10)
    st.write(top_cast)
    fig, ax = plt.subplots()
    top_cast.plot(kind='bar', ax=ax, color='#5F27CD')
    st.pyplot(fig)

st.subheader("Top 10 Genres")
top_genres = filtered_df['listed_in'].str.split(', ').explode().value_counts().head(10)
st.write(top_genres)
fig, ax = plt.subplots()
top_genres.plot(kind='bar', ax=ax, color='#00D2D3')
st.pyplot(fig)

st.header("📅 Trends Over Time")
st.subheader("Content Added by Year")
filtered_df['year_added'] = filtered_df['date_added'].dt.year
yearly_content = filtered_df['year_added'].value_counts().sort_index()
st.write(yearly_content)
fig, ax = plt.subplots()
yearly_content.plot(kind='line', ax=ax, color='#FF9F43', marker='o')
ax.set_title('Yearly Content Additions')
st.pyplot(fig)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
