import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Advanced Sleep Data EDA", layout="wide", page_icon="🧠")
st.title("🧠 Advanced Sleep Dataset Exploratory Data Analysis")
st.markdown("Comprehensive analysis of sleep patterns across species with ML insights")

# Load and clean data
def load_data():
    df = pd.read_csv('dataset_2191_sleep.csv')
    df.replace('?', np.nan, inplace=True)
    df['max_life_span'] = pd.to_numeric(df['max_life_span'], errors='coerce')
    df['gestation_time'] = pd.to_numeric(df['gestation_time'], errors='coerce')
    df['total_sleep'] = pd.to_numeric(df['total_sleep'], errors='coerce')
    df_clean = df.dropna()
    return df_clean

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Controls")
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR method)", value=False)
if remove_outliers:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    st.sidebar.success(f"Outliers removed. Remaining records: {len(df)}")

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Correlations", "📉 Relationships", "📦 Categorical", "🤖 ML Insights"])

with tab1:
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Avg Total Sleep", f"{df['total_sleep'].mean():.2f}h")

    with st.expander("Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Data Types & Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue())

    with st.expander("Summary Statistics"):
        st.dataframe(df.describe(), use_container_width=True)

    with st.expander("Outlier Analysis"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_info[col] = outliers
        st.write("Outliers per column (IQR method):")
        st.json(outlier_info)

with tab2:
    st.header("📈 Data Distributions")
    col1, col2 = st.columns([1, 2])
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select column", numeric_cols, key="dist_col")
        show_kde = st.checkbox("Show KDE", value=True)
        bins = st.slider("Number of bins", 5, 50, 20)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], bins=bins, kde=show_kde, ax=ax)
        ax.set_title(f'Distribution of {selected_col}')
        st.pyplot(fig)

    st.subheader("All Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:6]):
        sns.histplot(df[col], bins=15, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.header("🔗 Correlations")
    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr = df.corr(method=corr_method)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title(f'{corr_method.capitalize()} Correlation Matrix')
    st.pyplot(fig)

    st.subheader("Top Correlations with Total Sleep")
    sleep_corr = corr['total_sleep'].drop('total_sleep').abs().sort_values(ascending=False)
    st.dataframe(sleep_corr.reset_index().rename(columns={'total_sleep': 'Correlation'}))

with tab4:
    st.header("📉 Relationships with Total Sleep")
    x_cols = ['body_weight', 'brain_weight', 'max_life_span', 'gestation_time']
    selected_x = st.selectbox("Select X variable", x_cols, key="rel_x")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=selected_x, y='total_sleep', ax=ax)
        sns.regplot(data=df, x=selected_x, y='total_sleep', ax=ax, scatter=False, color='red')
        ax.set_title(f'{selected_x} vs Total Sleep')
        st.pyplot(fig)
    
    with col2:
        # Statistical test
        corr_coef, p_value = stats.pearsonr(df[selected_x], df['total_sleep'])
        st.write(f"Pearson Correlation: {corr_coef:.3f}")
        st.write(f"P-value: {p_value:.3f}")
        if p_value < 0.05:
            st.success("Statistically significant correlation")
        else:
            st.warning("No significant correlation")

    st.subheader("All Scatter Plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(x_cols):
        row, col_idx = divmod(i, 2)
        sns.scatterplot(data=df, x=col, y='total_sleep', ax=axes[row, col_idx])
        sns.regplot(data=df, x=col, y='total_sleep', ax=axes[row, col_idx], scatter=False, color='red')
        axes[row, col_idx].set_title(f'{col} vs Total Sleep')
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    st.header("📦 Categorical Variables Analysis")
    cat_cols = ['predation_index', 'sleep_exposure_index', 'danger_index']
    selected_cat = st.selectbox("Select categorical variable", cat_cols, key="cat_var")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x=selected_cat, y='total_sleep', ax=ax)
        ax.set_title(f'{selected_cat} vs Total Sleep')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df, x=selected_cat, y='total_sleep', ax=ax)
        ax.set_title(f'{selected_cat} vs Total Sleep (Violin)')
        st.pyplot(fig)

    st.subheader("All Categorical Plots")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(cat_cols):
        sns.boxplot(data=df, x=col, y='total_sleep', ax=axes[0, i])
        axes[0, i].set_title(f'{col} Boxplot')
        sns.violinplot(data=df, x=col, y='total_sleep', ax=axes[1, i])
        axes[1, i].set_title(f'{col} Violin')
    plt.tight_layout()
    st.pyplot(fig)

with tab6:
    st.header("🤖 Machine Learning Insights")
    st.subheader("Predict Total Sleep Duration")
    
    # Prepare data
    features = ['body_weight', 'brain_weight', 'max_life_span', 'gestation_time', 'predation_index', 'sleep_exposure_index', 'danger_index']
    X = df[features]
    y = df['total_sleep']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
    with col3:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.coef_})
    feature_importance = feature_importance.sort_values('importance', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
    ax.set_title('Feature Importance (Linear Regression Coefficients)')
    st.pyplot(fig)
    
    # Predictions vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Total Sleep')
    ax.set_ylabel('Predicted Total Sleep')
    ax.set_title('Actual vs Predicted Total Sleep')
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.write("Data source: Sleep dataset")
st.sidebar.write(f"Analysis on {len(df)} records")
if st.sidebar.button("Export Cleaned Data"):
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, "cleaned_sleep_data.csv", "text/csv")
