import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords

# --- Load Data ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

st.title("Keyword Data Analysis & Visualization")

uploaded_file = st.file_uploader("Upload your kw_data.csv file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # --- Data Overview ---
    st.header("Data Overview")
    st.dataframe(df.head())

    # --- Descriptive Statistics ---
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

    # --- Correlation Analysis ---
    st.subheader("Correlation Analysis (Select Columns)")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_corr_cols = st.multiselect("Select numerical columns for correlation analysis", numerical_cols, default=numerical_cols[:5])
    if selected_corr_cols:
        corr_matrix = df[selected_corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, labels=dict(color="Correlation"), x=selected_corr_cols, y=selected_corr_cols)
        st.plotly_chart(fig_corr)

    # --- Trend Analysis ---
    st.header("Search Volume Trend Analysis")
    trend_col = st.selectbox("Select a trend column to visualize",
                             ['search volume trend monthly', 'search volume trend quarterly', 'search volume trend yearly'])
    trend_counts = df[trend_col].value_counts().sort_index()
    fig_trend = px.bar(trend_counts, x=trend_counts.index, y=trend_counts.values,
                       labels={'x': trend_col, 'y': 'Number of Keywords'})
    st.plotly_chart(fig_trend)

    # --- Categorical Analysis ---
    st.header("Categorical Feature Analysis")
    categorical_col = st.selectbox("Select a categorical column to visualize",
                                   df.select_dtypes(include=['object']).columns.tolist())
    if categorical_col:
        category_counts = df[categorical_col].value_counts()
        fig_cat = px.bar(category_counts, x=category_counts.index, y=category_counts.values,
                          labels={'x': categorical_col, 'y': 'Number of Keywords'})
        st.plotly_chart(fig_cat)

    # --- Creating Metrics for Ad Potential ---
    st.header("Ad Potential Metrics")
    estimated_conversion_rate = st.sidebar.slider("Estimated Conversion Rate (%)", 0.01, 10.0, 2.0, 0.01) / 100
    estimated_aov = st.sidebar.number_input("Estimated Average Order Value", min_value=0.0, value=50.0)

    df['potential_daily_revenue'] = df['daily clicks average'] * estimated_conversion_rate * estimated_aov
    df['potential_daily_profit'] = df['potential_daily_revenue'] - df['daily cost average']
    df['roas'] = df['potential_daily_revenue'] / df['daily cost average']
    df['ctr'] = df['daily clicks average'] / df['daily impressions average']
    df['cpa'] = df['daily cost average'] / (df['daily clicks average'] * estimated_conversion_rate)
    df['efficiency_score'] = df['search volume'] / df['cpc average']

    st.subheader("Top Keywords by Potential Daily Profit")
    top_profit = df.sort_values(by='potential_daily_profit', ascending=False).head(10)
    st.dataframe(top_profit[['keyword', 'potential_daily_profit']])
    fig_profit = px.bar(top_profit, x='keyword', y='potential_daily_profit',
                         labels={'potential_daily_profit': 'Potential Daily Profit'})
    st.plotly_chart(fig_profit)

    # --- Segmenting Keywords ---
    st.header("Keyword Segmentation")
    segmentation_col = st.selectbox("Select a column to segment by", df.columns.tolist())
    if segmentation_col:
        segment_options = df[segmentation_col].unique().tolist()
        selected_segments = st.multiselect(f"Select {segmentation_col} values to filter", segment_options, default=segment_options[:min(5, len(segment_options))])
        if selected_segments:
            segmented_df = df[df[segmentation_col].isin(selected_segments)]
            st.dataframe(segmented_df)

            # Visualize segmented data (example: potential profit)
            if 'potential_daily_profit' in segmented_df.columns:
                fig_segment_profit = px.bar(segmented_df, x='keyword', y='potential_daily_profit',
                                             labels={'potential_daily_profit': 'Potential Daily Profit'},
                                             title=f'Potential Daily Profit for Segment: {segmentation_col}')
                st.plotly_chart(fig_segment_profit)

    # --- Topic Clustering ---
    st.header("Topic Clustering")
    clustering_method = st.selectbox("Select a Topic Clustering Method",
                                     ["Manual Grouping", "Rule-Based Grouping", "Keyword Overlap", "Text Similarity (FuzzyWuzzy)", "Text Similarity (Cosine)", "Using Existing Search Intent"])

    if clustering_method == "Manual Grouping":
        st.subheader("Manual Grouping (Based on Keyword Content)")
        def assign_topic_cluster_manual(keyword):
            keyword = keyword.lower()
            if 'air wick' in keyword or 'airwick' in keyword:
                if 'refill' in keyword:
                    return 'Air Wick Refills'
                elif 'plug in' in keyword:
                    return 'Air Wick Plug-Ins'
                elif 'essential mist' in keyword:
                    return 'Air Wick Essential Mist Diffusers'
                elif 'spray' in keyword:
                    return 'Air Wick Sprays'
                elif 'diffuser' in keyword:
                    return 'Air Wick Diffusers'
                else:
                    return 'General Air Wick'
            elif 'glade' in keyword:
                if 'refill' in keyword:
                    return 'Glade Refills'
                elif 'plug in' in keyword:
                    return 'Glade Plug-Ins'
                elif 'automatic spray' in keyword:
                    return 'Glade Automatic Sprays'
                elif 'air freshener' in keyword:
                    return 'General Glade Air Fresheners'
                else:
                    return 'General Glade'
            elif 'febreze' in keyword:
                if 'plug in' in keyword:
                    return 'Febreze Plug-Ins'
                else:
                    return 'General Febreze'
            elif 'essential oil' in keyword or 'oil diffuser' in keyword:
                return 'Essential Oil Diffusers'
            elif 'home fragrance' in keyword:
                return 'General Home Fragrances'
            elif 'candle' in keyword:
                return 'Candles & Accessories'
            else:
                return 'Other'

        df['topic_cluster_manual'] = df['keyword'].apply(assign_topic_cluster_manual)
        cluster_counts = df['topic_cluster_manual'].value_counts()
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Topic Cluster'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'topic_cluster_manual']].head(10))

    elif clustering_method == "Rule-Based Grouping":
        st.subheader("Rule-Based Grouping (Based on Keyword Content)")
        def assign_topic_cluster_rules(keyword):
            keyword = keyword.lower()
            if 'refill' in keyword:
                return 'Refills'
            elif 'plug in' in keyword or 'plugin' in keyword:
                return 'Plug-Ins'
            elif 'essential mist' in keyword:
                return 'Essential Mist Diffusers'
            elif 'spray' in keyword:
                return 'Sprays'
            elif 'diffuser' in keyword:
                return 'Diffusers'
            elif 'costco' in keyword:
                return 'Costco Specific'
            elif 'walmart' in keyword:
                return 'Walmart Specific'
            elif 'automatic' in keyword:
                return 'Automatic Dispensers'
            elif 'scent' in keyword or 'fragrance' in keyword:
                return 'Scents & Fragrances'
            else:
                return 'General Air Fresheners'

        df['topic_cluster_rules'] = df['keyword'].apply(assign_topic_cluster_rules)
        cluster_counts = df['topic_cluster_rules'].value_counts()
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Topic Cluster'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'topic_cluster_rules']].head(10))

    elif clustering_method == "Keyword Overlap":
        st.subheader("Grouping Based on Shared Words (Keyword Overlap)")
        stop_words = set(stopwords.words('english'))
        def get_significant_words(keyword):
            keyword = keyword.lower()
            words = re.findall(r'\b\w+\b', keyword)
            return [word for word in words if word not in stop_words]

        df['significant_words'] = df['keyword'].apply(get_significant_words)

        word_to_keywords = defaultdict(list)
        for index, row in df.iterrows():
            for word in row['significant_words']:
                word_to_keywords[word].append(row['keyword'])

        topic_clusters_overlap = {}
        for word, keywords in word_to_keywords.items():
            if len(keywords) > 1:
                cluster_name = f"Cluster with '{word}'"
                for kw in keywords:
                    if kw not in topic_clusters_overlap:
                        topic_clusters_overlap[kw] = cluster_name

        df['topic_cluster_overlap'] = df['keyword'].map(topic_clusters_overlap).fillna('Single Keyword')
        cluster_counts = df['topic_cluster_overlap'].value_counts().head(20)
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Topic Cluster'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'topic_cluster_overlap']].head(10))

    elif clustering_method == "Text Similarity (FuzzyWuzzy)":
        st.subheader("Grouping Based on Text Similarity (FuzzyWuzzy)")
        threshold = st.slider("FuzzyWuzzy Similarity Threshold", 0, 100, 80)
        def group_by_similarity(keywords, threshold):
            clusters = defaultdict(list)
            for i, kw1 in enumerate(keywords):
                if not any(kw1 in cluster for cluster in clusters.values()):
                    clusters[kw1].append(kw1)
                    for j in range(i + 1, len(keywords)):
                        kw2 = keywords[j]
                        similarity_ratio = fuzz.ratio(kw1, kw2)
                        if similarity_ratio >= threshold:
                            clusters[kw1].append(kw2)
            return clusters

        keywords_list = df['keyword'].tolist()
        similarity_clusters = group_by_similarity(keywords_list, threshold)
        df['topic_cluster_similarity_fuzzy'] = df['keyword'].apply(lambda x: next((key for key, value in similarity_clusters.items() if x in value), None))
        cluster_counts = df['topic_cluster_similarity_fuzzy'].value_counts().head(20)
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Topic Cluster'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'topic_cluster_similarity_fuzzy']].head(10))

    elif clustering_method == "Text Similarity (Cosine)":
        st.subheader("Grouping Based on Text Similarity (Cosine Similarity)")
        threshold = st.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.6)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['keyword'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        def group_by_cosine_similarity(keywords, similarity_matrix, threshold):
            clusters = defaultdict(list)
            added = [False] * len(keywords)
            for i in range(len(keywords)):
                if not added[i]:
                    clusters[keywords[i]].append(keywords[i])
                    added[i] = True
                    for j in range(i + 1, len(keywords)):
                        if not added[j] and similarity_matrix[i][j] >= threshold:
                            clusters[keywords[i]].append(keywords[j])
                            added[j] = True
            return clusters

        cosine_clusters = group_by_cosine_similarity(df['keyword'].tolist(), cosine_sim, threshold)
        df['topic_cluster_similarity_cosine'] = df['keyword'].apply(lambda x: next((key for key, value in cosine_clusters.items() if x in value), None))
        cluster_counts = df['topic_cluster_similarity_cosine'].value_counts().head(20)
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Topic Cluster'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'topic_cluster_similarity_cosine']].head(10))

    elif clustering_method == "Using Existing Search Intent":
        st.subheader("Topic Clusters based on Existing 'Search Intent'")
        cluster_counts = df['search intent info main intent'].value_counts()
        fig_cluster = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                             labels={'y': 'Number of Keywords', 'index': 'Search Intent'})
        st.plotly_chart(fig_cluster)
        st.dataframe(df[['keyword', 'search intent info main intent']].head(10))