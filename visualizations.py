import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import re

from data_loader import get_dataset

@st.cache_data
def reviews_by_aspect(df):
    count_df = df[['aspects', 'sentiment_label']].value_counts().reset_index()
    color_map = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}
    fig = px.bar(count_df, x='aspects', y='count', color='sentiment_label', color_discrete_map=color_map, barmode='stack')
    fig.update_xaxes(
        title_text='Aspects'
    )
    fig.update_yaxes(
        title_text='Number of Reviews'
    )
    fig.update_layout(
        title='Number of Reviews by Aspects'
    )

    return fig


@st.cache_data
def prepare_reviews_over_time(df):
    time_df = df[['reviews_date', 'primaryCategories', 'sentiment_label']].value_counts().reset_index()
    time_df['primaryCategories'] = time_df['primaryCategories'].str.split(',')
    time_df = time_df.explode('primaryCategories')
    time_df['primaryCategories'] = time_df['primaryCategories'].str.strip()
    time_df = time_df[(time_df['reviews_date'] > '2016-01-01') & (time_df['reviews_date'] <= '2016-12-31')]
    time_df = time_df.sort_values('reviews_date')

    return time_df


def reviews_over_time(df):
    sentiment = st.selectbox('Select Sentiment', options=['All', 'Negative', 'Neutral', 'Positive'])

    time_df = prepare_reviews_over_time(df)

    color_map = {'Health & Beauty': 'red', 'Electronics': 'blue', 'Media': 'green', 'Toys & Games': 'black', 'Office Supplies': 'orange'}
    
    if sentiment == 'All':
        fig = px.line(time_df, x='reviews_date', y='count', color='primaryCategories', color_discrete_map=color_map, markers=True)
        fig.update_layout(
            title='Number of Reviews Over Time by Primary Categories'
        )
    
    else:
        fig = px.line(time_df[time_df['sentiment_label'] == sentiment], x='reviews_date', y='count', color='primaryCategories',
                      color_discrete_map=color_map, markers=True)
        fig.update_layout(
            title=f'Number of {sentiment} Reviews Over Time by Primary Categories'
        )

    fig.update_traces(mode='markers+lines')
    fig.update_xaxes(
        title_text='Date (year 2016)'
    )
    fig.update_yaxes(
        title_text='Number of Reviews'
    )

    return fig


@st.cache_data
def reviews_by_day_month(df):
    date_df = df.copy()
    date_df['month'] = date_df['reviews_date'].dt.month
    date_df['day'] = date_df['reviews_date'].dt.day
    date_df['dof'] = date_df['reviews_date'].dt.day_name()
    date_df['hour'] = date_df['reviews_date'].dt.hour

    month_day_df = date_df[['month', 'day', 'sentiment_label']].value_counts().reset_index()
    color_map = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}
    fig = px.scatter(month_day_df,x='month', y='day', color='sentiment_label', size='count', size_max=40, color_discrete_map=color_map)
    fig.update_layout(
        width=1000,
        height=1000,
        title='Number of Reviews by Day and Month'
    )
    fig.update_xaxes(
        dtick=1,
        tick0=1,
        range=[0.5, 12.5],
        title_text="Month",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(180, 180, 180, 0.1)'
    )
    fig.update_yaxes(
        dtick=1,
        tick0=1,
        range=[0.5, 31.5],
        title_text="Day",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(180, 180, 180, 0.1)"
    )

    return fig


@st.cache_data
def reviews_by_dof_and_sentiment(df):
    date_df = df.copy()
    date_df['month'] = date_df['reviews_date'].dt.month
    date_df['day'] = date_df['reviews_date'].dt.day
    date_df['dof'] = date_df['reviews_date'].dt.day_name()
    date_df['hour'] = date_df['reviews_date'].dt.hour

    dof_sentiment_df = date_df[['dof','sentiment_label']].value_counts().reset_index()
    color_map = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}
    fig = px.bar(dof_sentiment_df, x='dof', y='count', color='sentiment_label', color_discrete_map=color_map, barmode='stack')
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis={'categoryorder':'total descending'},
        template='plotly_white',
        height=600,
        title='Number of Reviews by Sentiment and Day of Week'
    )
    fig.update_xaxes(
        title_text='Day of Week'
    )
    fig.update_yaxes(
        title_text='Number of Reviews'
    )

    return fig


def reviews_table(df):
    products_df = df[['id', 'sentiment_label']].value_counts().reset_index(name='count')

    original_df = pd.read_parquet('original_data.parquet')
    original_df = original_df[['id', 'name']].drop_duplicates()

    products_df = products_df.merge(original_df, on='id', how='left')

    sentiments = st.multiselect('Select Sentiments', options=products_df['sentiment_label'].unique(), default=products_df['sentiment_label'].unique())

    filtered_df = products_df[products_df['sentiment_label'].isin(sentiments)].sort_values(by='count', ascending=False)

    row_colors = ['#2a2a2a' if i % 2 == 0 else '#1f1f1f' for i in range(len(filtered_df))]

    fig = go.Figure(data=[go.Table(header=dict(values=['Product', 'Number of Reviews'], fill_color='#1f1f1f', font=dict(color='white', size=14), align='left'),
                                   cells=dict(values=[filtered_df['name'], filtered_df['sentiment_label'], filtered_df['count']], fill_color=[row_colors, row_colors, row_colors], font=dict(color='white', size=12), align='left'))
                                   ])
    
    return fig


def parse_percentage(value):
    if isinstance(value, str):
        match = re.search(r'([+-]?\d+(\.\d+)?)%', value)
        if match:
            return float(match.group(1))
        else:
            return None
    return value


def reviews_percentage_diff(products_or_categories):
    if products_or_categories == 'Products':
        df = pd.read_parquet('processed_reviews_products_count.parquet')
        available_tops = ['Top 10 Highest', 'Top 10 Lowest', 'All Products']
        tops = st.selectbox('Choose top of products to display', available_tops)

    elif products_or_categories == 'Categories':
        df = pd.read_parquet('processed_reviews_categories_count.parquet')

    else:
        st.write('Plese select an option')
        return None

    available_sentiments = ['Pos/Neg Percentage', 'Pos/Neu Percentage', 'Neg/Neu Percentage', 'Pos/All Percentage', 'Neg/All Percentage']
    sentiments = st.selectbox('Choose a sentiment', available_sentiments)

    sentiments = [sentiments]

    if not sentiments:
        st.warning('Please pick one')
        return None

    df_plot = df.copy()

    for col in sentiments:
        df_plot[col + '_num'] = df_plot[col].apply(parse_percentage)

    if products_or_categories == 'Products':
        df_plot = df_plot.melt(id_vars='name', value_vars=[s + '_num' for s in sentiments], var_name='sentimentMetric', value_name='value')
    elif products_or_categories == 'Categories':
        df_plot = df_plot.melt(id_vars='primaryCategories', value_vars=[s + '_num' for s in sentiments], var_name='sentimentMetric', value_name='value')

    df_plot['sentimentMetric'] = df_plot['sentimentMetric'].str.replace('_num', '')

    df_plot['color'] = df_plot['value'].apply(lambda x: 'green' if x >= 0 else 'red')

    if products_or_categories == 'Products':
        if tops == 'Top 10 Highest':
            df_plot = df_plot.sort_values(by='value', ascending=False).groupby('sentimentMetric').head(10)
        elif tops == 'Top 10 Lowest':
            df_plot = df_plot.sort_values(by='value', ascending=True).groupby('sentimentMetric').head(10)
        
        fig = px.bar(df_plot, x='name', y='value', color='color', color_discrete_map={'green': 'green', 'red': 'red'},
                 facet_col='sentimentMetric', text=df_plot['value'])
        
        fig.update_xaxes(
            title_text='Products',ticktext=[n[:40] for n in df_plot['name']], tickvals=df_plot['name']
        )

    elif products_or_categories == 'Categories':
        df_plot = df_plot.sort_values(by='value', ascending=False)
        fig = px.bar(df_plot, x='primaryCategories', y='value', color='color',
                     color_discrete_map={'green': 'green', 'red': 'red'},facet_col='sentimentMetric', text=df_plot['value']
                     )
        
        fig.update_xaxes(
            title_text='Categories'
        )

    fig.update_yaxes(
        title_text='Percentage Difference'
    )
    fig.update_layout(
        title='Percentage Difference Between Sentiments',
        width=1000,
        height=1000
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

    print(df_plot)

    return fig