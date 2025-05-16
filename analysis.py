# sourcery skip: pandas-avoid-inplace
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import pytz
from datetime import datetime
from textblob import TextBlob # type: ignore
import nltk
import webbrowser

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load data
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')

# Data cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column] = apps_df[column].fillna(apps_df[column].mode()[0])
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df = reviews_df.dropna(subset=['Translated_Review'])

apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].astype(str).str.replace('$', '').astype(float)
apps_df['Reviews'] = apps_df['Reviews'].astype(str).str.replace(',', '').astype(int)

# Convert Size to MB
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan
apps_df['Size'] = apps_df['Size'].astype(str).apply(convert_size)

# Add log columns
apps_df['log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['log_Reviews'] = np.log1p(apps_df['Reviews'])

# Add rating group
def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'
apps_df['Rating_group'] = apps_df['Rating'].apply(rating_group)

# Add revenue column
apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Dashboard HTML setup
html_files_path = "./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers = ""
plot_width = 400
plot_height = 300
plot_bgcolor = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}

def save_plt_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
       <div class="plot">{html_content}</div>
       <div class="insight">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

# Figure 1: Top Categories
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on PLay Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width,
    height=plot_height
)
fig1.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig1, "Category Graph 1.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity")

# Figure 2: Type Pie
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='Top Categories on Play Store',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=plot_width,
    height=plot_height
)
fig2.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig2, "Type Graph 2.html", "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.")

# Figure 3: Rating Histogram
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=plot_width,
    height=plot_height
)
fig3.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10),
    bargap=0.2
)
save_plt_as_html(fig3, "Rating Graph 3.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favourably by users.")

# Figure 4: Sentiment Distribution
sentiment_counts = reviews_df['Sentiment_score'].value_counts()
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width,
    height=plot_height
)
fig4.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig4, "Sentiment Graph 4.html", "Sentiment in reviews show a mix of positivity and negativity, with a slight lean towards")

# Figure 5: Installs by Category
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=plot_width,
    height=plot_height
)
fig5.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig5, "Install Graph 5.html", "The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage.")

# Figure 6: Updates per Year
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df = apps_df.dropna(subset=['Last Updated'])
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig6, "Updates Graph 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# Figure 7: Revenue by Category
revenue_per_year = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width,
    height=plot_height
)
fig7.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig7, "Revenue Graph 7.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential")

# Figure 8: Top Genres
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=plot_width,
    height=plot_height
)
fig8.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig8, "Genre Graph 8.html", "Action and Casual genres are the most common, reflecting user's preference for engaging and easy-to-play games.")

# Figure 9: Last Updated vs Rating
fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Updated on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=plot_width,
    height=plot_height
)
fig9.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig9, "Update Graph 9.html", "The Scatter plot shows a weak correlation between the last updated and ratings, suggesting that more frequent updates dont always result in better ratings.")

# Figure 10: Paid vs Free Ratings
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for paid vs Free apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width,
    height=plot_height
)
fig10.update_layout(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=plot_bgcolor,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plt_as_html(fig10, "Paid Free Graph 10.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for")

# Dashboard HTML
dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            background: #222;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .plot {{
            flex: 1 1 auto;
        }}
        .insight {{
            margin-top: 8px;
            background-color: rgba(0,0,0,0.7);
            padding: 6px 10px;
            border-radius: 5px;
            color: #fff;
            font-size: 0.95em;
            min-height: 32px;
            display: block;
            position: static;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plot_containers}
    </div>
</body>
</html>
"""

dashboard_path = os.path.join(html_files_path, "web_page.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(dashboard_html)
webbrowser.open('file://' + os.path.realpath(dashboard_path))

# INTERNSHIP TASK 1: Sentiment Distribution Stacked Bar Chart
filtered_apps = apps_df[apps_df['Reviews'] > 1000]
top_categories = filtered_apps['Category'].value_counts().nlargest(5).index
filtered_apps = filtered_apps[filtered_apps['Category'].isin(top_categories)]
merged = pd.merge(filtered_apps, reviews_df, left_on='App', right_on='App', how='inner')

def sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
merged['Sentiment_Label'] = merged['Sentiment_score'].apply(sentiment_label)

def rating_group_task1(rating):
    if rating >= 4:
        return '4-5 stars'
    elif rating >= 3:
        return '3-4 stars'
    else:
        return '1-2 stars'
merged['Rating_Group'] = merged['Rating'].apply(rating_group_task1)

grouped = merged.groupby(['Category', 'Rating_Group', 'Sentiment_Label']).size().reset_index(name='Count')
pivot = grouped.pivot_table(index=['Category', 'Rating_Group'], columns='Sentiment_Label', values='Count', fill_value=0)

fig, ax = plt.subplots(figsize=(12, 7))
colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
for sentiment in ['Positive', 'Neutral', 'Negative']:
    pivot[sentiment] = pivot.get(sentiment, 0)
pivot = pivot[['Positive', 'Neutral', 'Negative']]
for i, category in enumerate(top_categories):
    cat_data = pivot.loc[category]
    cat_data.plot(kind='bar', stacked=True, color=[colors['Positive'], colors['Neutral'], colors['Negative']], ax=ax, position=i, width=0.15, legend=False)
ax.set_title('Sentiment Distribution by Rating Group and Top 5 Categories')
ax.set_xlabel('Category, Rating_Group')
ax.set_ylabel('Number of Reviews')
ax.legend(['Positive', 'Neutral', 'Negative'], title='Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# INTERNSHIP TASK 2: Choropleth Map (6 PM - 8 PM IST)
if 'Country' not in apps_df.columns:
    apps_df['Country'] = 'United States'
filtered = apps_df[~apps_df['Category'].str.startswith(('A', 'C', 'G', 'S'))]
top5_categories = filtered.groupby('Category')['Installs'].sum().nlargest(5).index
filtered = filtered[filtered['Category'].isin(top5_categories)]
agg = filtered.groupby(['Country', 'Category'])['Installs'].sum().reset_index()
agg['Highlight'] = agg['Installs'] > 1_000_000
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
show_map = 18 <= now_ist.hour < 20
print(f"Current IST hour: {now_ist.hour}:{now_ist.minute}")
if show_map:
    fig = px.choropleth(
        agg,
        locations='Country',
        locationmode='country names',
        color='Installs',
        hover_name='Category',
        animation_frame='Category',
        color_continuous_scale='Viridis',
        title='Global Installs by Category (Top 5, Excluding A/C/G/S)',
        labels={'Installs': 'Total Installs'},
    )
    fig.update_traces(marker_line_width=agg['Highlight'].map({True: 3, False: 0.5}))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size': 16},
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.show()
else:
    print("Choropleth map is only available between 6 PM and 8 PM IST.")

# INTERNSHIP TASK 3: Bubble Chart (5 PM - 7 PM IST)
reviews_df['Subjectivity'] = reviews_df['Translated_Review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
avg_subjectivity = reviews_df.groupby('App')['Subjectivity'].mean().reset_index()
apps_df = pd.merge(apps_df, avg_subjectivity, left_on='App', right_on='App', how='left')
categories = ['GAME', 'BEAUTY', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'ENTERTAINMENT', 'SOCIAL', 'EVENT']
filt = (
    (apps_df['Rating'] > 3.5) &
    (apps_df['Category'].str.upper().isin(categories)) &
    (apps_df['Reviews'] > 500) &
    (~apps_df['App'].str.contains('S', case=False)) &
    (apps_df['Subjectivity'] > 0.5) &
    (apps_df['Installs'] > 50000)
)
filtered = apps_df.loc[filt].copy()
def translate_category(cat):
    if cat.upper() == 'BEAUTY':
        return 'सौंदर्य'
    elif cat.upper() == 'BUSINESS':
        return 'வணிகம்'
    elif cat.upper() == 'DATING':
        return 'Dating'
    else:
        return cat
filtered['Category_Display'] = filtered['Category'].apply(translate_category)
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
show_bubble = 17 <= now_ist.hour < 19
if show_bubble:
    fig = px.scatter(
        filtered,
        x='Size',
        y='Rating',
        size='Installs',
        color='Category_Display',
        hover_name='App',
        title='App Size vs. Average Rating (Bubble size = Installs)',
        labels={'Size': 'App Size (MB)', 'Rating': 'Average Rating', 'Installs': 'Number of Installs'},
        size_max=60,
    )
    for trace in fig.data:
        if 'GAME' in trace.name.upper():
            trace.marker.color = 'pink'
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size': 16},
        xaxis=dict(title_font={'size': 12}),
        yaxis=dict(title_font={'size': 12}),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.show()
else:
    print("Bubble chart is only available between 5 PM and 7 PM IST.")