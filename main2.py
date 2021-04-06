import symbol

import streamlit as st
import pandas as pd
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


sns.set_style('darkgrid')

st.title('Scoff')

st.sidebar.header('User Input Features')


# Web scraping of S&P 500 data
#

@st.cache(allow_output_mutation=True)
def load_data():
    #df = pd.read_csv('scoff.csv')
    #return df
    url = 'https://github.com/azuazyew/data/blob/main/scoff.csv'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df


df = load_data()
sector = df.groupby('user_id')

# Sidebar - Sector selection
sorted_sector_unique = sorted(df['user_id'].unique())
selected_sector = st.sidebar.multiselect('User', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[(df['user_id'].isin(selected_sector))]

st.header('User Result')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(
    df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)


# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download Result CSV</a>'
    return href


st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

def max(symbol):
    df_max = df_selected_sector['verbal mean score'].max()
    return df_max

# Plot Closing Price of Query Symbol
def price_plot(symbol):
    barwidth = 0.25
    user = df_selected_sector['user_id']
    score = df_selected_sector['score 1']
    score1 = df_selected_sector['score 2']
    score2 = df_selected_sector['score 3']
    br1 = np.arange(len(score))
    br2 = [x + barwidth for x in br1]
    br3 = [x + barwidth for x in br2]
    plt.bar(br1, score, color='g', width=barwidth, label='score1')
    plt.bar(br2, score1, color='r', width=barwidth, label='score2')
    plt.bar(br3, score2, color='b', width=barwidth, label='score3')

    plt.xlabel('user', fontweight='bold', fontsize=15)
    plt.ylabel('score', fontweight='bold', fontsize=15)
    plt.xticks([r + barwidth for r in range(len(score))], user)

    return st.pyplot()


def verbalmeanscore(symbol):

    df_selected_sector["verbal mean score"] = ((df_selected_sector["score 1"] + df_selected_sector["score 2"] + df_selected_sector["score 3"] + df_selected_sector["score 4"] + df_selected_sector["score 5"]) / 5).round()

    df_selected_sector["quantitative mean score"] = ((df_selected_sector["score 6"] + df_selected_sector["score 7"] + df_selected_sector["score 8"] + df_selected_sector["score 9"] + df_selected_sector["score 10"]) / 5).round()

    df_selected_sector["aptitude mean score"] = ((df_selected_sector["score 11"] + df_selected_sector["score 12"] + df_selected_sector["score 13"] + df_selected_sector["score 14"] + df_selected_sector["score 15"]) / 5).round()

    df_final = df_selected_sector.drop(['score 1', 'score 2', 'score 3', 'score 4', 'score 5', 'score 6', 'score 7', 'score 8', 'score 9', 'score 10', 'score 11', 'score 12', 'score 13', 'score 14', 'score 15'], axis=1)
    return df_final

df_final = verbalmeanscore(symbol)

def modelbuilding(symbol):
    y = df_final['verbal mean score']
    x = df.drop(['verbal mean score', 'user id', 'quantitative mean score', 'aptitude mean score'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    difference = abs(predictions - y_test)
    difference.mean()

    return predictions

num_company = st.sidebar.slider('Number of users', 1, 2)

if st.button('Show Plots'):

    st.header('user data')
    for i in list(df_selected_sector.user_id)[:num_company]:
        price_plot(i), verbalmeanscore(i), max(i)#modelbuilding(i)
