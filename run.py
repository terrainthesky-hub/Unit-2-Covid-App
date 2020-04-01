# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app, server
from pages import index, predictions, insights, process
import dash
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import os
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import dash_core_components as dcc
import category_encoders as ce

path = Path('COVID-19 Cases (1).csv')

df = pd.read_csv(r'C:\Users\lesle\Desktop\covid\COVID-19 Cases (1).csv')

df['Date'] = pd.to_datetime(df['Date']) 
df['Date_year'] = df['Date'].dt.year
df['Date_month'] = df['Date'].dt.month
df['Date_day'] = df['Date'].dt.day
df['Latest_Date'] = pd.to_datetime(df['Date'])
df['Latest_Date_year'] = df['Latest_Date'].dt.year
df['Latest_Date_month'] = df['Latest_Date'].dt.month
df['Latest_Date_day'] = df['Latest_Date'].dt.day
df['Province_State'] = df['Province_State'].replace({np.NaN: 'Missing'})

train = df[df['Date'] < '2020-03-20']
test = df[df['Date'] >= '2020-03-20']

target = 'Cases'
#Dropping data leakage of sales from other countries



X_train = train.drop(columns=['Cases', 'Date', 'Latest_Date', 'Admin2', 'Combined_Key', 'FIPS'])
y_train = train[target]
X_test = test.drop(columns=['Cases', 'Date', 'Latest_Date', 'Admin2', 'Combined_Key', 'FIPS'])
y_test = test[target]


import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor


pipeline = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    AdaBoostRegressor(n_estimators=200, random_state=42)
    
)


pipeline.fit(X_train, y_train)

y_test = y_test.fillna(y_test.mean())


y_pred = pipeline.predict(X_test)
r2_score(y_test, y_pred)

test['predicted_cases'] = pd.DataFrame(y_pred)
test_california = test[test['Province_State'].str.contains('California') & (test['Case_Type'].str.contains('Confirmed'))]

import plotly.graph_objects as go

fig1 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_california['Date'], y=test_california['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_california['Date'], y=test_california['Cases'])
          ])
# Change the bar mode
fig1.update_layout(barmode='group')

test_new_york = test[test['Province_State'].str.contains('New York') & (test['Case_Type'].str.contains('Confirmed'))]

import plotly.graph_objects as go
fig2 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_new_york['Date'], y=test_new_york['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_new_york['Date'], y=test_new_york['Cases'])
          ])
# Change the bar mode
fig2.update_layout(barmode='group')

test_florida = test[test['Province_State'].str.contains('Florida') & (test['Case_Type'].str.contains('Confirmed'))]
fig3 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_florida['Date'], y=test_florida['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_florida['Date'], y=test_florida['Cases'])
          ])
# Change the bar mode
fig3.update_layout(barmode='group')

test_tenessee = test[test['Province_State'].str.contains('Tennessee') & (test['Case_Type'].str.contains('Confirmed'))]
fig4 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_tenessee['Date'], y=test_tenessee['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_tenessee['Date'], y=test_tenessee['Cases'])
          ])
fig4.update_layout(barmode='group')

test_colorado = test[test['Province_State'].str.contains('Colorado') & (test['Case_Type'].str.contains('Confirmed'))]
fig5 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_colorado['Date'], y=test_colorado['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_colorado['Date'], y=test_colorado['Cases'])
          ])
fig5.update_layout(barmode='group')
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    html.H1(children='Predicted vs Actual Cases of Corona Virus'),
    dcc.Dropdown(id='demo-dropdown', 
        options=[{'label': 'California', 'value': 'Cali'}, 
                 {'label': 'New York', 'value': 'NY'},
                 {'label': 'Florida', 'value': 'FL'},
                 {'label': 'Tenessee', 'value': 'TN'},
                 {'label': 'Colorado', 'value': 'CO'}
                ]
    ),
    html.Div(id='my-div', children='''
        Charts will go here
    '''),
])
@app.callback(
    Output(component_id='my-div', component_property='children'), 
    [Input(component_id='demo-dropdown', component_property='value')]
)
def update_chart(dropdown_option):
    if dropdown_option == 'Cali':
        return dcc.Graph(figure=fig1)
    if dropdown_option == 'NY':
        return dcc.Graph(figure=fig2)
    if dropdown_option == 'FL':
        return dcc.Graph(figure=fig3)
    if dropdown_option == 'TN':
        return dcc.Graph(figure=fig4)
    if dropdown_option == 'CO':
        return dcc.Graph(figure=fig5)

    
if __name__ == '__main__':
    app.run_server(debug=True)
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
    
#     dcc.Dropdown(id='demo-dropdown', 
#         options=[{'label': 'California', 'value': 'Cali'}, 
#                  {'label': 'New York', 'value': 'NY'}
#                 ]
#     ),
# ])
# @app.callback(
#     Output(component_id='my-div', component_property='children'), 
#     [Input(component_id='demo-dropdown', component_property='value')]
# )


# def update_chart(dropdown_option):
#     if dropdown_option == 'Cali':
#         return dcc.Graph(figure=fig1)
#     if dropdown_option == 'NY':
#         return dcc.Graph(ifigure=fig2)
        
        
# if __name__ == '__main__':
#     app.run_server(debug=True)
