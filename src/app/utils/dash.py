import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import dash
from dash import dcc
from dash import html
import plotly.express as px

import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#F1FFFE",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.Img(src='./img/amoeb.png', style={'height': '20%', 'width': '100%'}),
        html.P("Albert Rehnberg", className=""),
        html.Hr(),
        html.P(
            "Central Demand Forecast", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("View Data", href="/PD", active="exact"),
                dbc.NavLink("Clusters", href="/C", active="exact"),
                dbc.NavLink("Features SHAP", href="/BF", active="exact"),
                dbc.NavLink("Regression Models", href="/RO", active="exact"),
                dbc.NavLink("Prediction", href="/PR", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("url", "pathname")]
)
def render_content(value):
    if value == "/PD":
        return [
            html.H3('Energy Demad 2017-18'),
            dcc.Graph(
                id='Hourly Power',
                figure={'data': [
                    {'x': cluster.index, 'y': cluster['Power_kW'], 'type': 'line', 'name': 'power'},
                ],
                    'layout': {
                        'title': 'Central Power Consumption (Kw)'
                    }
                }
            ),
        ]


    elif value == "/C":
        return html.Div([
            html.Div([
                dbc.Tabs([
                    dbc.Tab(label="Power Vs Temp", tab_id="temp_C"),
                    dbc.Tab(label="Power Vs Humidity", tab_id="HR"),
                    dbc.Tab(label="Power Vs Holiday", tab_id="windSpeed_m/s"),
                    dbc.Tab(label="Power Vs Solar Radiation", tab_id="solarRad_W/m2"),
                    dbc.Tab(label="Power Vs Hour", tab_id="Hour"),
                    dbc.Tab(label="Power Vs Previous Hour Power", tab_id="Power-1"),
                ],
                    id="tabs",
                    active_tab="temp_C",
                ),

                html.Div([
                    dcc.Graph(id='cluster_graph')
                ])

            ])
        ]),
    elif value == "/BF":
        return html.Div([
            html.H3("Feature Impact Analysis"),
            html.Hr(),
            html.Img(src='assets/shaap.png', style={"width": "70%"}),
        ], style={"margin-left": "30%"}),
    elif value == "/RO":
        return html.Div([html.H3("Regression Model Metrics"),
                         html.Hr(),
                         html.Div([
                             dcc.RadioItems(
                                 id='RR',
                                 options=[
                                     {'label': 'Random Forest', 'value': 'RF'},
                                     {'label': 'Extreme Gradient Boosting', 'value': 'EGB'},
                                     {'label': 'Neural Network', 'value': 'NN'}
                                 ],
                                 value='RF',
                                 labelStyle={'display': 'block'}
                             ),

                             html.Hr(),
                             html.Div(id='output')

                         ])
                         ]),
    elif value == "/PR":
        return html.Div([html.H3("Prediction Metrics"),
                         html.Hr(),

                         html.Div([
                             dcc.RadioItems(
                                 id='RR1',
                                 options=[
                                     {'label': 'Random Forest', 'value': 'RF'},
                                     {'label': 'Extreme Gradient Boosting', 'value': 'EGB'},
                                     {'label': 'Neural Network', 'value': 'NN'}
                                 ],
                                 value='RF',
                                 labelStyle={'display': 'block'}
                             ),
                             html.Hr(),
                             html.Div(id='output1')

                         ])
                         ]),


@app.callback(
    dash.dependencies.Output('cluster_graph', 'figure'),
    [dash.dependencies.Input('tabs', 'active_tab')])
def update_figure(active_tab):
    return px.scatter(cluster, x="Power_kW", y=active_tab, color="cluster", hover_name="cluster")


@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('RR', 'value')])
def update_figure(value):
    if value == 'RF':
        return html.Div([
            html.H3('Random Forest Regression'),
            html.Br(),
            html.Img(src='assets/RandomForest.png'),
            html.Img(src='assets/scatterRandomForest.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 15.191"),
            html.Br(),
            html.P('Mean Squared Error =1139.72'),
            html.Br(),
            html.P('Root Mean Squared Error =33.75'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error =0.16'),
        ])
    elif value == 'EGB':
        return html.Div([
            html.H3('Extreme Gradient Boosting'),
            html.Br(),
            html.Img(src='assets/EGB.png'),
            html.Img(src='assets/scatterEGB.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 16.66"),
            html.Br(),
            html.P('Mean Squared Error =34.44'),
            html.Br(),
            html.P('Root Mean Squared Error = 19.28'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error =0.16'),
        ])
    elif value == 'NN':
        return html.Div([
            html.H3('Neural Network'),
            html.Br(),
            html.Img(src='assets/NeuralN.png'),
            html.Img(src='assets/scatterNeuralN.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 25.37"),
            html.Br(),
            html.P('Mean Squared Error = 2213.46'),
            html.Br(),
            html.P('Root Mean Squared Error = 47.04'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error = 0.23'),
        ])


@app.callback(
    dash.dependencies.Output('output1', 'children'),
    [dash.dependencies.Input('RR1', 'value')])
def update_figure(value):
    if value == 'RF':
        return html.Div([
            html.H3('Random Forest Regression'),
            html.Br(),
            html.Img(src='assets/RandomForest19.png'),
            html.Img(src='assets/scatterRandomForest19.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 11.86"),
            html.Br(),
            html.P('Mean Squared Error =582.64'),
            html.Br(),
            html.P('Root Mean Squared Error =24.13'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error =0.12'),
        ])
    elif value == 'EGB':
        return html.Div([
            html.H3('Extreme Gradient Boosting'),
            html.Br(),
            html.Img(src='assets/EGB19.png'),
            html.Img(src='assets/scatterEGB19.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 16.97"),
            html.Br(),
            html.P('Mean Squared Error = 836.39'),
            html.Br(),
            html.P('Root Mean Squared Error = 19.28'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error = 0.24'),
        ])
    elif value == 'NN':
        return html.Div([
            html.H3('Neural Network'),
            html.Br(),
            html.Img(src='assets/NeuralN19.png'),
            html.Img(src='assets/scatterNeuralN19.png'),
            html.Hr(),
            html.P("Mean Absolute Error = 14.56"),
            html.Br(),
            html.P('Mean Squared Error = 616.88'),
            html.Br(),
            html.P('Root Mean Squared Error = 24.84'),
            html.Br(),
            html.P('Cross Validated Root Mean Squared Error = 0.19'),
        ])


if __name__ == '__main__':
    app.run_server(debug=False, port=9050)
