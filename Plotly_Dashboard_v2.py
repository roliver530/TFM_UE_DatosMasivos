# Import necessary modules
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table


#add in the dataframe
df = pd.read_csv("C:/Users/Ryanw/Desktop/TFM/Results.csv")

#Correct the date
df['date']=pd.to_datetime(df['date'])

#create dataframe for the table


df2 = df[['date','content','Vader_Score_Class','TextBlob_Subj_Class','TextBlob_Score_Class']]

# Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Build sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#E9EDC9",
    'text':'white'
}

# Card content Tweets by Sentiment
card_content1 = [
    dbc.CardBody(
        [
            html.H5("Comment input devices", className="card-title text-center, color:'blue"),
            dcc.Graph(id='sourceLabel', figure=px.pie(
                data_frame=df,
                names='sourceLabel',

            )),

        ]
    )
]

# Card content Tweets by Sentiment
card_content2 = [
    dbc.CardBody(
        [
            html.H5("Tweets by Sentiment", className="card-title text-center"),
            dcc.Graph(id='donut', figure={}),

        ]
    ),
]

# Card content 3
card_content3 = [
    dbc.CardBody(
        [
            html.H5("Subjectivity Score", className="card-title text-center"),
            dcc.Graph(id='subj', figure=px.pie(
                                        data_frame=df,
                                        names="TextBlob_Subj_Class",
                                        hole= .5))
            ]
    )]


# Application layout
app.layout = html.Div(dbc.Container([

    # add navigation bad
    dbc.Card(
        dbc.CardBody(
            [
                html.H5("Navigation Pain", className="display-6 text-center"),
                html.Hr(),
                html.P("Total # of tweets analyzed", className='text-center'),
                html.H4(len(df),className="text-center"),
                html.Hr(),
                html.P("Please select which Topic Modeling function", className='text-center, mb-4'),
                dcc.RadioItems(id='topic_model',
                               options=[
                                   {'label': 'Vector Model', 'value': 'Topics_Vec'},
                                   {'label': 'Cluster Model', 'value': 'Topics_Cluster'},
                                   {'label': 'Dimention Red. Model', 'value': 'Topics_Dim'},
                               ],
                               value='Topics_Dim'),

                html.Hr(),
                html.P("Select which Sentiment Analysis Algorithm", className='text-center, mb-4'),
                dcc.RadioItems(id='sentiment_type',
                               options=[
                                   {'label': 'NLTK VADER', 'value': 'Vader_Score_Class'},
                                   {'label': 'TextBlob Sentiment', 'value': 'TextBlob_Score_Class'},
                               ],
                               value="TextBlob_Score_Class"),

                html.Hr(),

            ]
        ),
        style=SIDEBAR_STYLE,
    ),

#Add title and container
    dbc.Col([html.H1("Reputational Risk Discovery Dashboard")], className='text-center text-primary, mb-4', width=12),
    dbc.Row([


        #dbc.Col(dbc.Card(card_content1, color="primary", outline=True), width=4),
        dbc.Col(dbc.Card(card_content2), width=6),
        dbc.Col(dbc.Card(card_content3), width=6),

    ], className="mb-4"),
    dbc.Row([

        dbc.Col(dcc.Graph(id="bar_topic", figure={}), width=12),

    ]),

    dbc.Row([

        dbc.Col(dcc.Graph(id="topic_history", figure={}), width=12),

    ]),
    dbc.Row([

        dbc.Col(dbc.Card(
            dbc.CardBody([dash_table.DataTable(
            id="table",
            data=df2.to_dict('records'),
            style_data={
                'whiteSpace': 'normal',
                'height':'auto',
            'textOverflow': 'ellipsis'
            },
            sort_mode = 'multi',
            selected_rows=[],
            style_as_list_view=True,
            style_table={'overflowX': 'auto'},
            style_cell={
                    'height': 'auto',
                    # all three widths are needed
                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'whiteSpace': 'normal'
                },
                virtualization=True

        )])),width=12),

    ]),
]),style={'backgroundColor':'2A9D8F',
          })

#update the sentiment analysis chart
@app.callback(
    Output('donut', 'figure'),
    Input('sentiment_type', 'value'))


def update_donut(sentiment):
    dff = df
    donut = px.pie(
        data_frame=dff,
        names=sentiment,
                     )
    return donut


@app.callback(
    Output("bar_topic","figure"),
    Input("topic_model","value"))

def update_bar(topic_model):
    dff = df
    dff = dff.groupby(['date', topic_model], as_index=False)['id'].count()
    fig = px.bar(dff,
                 x=topic_model,
                 y='id',
                 orientation='v',
                 opacity=0.9,
                 barmode='relative',
                 color=topic_model,
                 title='Number of Tweets by Topic',
                 labels={
                     "date": "Dates",
                     "id": "# of Tweets related to Topic",
                     topic_model:"Selected model"
                 },
                )
    return fig


#Topic model
@app.callback(
    Output("topic_history","figure"),
    Input("topic_model","value"))

def update_topic_history(topic_model):
   dff = df
   dff = dff.groupby(['date',topic_model],as_index=False)['id'].count()
   fig = px.line(data_frame=dff, x='date', y='id', color=topic_model, title="Topics by year",
                 labels={
                     "date": "Dates",
                     "id": "# of Tweets related to Topic",
                     "title": "Topics by Month/Year"
                 },
                 )

   return fig



# initialize app
if __name__ == '__main__':
    app.run_server(debug=True)





