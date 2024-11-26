from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import requests
import json
from loguru import logger
import os

from src.const import get_constants
from src.dash1 import generate_visualizations as generate_visualizations1
from src.dash2 import generate_visualizations as generate_visualizations2
from src.dash3 import generate_visualizations as generate_visualizations3
# from src.dash4 import generate_visualizations as generate_visualizations4

HA = pd.read_csv('data/HeartAttack.csv')
HA['Creatina']=pd.to_numeric(HA['Creatina en suero'].str.replace(",", "."))
num_pacientes,edad_promedio,tiempo_seguimiento,factores_de_riesgo = get_constants(HA)

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Factores de Riesgo asociados a E.C')
server = app.server

# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://{}:8001/api/v1/predict".format(api_url)


def generate_stats_card(title, value, image_path):
    return html.Div(
        dbc.Card([
            dbc.CardImg(src=image_path, top=True, style={'width': '55px','alignSelf': 'center'}),
            dbc.CardBody([
                html.P(value, className="card-value", style={'margin': '0px','fontSize': '22px','fontWeight': 'bold'}),
                html.H4(title, className="card-title", style={'margin': '0px','fontSize': '12px','fontWeight': 'bold'})
            ], style={'textAlign': 'center'}),
        ], style={'paddingBlock':'10px',"backgroundColor":'#8889a9','border':'none','borderRadius':'10px','color':'white'})
    )


tab_style = {
    'idle':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'backgroundColor': '#8889a9',
        'border':'none',
        'color':'white'
    },
    'active':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'border':'none',
        'backgroundColor': '#dcdcdc',
        'color':'white'
    }
}

drop_down_stye= {        
    'display':'flex',
    'alignItems':'center',
    'justifyContent':'center',
    'fontWeight': 'bold',
    'backgroundColor': '#8889a9',
    'border':'none',
    'color':'black',
    'height':'35px',
    'width': '195px',
    'textAlign': 'center'
}


titles_dropwdown = {
    'color':'white',
    'fontWeight': 'bold',
    'fontSize':'10px'
}


MAX_OPTIONS_DISPLAY = 3300
fig1, fig2, fig3 = generate_visualizations1(HA)

# Define the layout of the app
app.layout = html.Div([
    dcc.Textarea(
    id='textarea-title',
    value='Factores de Riesgo asociados a Enfermedades Cardíacas',
    style={'height':'45px','textAlign': 'center','width': '100%',"backgroundColor":'#2e304b','border':'none','color':'white','fontWeight': 'bold','fontSize': '32px'},
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src="./assets/HA_banner.png",width=150), width=2),
            dbc.Col(
                dcc.Tabs(id='graph-tabs', value='Demográfica', children=[
                    dcc.Tab(label='Demográfica', value='Demográfica',style=tab_style['idle'],selected_style=tab_style['active']),
                    dcc.Tab(label='Antecedentes', value='Antecedentes',style=tab_style['idle'],selected_style=tab_style['active']),
                    dcc.Tab(label='Factores', value='Factores',style=tab_style['idle'],selected_style=tab_style['active'])
                ], style={'marginTop': '15px', 'width':'800px','height':'50px'})
            ,width=6)
        ]),
        dbc.Row([
            dbc.Col(generate_stats_card("Pacientes",num_pacientes,"./assets/paciente_icon.png"), width=3),
            dbc.Col(generate_stats_card("Edad Promedio", edad_promedio,"./assets/edad_icon.png"), width=3),
            dbc.Col(generate_stats_card("Tiempo Seguimiento (días)",tiempo_seguimiento,"./assets/tiempo_seguimiento_icon.png"), width=3),
            dbc.Col(generate_stats_card("Factores de Riesgo analizados",factores_de_riesgo,"./assets/factores_de_riesgo_icon.png"), width=3),
        ],style={'marginBlock': '10px'}),
    dcc.Textarea(
            id='textarea-pred',
            value='Ingresa algunos datos y evalúa tu salud cardíaca:',
            style={'height':'25px','textAlign': 'left','width': '100%',"backgroundColor":'#2e304b','border':'none','color':'white','fontWeight': 'bold','fontSize':'14px'},
    ),
        # DropDowns 1
        dbc.Row([
            dbc.Col(html.Div(["Edad",
                    dcc.Input(id='edad',value='50',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Anemia",
                    dcc.Dropdown(id='anemia',value='No',options=['Sí', 'No'],style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Creatinina",
                    dcc.Input(id='creatinine_phosphokinase',value='155',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Diabetes",
                    dcc.Dropdown(id='diabetes',value='No',options=['Sí', 'No'],style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Fracción de Eyección",
                    dcc.Input(id='ejection_fraction',value='25',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Presión Sanguínea Alta",
                    dcc.Dropdown(id='high_blood_pressure',value='No',options=['Sí', 'No'],style=drop_down_stye)]), width = 2, style= titles_dropwdown)
        ],style={'marginBlock': '5px'}),
        # DropDowns 2
        dbc.Row([
            dbc.Col(html.Div(["Plaquetas",
                    dcc.Input(id='platelets',value='258000',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Creatinina sérica",
                    dcc.Input(id='serum_creatinine',value='0',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Sodio",
                    dcc.Input(id='serum_sodium',value='13',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Género",
                    dcc.Dropdown(id='sex',value='Mujer',options=['Hombre', 'Mujer'],style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Fumador",
                    dcc.Dropdown(id='smoking',value='No',options=['Sí', 'No'],style=drop_down_stye)]), width = 2, style= titles_dropwdown),
            dbc.Col(html.Div(["Última revisión (días)",
                    dcc.Input(id='time',value='5',type='number',style=drop_down_stye)]), width = 2, style= titles_dropwdown)
        ],style={'marginBlock': '10px'}),
        dbc.Row([
            html.Div(id='resultado')
        ], style={'height':'25px','textAlign': 'left','width': '100%',"backgroundColor":'#2e304b','border':'none','color':'grey','fontWeight': 'bold','fontSize':'10px'}),
        dcc.Textarea(
            id='textarea-descriptivos',
            value='Consulta los datos de la población de estudio',
            style={'height':'25px','textAlign': 'left','width': '100%',"backgroundColor":'#2e304b','border':'none','color':'white','fontWeight': 'bold','fontSize':'14px'},
    ),
        dbc.Row([
            dcc.Loading([
                html.Div(id='tabs-content')
            ],type='default',color='#2e304b')
        ])
    ], style={'padding': '0px'})
],style={'backgroundColor': '#2e304b', 'minHeight': '100vh'})

map_si_no = {'No':0, 'Sí':1}
map_hombre_mujer = {'Mujer':0, 'Hombre':1}

# Method to update prediction
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [Input(component_id='edad', component_property='value'), 
     Input(component_id='anemia', component_property='value'), 
     Input(component_id='creatinine_phosphokinase', component_property='value'), 
     Input(component_id='diabetes', component_property='value'),
     Input(component_id='ejection_fraction', component_property='value'),
     Input(component_id='high_blood_pressure', component_property='value'),
     Input(component_id='platelets', component_property='value'),
     Input(component_id='serum_creatinine', component_property='value'),
     Input(component_id='serum_sodium', component_property='value'),
     Input(component_id='sex', component_property='value'),
     Input(component_id='smoking', component_property='value'),
     Input(component_id='time', component_property='value'),]
)

def update_output_div(edad, anemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    myreq = {
        "inputs": [
            {
                "age": int(edad),
                "anaemia": map_si_no[anemia],
                "creatinine_phosphokinase": int(creatinine_phosphokinase),
                "diabetes": map_si_no[diabetes],
                "ejection_fraction": int(ejection_fraction),
                "high_blood_pressure": map_si_no[high_blood_pressure],
                "platelets": int(platelets),
                "serum_creatinine": int(platelets),
                "serum_sodium": int(serum_sodium),
                "sex": map_hombre_mujer[sex],
                "smoking": map_si_no[smoking],
                "time": int(time)
            }
        ]
      }
    headers =  {"Content-Type":"application/json", "accept": "application/json"}
    
    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()
    logger.info("Response: {}".format(data))

    # Pick result to return from json format
    nr = round(data["predictions"][0],2)
    if nr >= 0.9:
        result = f"{nr*100}% Alto riesgo cardíaco, consulte a su médico."
    elif nr >= 0.6: 
        result = f"{nr*100}% Riesgo Moderado Alto, consulte a su médico."
    elif nr >= 0.3:
        result = f"{nr*100}% Riesgo Moderado Bajo, continúe trabajando por su salud."
    else:
        result = f"{nr*100}% Riesgo Bajo, felicidades."
    
    return result 

# Method to change Tab
@app.callback(
    Output('tabs-content', 'children'),
    [Input('graph-tabs', 'value')]
)
def update_tab(tab):
    if tab == 'Demográfica':
        fig1, fig2, fig3 = generate_visualizations1(HA)
        return html.Div([
        html.Div([
            dcc.Graph(id='graph1', figure=fig1),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph2', figure=fig2),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph3', figure=fig3),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([dcc.Textarea(
        id='textarea-demografica',
        value='''
        
        

        
        La edad promedio de los pacientes que se les realizan seguimiento es de 61 años, el rango de edad esta entre los 45 a los 95 años. El 64.9% son hombres, y el 35.1% restantes son mujeres. De los 299 pacientes a lo que se le realizo el seguimiento, 203 han sobrevivido a una insuficiencia cardiaca, y desafortunadamente 96 pacientes fallecieron por esta causa.''',
        style={'height':'450px','textAlign': 'center','width': '100%',"backgroundColor":'#8889a9','border':'none','color':'white','fontSize':'14px', 'display':'flex'},
        )], style={'width': '50%', 'display': 'inline-block', 'justifyContent':'center'})
    ])
    elif tab == 'Antecedentes':
        fig1, fig2, fig3, fig4, fig5 = generate_visualizations2(HA)
        return html.Div([
        html.Div([
            dcc.Graph(id='graph1', figure=fig1),
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph2', figure=fig2),
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph3', figure=fig3),
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph4', figure=fig4),
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph5', figure=fig5),
        ], style={'width': '100%', 'display': 'inline-block'})
    ])
    elif tab == 'Factores':
        fig1, fig2, fig3 = generate_visualizations3(HA)
        return html.Div([
        html.Div([
            dcc.Graph(id='graph1', figure=fig1),
        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph2', figure=fig2),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='graph3', figure=fig3),
        ], style={'width': '50%', 'display': 'inline-block'})
        ])


if __name__ == '__main__':
    logger.info("Running dash")
    app.run_server(debug=True)

