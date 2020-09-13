import pandas as pd
import numpy as np
import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly
import plotly.graph_objs as go
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint
from scipy import interpolate
from datetime import datetime as dt
import dash,copy
from dash.dependencies import Input, Output, State
import mydcc
import base64

############################################ the model ################################################
def deriv(y, t, beta, mu,  eta, gamma, sigma, l1, l2, l3, N, p_Is_to_C, p_C_to_D, Beds):
    S, E, Is, C, R, D, Ia, Q = y

    dSdt = -(beta(t) * Is + mu(t) * Ia) * S/ N
    dEdt = (beta(t) * Is + mu(t) * Ia) * S/ N - sigma * E
    dIsdt = eta * sigma * E - l1 * p_Is_to_C * Is - gamma * (1 - p_Is_to_C) * Is
    dCdt = l1 * p_Is_to_C * Is - l2 * p_C_to_D * min(Beds(t), C) - max(0, C-Beds(t)) - l3 * (1 - p_C_to_D) * min(Beds(t), C)
    dRdt = gamma * (1 - p_Is_to_C) * Is + l3 * (1 - p_C_to_D) * min(Beds(t), C)
    dDdt = l2 * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
    dIadt = (1-eta) * sigma * E - gamma * Ia
    dQdt =  gamma * Ia

    return dSdt, dEdt, dIsdt, dCdt, dRdt, dDdt, dIadt, dQdt


def R_1(t, R_1_start, k, x0, R_1_end, L, k1, R_2_start):
    return (R_1_start - R_1_end) / (1 + np.exp(-k * (-t + x0))) + R_1_end if t <= L else R_2_start/(1 + (R_2_start - R_1_end)/R_1_end * np.exp(- k1 * t))


def Model(initial_cases, initial_date, N, alpha, beds_per_100k, gamma, sigma, R_1_start, R_2_start, k, k1, L, x0, R_1_end, p_Is_to_C, p_C_to_D, s, theta, l1, l2, l3, eta): 
    days = 750

    def beta(t):
        return R_1(t, R_1_start, k, x0, R_1_end, L, k1, R_2_start) * gamma 

    def mu(t):
        return R_1(t, R_1_start, k, x0, R_1_end, L, k1, R_2_start) * theta * gamma

    def Beds(t):
        beds_0 = beds_per_100k / 100_000 * N
        return alpha * beds_0 + s * beds_0 * t  # 0.003

    y0 = N-initial_cases, initial_cases, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(0, days-1, days)
    ret = odeint(deriv, y0, t, args=(beta, mu, eta, gamma, sigma, l1, l2, l3, N, p_Is_to_C, p_C_to_D, Beds))
    S, E, Is, C, R, D, Ia, Q = ret.T
    
    R_0_over_time = [beta(i)/gamma for i in range(len(t))]
    R_1_over_time = [mu(j)/gamma for j in range(len(t))]

    total_Infected = [0] + [sum(sigma*E[:i]) if sum(sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
    daily_Infected = [0] + [sigma * E[i] if sigma * E[i] > 0 else 0 for i in range(1, len(t))]

    total_CFR = [0] + [100 * D[i] / sum(sigma*E[:i]) if sum(
        sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
    daily_CFR = [0] + [100 * ((D[i]-D[i-1]) / ((R[i]-R[i-1]) + (D[i]-D[i-1]))) if max(
        (R[i]-R[i-1]), (D[i]-D[i-1])) > 10 else 0 for i in range(1, len(t))]

    daily_treatment = [0] + [min(C[i], Beds(i)) for i in range(1, len(t))]
    daily_recovery =  [0] + [(1-p_C_to_D) * min(C[i], Beds(i)) for i in range(1, len(t))] 
    daily_death = [0] + [p_C_to_D * min(C[i], Beds(i)) for i in range(1, len(t))]
    
    dates = pd.date_range(start=np.datetime64(initial_date), periods=days, freq="D")

    return dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, [Beds(i) for i in range(len(t))]


############################################ the dash app layout (Navigation Bar) ################################################

external_stylesheets = [dbc.themes.BOOTSTRAP]
#external_stylesheets = [dbc.themes.GRID]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "Prediction"
server = app.server


logo_jpg = 'logo.jpg'
logo_base64 = base64.b64encode(open(logo_jpg, 'rb').read()).decode('ascii')

model_jpg = 'model.jpg'
model_base64 = base64.b64encode(open(model_jpg, 'rb').read()).decode('ascii')

##  Navbar ##
nav_bar = html.A(
    dbc.Row(
    [
         dbc.Col(html.Img( src='data:image/jpg;base64,{}'.format(model_base64), #)),
          style={'marginLeft':1000,'padding': '5px 5px 5px 5px'})),
    ],
 #   no_gutters=True,
    className="ml-auto",
    align="end",
),	href= 'https://www.scribd.com/document/475653133/tikz',
),

nav_menu = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img( src='data:image/jpg;base64,{}'.format(logo_base64))),
                ],
                align="start",
                no_gutters=True,
            ),
            href="/page-a",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(nav_bar, id="navbar-collapse", navbar=True, ),
    ],
    color="darkslategray", 
    dark=True, 
    style={'width':'100%', 'bgcolor':"darkslategray",'md':12}
)


############################################ the dash app layout (Contact-Page) ################################################

shalu_jpg = 'shalu1.jpg'
me_base64 = base64.b64encode(open(shalu_jpg, 'rb').read()).decode('ascii')
card1 = dbc.Card(
    [
        dbc.CardImg(src='data:image/jpg;base64,{}'.format(me_base64), top=True, style={'width':'6rem','height':'8rem','marginLeft': '6%'}), 
        dbc.CardBody(
            [
                html.H5("Shalu Dwivedi", className="card-title"),
                html.P(
                    "MS, Data Science,"),
                html.P(
                    "Université Grenoble Alpes, France",
                    className="mb-3",
                ),
              
            ]
         ),
    ], 
    style={"width": "19rem", 'height':'15rem', 'backgroundColor':'skyblue', 'border':'none', 'font-size':12 },
)

me_jpg = 'suman1.jpg'
me_base64 = base64.b64encode(open(me_jpg, 'rb').read()).decode('ascii')
card2 = dbc.Card(
    [
        dbc.CardImg(src='data:image/jpg;base64,{}'.format(me_base64), top=True, style={'width':'6rem','height':'8rem','marginLeft': '6%'},),
        dbc.CardBody(
            [
                html.H5("Suman Chakraborty", className="card-title"),
                html.P(
                    "Department of Bioinformatics,"),
                html.P(
                    "Friedrich-Schiller-Universität Jena, Germany",
                    className="mb-3",
                
                ),
                #dbc.Button("view profile", color="primary"),
            #href="https://www.uni-jena.de/"
            ]
        ),
    ],
    style={"width": "19rem", 'height':'15rem', 'backgroundColor':'skyblue', 'border':'none', 'font-size':12  },
),


jan_jpg = 'jan1.jpg'
me_base64 = base64.b64encode(open(jan_jpg, 'rb').read()).decode('ascii')
card4 = dbc.Card(
    [
        dbc.CardImg(src='data:image/jpg;base64,{}'.format(me_base64), top=True, style={'width':'6rem','height':'8rem','marginLeft': '6%'}),
        dbc.CardBody(
            [
                html.H5("Dr. Jan Ewald", className="card-title"),
                html.P(
                    "Department of Bioinformatics,"),
                html.P(
                    "Friedrich-Schiller-Universität Jena, Germany",
                    className="mb-3",
                ),
              ]
        ),
    ],
    style={"width": "19rem", 'height':'15rem', 'backgroundColor':'skyblue', 'border':'none', 'font-size':12  },
),


############################################ the dash app layout (Country Name) ################################################

controls1 = dbc.Card(
    [
	dbc.FormGroup(
            [
                  dcc.RadioItems(
                    id="population", options=[
						{'label': 'Argentina', 'value': 45191000},
						{'label': 'Belgium', 'value': 11589000},
						{'label': 'Brazil', 'value': 212538000},
						{'label': 'Colombia', 'value': 50878000},
						{'label': 'France', 'value': 65255000},
						{'label': 'Germany', 'value': 83765000},
						{'label': 'India', 'value': 1379956000},
						{'label': 'Italy', 'value': 60446000},
                 				{'label': 'Japan', 'value': 126398000},
                				{'label': 'Mexico', 'value': 128921000},
                 				{'label': 'Netherlands', 'value': 17133000},
                 				{'label': 'Peru', 'value': 32970000},
				                {'label': 'Poland', 'value': 37844000},
                 				{'label': 'Russia', 'value': 145927000},
                 				{'label': 'Spain', 'value': 46740000},                            
               					{'label': 'South Africa', 'value': 59307000},
				                {'label': 'Sweden', 'value': 10096000},
				                {'label': 'Switzerland', 'value': 8655000},                 
				                {'label': 'UK', 'value': 67871000},
                 				{'label': 'USA', 'value': 330905000}
  					  ],
   					 value= 83765000, labelStyle={'display': 'button', "width":'10%',}
				), 
 	 ]
       		 ),
        
        dbc.Button("Apply", id="submit-button-state", style={'backgroundColor':'indianred', 'display': 'inline-block', "width":"10%"}, block=True)
    ],
    body=True, style={ 'marginBottom':'1%', 'font-size': 14, 'backgroundColor':'aqua', 'width':'98%', "margin-right": "auto",  "margin-left": "auto" }, 
),

##Layout

app.layout = html.Div([
		dcc.Location(id='url', refresh=False),
		nav_menu,
		html.Div([html.Div([

		html.H2("Prediction: Country specific future scenarios of COVID-19 pandemic", className="display-8", style = {'display': 'block', 'backgroundColor':'skyblue', 'width':'98%', "margin-right": "auto",  "margin-left": "auto"}),
    #html.P("Country specific future scenarios of COVID-19 pandemic.", className="lead"),
       		html.Hr(className="my-2"),
                dcc.Markdown(''' We have developed a SEIR (compartmental) model to predict the future COVID-19 cases and evaluate its impact on the health care system. Our website contains the 
                    simulation results from our model. Please do not confuse the predicted cases of infections with the number of detected or reported case numbers. We know that the actual case 
                    numbers are far higher than what is reported. The most suitable reason behind this gap is insufficient tests, as well as inaccuracy in tracking the potential infectious. 
                  Apparently, it is almost impossible to find the entire infected cases not only because most infected have very mild or no symptoms (not even noticeable), but also it is hard to 
                  track how many contacts have been made by an infected before being tested as positive. To view the results, please select the country you like, and press the 'Apply' button.''',

          style = {'display': 'block', 'backgroundColor':'skyblue', 'textAlign': 'justify','width':'98%', 'font-size': 14, "margin-right": "auto", "margin-left": "auto", 'marginBottom':'2%'}), 

	html.H6(dbc.Row(
		    [
	                # here we place the controls we just defined,
               		dbc.Col(controls1, md=12),
               		# now we place the graphs on the page, taking up the right.
               dbc.Col(
                    [
                        html.Hr(className="my-2"),
                        dcc.Markdown('''Susceptible is the population who is supposed to be infected by COVID-19. Initially, it is the entire population of the country and as the pandemic progress, 
                            the volume of susceptible population declines. Final size of susceptible population (in our case, the susceptible on January 2022) represents the number of remaining 
                            uninfected people.''',
                        style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                # the graph for Susceptible people.
                                dbc.Col(dcc.Graph(id='sus_graph'), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown(''' 'Daily infected' literally means the number of infections per day.''' ,
                        style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                       dbc.Row(
                            [
                                # the graph for the total infected.
                                dbc.Col(dcc.Graph(id='di_graph'), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown(''' 'Total infected' has been just the cumulative representation of 'Daily infected'.''' ,
                        style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),    
 
                       dbc.Row(
                            [
                                # the graph for the daily infected.
                                dbc.Col(dcc.Graph(id="ti_graph"), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''Active cases represent the population who are still infected on a daily basis. This compartment counts an infected for several days until that particular 
                            infected become recovered or dead.''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                # the graph for the current infected(symptomatic and asymptomatic).
                                dbc.Col(dcc.Graph(id='sinf_graph'), md=12)
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''Critical compartment is designed to represent the number of infected having severe symptoms and require ICU facilities on a daily basis. Here, a patient can 
                            remain critical for several days until his/her recovery or death.''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                # the graph for the Critical.
                                dbc.Col(dcc.Graph(id='cri_graph'), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''Recovered compartment is the cumulative representation of recoveries on a daily basis. ''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                 # the graph for the recovered(R).
                                dbc.Col(dcc.Graph(id='rec_graph'), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''Dead compartment is the cumulative representation of deaths on a daily basis. ''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

 
                       dbc.Row(
                            [
                               # the graph for the death(D).
                                dbc.Col(dcc.Graph(id='death_graph'), md=12),
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown(''' 'Case Fatality Rate (CFR)' takes the ratio of 'Dead' and 'Total infected' per day. This way we can predict the impact of COVID-19 on infected population.''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                # the graph for the fatality rate over time.
                                dbc.Col(dcc.Graph(id='cfr_graph'), md=12)
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown(''' 'Recoveries and Deaths in ICU' compartment is specially designed to evaluate the performance of health care. The patients who are currently 'Critical' and 
                            admitted in ICUs are accommodated in this compartment. Here, patients can either recover or die, but definitely after getting treatment. A well equipped health care 
                            consists sufficient ICU beds for critical patients, whereas an under equipped health care system is responsible for deaths of critical patients without treatment.''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                 # the graph for the recovered vs critical in ICU.
                                dbc.Col(dcc.Graph(id='rec_vs_cri_graph'), md=12)
                            ]
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown(''' 'Deaths per day vs Deaths over capacity' compartment is for the unfortunate patients who die without getting ICU beds due to the overflown hospitals. Here, we 
                            compare the numbers of daily deaths with daily deaths due to the lack of ICUs. In some countries, the major proportion of the deaths may happen over hospital capacity 
                            because of insufficient ICUs and inability to admit critical patients. Patients are compelled to die due to this special situation, which can be described as “Triage”. 
                          By definition, Triage is a situation of classifying patients, according to the severity of their disease, age, as well as wealth to afford the treatment. 
                          An intense triage situation may indicate towards reform in health care of the corresponding countries.''',
                          style={'marginTop': '3%', 'marginBottom':'1%', 'textAlign': 'justify', 'font-size': 14 }),

                        dbc.Row(
                            [
                                # the graph for the daily deaths over time.
                                dbc.Col(dcc.Graph(id="deaths_graph"), md=12)
                            ]
                        ),
                        #html.Hr(className="my-2"),
                        #dcc.Markdown(''''''),
                   ],
                    md=12
                ),
            ],
            align="top",
        ), style = {'display': 'block', 'backgroundColor':'skyblue', 'width':'98%', "margin-right": "auto", "margin-left": "auto"}
),   
		html.Div([
				dcc.Markdown('''Declaration: 'Prediction' is our contribution to meet the globally growing concerns about COVID-19 pandemic. According to our analysis, subsequent second waves of 
                    stronger magnitude are inescapable in several countries. The dates and numbers you observe here, are simulated on the basis of current data trend only. So, we do not claim our 
                    results to match accurately in the future, rather we may suggest to consider 'Prediction' for prognosis. Several factors (like sudden changes of control measures or another 
                    lock downs by countries) may influence our prognosis, but we will certainly try to keep updating it according to the future trend of the data. Gradually, We will also accommodate 
                    other countries under 'Prediction'. We hope you may admire our effort.'''),
		], 
	    style={'width':'98%', "margin-right": "auto", "margin-left": "auto", 'marginTop': 40, 'marginBottom': 30, 'backgroundColor':'cyan', 'textAlign': 'justify', 'font-size': 14, 'border': 'navy dashed', 'padding': '10px 10px 10px 10px'}
		),

        html.H1("Contributers", className="display-8", style = {'display': 'block', 'backgroundColor':'skyblue', 'width':'16%', 
            "margin-right": "auto",  "margin-left": "auto"}),

    #html.H1("Contributers"),
                        #dcc.Markdown('''Write Here!'''),
		 html.H6( dbc.Row(
 			           [
	    dbc.Col(card1, md=4, style={"margin-right": "auto", "margin-left": "auto",'marginTop':'2%','padding':'10px 60px 80px 50px'}),
	    dbc.Col(card2, md=4, style={ "margin-right": "auto", "margin-left": "auto",'marginTop':'2%','padding':'10px 50px 80px 50px'}),
            dbc.Col(card4, md=4, style={ "margin-right": "auto", "margin-left": "auto",'marginTop':'2%','padding':'10px 50px 80px 50px'}),
					]
				   )
			),
    
    html.Div(dbc.Row([
	        dcc.Markdown('''Contacts: For any queries, requests or suggestions, you are most welcome to send an email to 'predictioncovid@gmail.com'. All contributors have access to the email. 
                We will be glad to answer according to our capacity.'''),
    	], 
	style={'width':'98%', "margin-right": "auto", "margin-left": "auto",'marginTop':'2%','marginBottom':'5%', 'border': 'navy dashed','textAlign': 'justify', 'font-size': 14, 'padding': '10px 10px 10px 10px'}
    ),),

], id = 'page-a' ),
],
               style = {'display': 'block', 'backgroundColor':'skyblue', 'width':'85%', 'marginTop': '2%', 'marginBottom':'2%', "margin-right": "auto", "margin-left": "auto", 
               'padding': '30px 10px 30px 10px'}
)

])


############################################ the dash app callbacks ################################################

@app.callback(
    [dash.dependencies.Output('sus_graph', 'figure'), dash.dependencies.Output('ti_graph','figure'),
     dash.dependencies.Output('di_graph', 'figure'), dash.dependencies.Output('sinf_graph', 'figure'),
     dash.dependencies.Output('cri_graph', 'figure'), dash.dependencies.Output('rec_graph', 'figure'),
     dash.dependencies.Output('death_graph', 'figure'), dash.dependencies.Output('cfr_graph', 'figure'),
     dash.dependencies.Output('rec_vs_cri_graph', 'figure'), dash.dependencies.Output('deaths_graph', 'figure'),],
     
    [dash.dependencies.Input('submit-button-state', 'n_clicks')],
    [#dash.dependencies.State('initial_date', 'date'),   
	 dash.dependencies.State('population', 'value'),   ])

def update_graph(_, population):
    
    last_population =  1_000_000
    if not (population ):
        population = last_population

#Germany
    if population==83765000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-28", population, 0.38, 67.4, 0.12984078673112479, 0.370412212666437, 3.774914306692381, 2.391521374361648, 2.5558296678605354, 0.005993498046712659, 180.3225428685107, 95.90497606043034, 0.6604888898883905, 0.031090750924749708, 0.3876199735537736, 0.0001495000000005142, 0.6112956476396956, 0.05000629213232478, 0.0923057014334686, 0.19109293175174272, 0.5409258981269481)

#India
    elif population==1379956000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-27", population, 0.10, 8.65, 0.1428541508163331, 0.3774442401686795, 3.4302250512082697, 1.0007005912433975, 1.6657944669933995, 0.15177083271888495, 255.95506574510938, 94.38312701825069, 1.4917953849161987, 0.01000143100156799, 0.2187670323177473, 0.006732944751207018, 0.7427829181614399, 0.050105836845896624, 0.05001145192912416, 0.10024160588622241, 0.5124987959493479)   
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-01-07", population, 0.10, 8.65, 0.14285517983097165, 0.35666658586003874, 3.606503965453624, 2.5199955217893875, 2.454279145307359, 0.3023190572522995, 240.42639189692994, 101.32440873143399, 1.543697634652902, 0.01000130779247402, 0.2454283921449709, 0.0034445356726693853, 0.653703267584974, 0.05001967317137948, 0.05000764711139616, 0.07278138830436388, 0.4702976160650683) #0.002758075290312366

#Brazil
    if population==212538000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-22", population, 0.10, 14.95, 0.1305245028905751, 0.4627117002701086, 3.754827792054293, 2.064909117189166, 2.083599363128264, 0.08732566287730936, 246.3040954283558, 85.62952781409258, 1.2663707059472125, 0.04229390529141109, 0.460770297263991, 0.009950499999873163, 0.6491978198998314, 0.05000013417990671, 0.050000411304217236, 0.05000788163161523, 0.5524370162316564)

#Italy
    elif population==60446000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-20", population, 0.21, 20.8, 0.12830369149390133, 0.48381748773041133, 3.8233823807596865, 2.7431818824249596, 3.249337661433929, 0.0845278913279641, 238.4981970221766, 87.44763134201042, 0.7983528956314749, 0.05326293026950918, 0.46373644465361347, 0.009998750448276614, 0.606100297995981, 0.057794375443971736, 0.06220862900930933, 0.12459093149630548, 0.5599174589970304)

#Peru
    elif population==32970000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-02-01", population, 0.01, 1.32, 0.10600111259175139, 0.35553229613395276, 3.202272582143562, 1.3766356913426219, 2.219171880214629, 0.8163497034165993, 171.23895399502118, 90.45422812412542, 1.3633765369750255, 0.062489317666343074, 0.7079615886182496, 0.001, 0.6745863911999823, 0.06687003769756564, 0.22474759316347473, 0.10769548789185433, 0.6181136962814743)
 
#Russia
    elif population==145927000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.15, 35.6, 0.14269804919303197, 0.20032001947715206, 4.184919897978229, 2.614752123021756, 2.4304057089263904, 0.38684061178610546, 230.1213943349641, 100.66203323097957, 1.4714916492935983, 0.010000978776059216, 0.20155704190341223, 0.002999999999999846, 0.7898249333418573, 0.05000012222551507, 0.050000038607246666, 0.050001931295602244, 0.42119706910056287)

#Spain
    elif population==46740000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-20", population, 0.247, 14.7, 0.1254554440309099, 0.42428295543397276, 3.829080163105371, 2.4932863210009852, 2.565524510578827, 0.01854529282707962, 196.0278056384305, 95.64233911155168, 0.5696715274075116, 0.054082267264861456, 0.651206924466544, 0.006423165217834747, 0.5081623043012692, 0.08639393586402293, 0.15999682590336084, 0.1131217266361835, 0.4999687616919529)

#Mexico
    elif population==128921000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-20", population, 0.26, 2.8, 0.128299478530902, 0.3302102275197734, 3.4732436474501696, 2.6739421451158627, 2.4662654144477965, 0.09255268029814677, 217.59663568489648, 98.33538226644626, 1.292147566141818, 0.03832476866599589, 0.5163765530314895, 0.009900517132143861, 0.6588080197287363, 0.05010177004791314, 0.13214050511149944, 0.10447585302557208, 0.5646311560190272)

#Iran
    elif population==83993000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-15", population, 0.1, 7.3, 0.09986370942406092, 0.3743841269593251, 3.7651483620592696, 2.6582842821877115, 2.5600179781365857, 0.005476077260293967, 141.52973334247693, 90.80017677483585, 0.8712217874662148, 0.0647438594490469, 0.5203123147666895, 0.001, 0.5976003990170534, 0.08614897731187705, 0.16390750851670452, 0.129968967664481, 0.570768085813796)

#UK
    elif population==67871000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-20", population, 0.157, 16.7, 0.12715967925579497, 0.37730941878658386, 3.6928441089123463, 2.4947944337054424, 2.4455148335281915, 0.027972945289788855, 227.0373803897242, 102.8957730927176, 0.7361125036336488, 0.04990947370921135, 0.5819273292129709, 0.009998234145419617, 0.5396349098586515, 0.07862972895061905, 0.0938828536720864, 0.08313369345665671, 0.5161576558658345)

#USA
    elif population==330905000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-30", population, 0.36, 53.5, 0.11859440223272465, 0.3829588041892806, 4.333863991194564, 1.6038555863262742, 2.598636222436476, 0.5542270116865097, 158.7224577029222, 92.11415826358717, 0.9005989492812969, 0.0670082739299351, 0.5004030953619553, 0.009950499999999998, 0.518618781852357, 0.11218423885618535, 0.08175608960436331, 0.08396592911838625, 0.5294223039749762)

#France
    elif population==65255000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-26", population, 0.244, 22.3, 0.12919289160603295, 0.39802968219140467, 3.665918660435665, 2.9269166975924867, 2.534294636205726, 0.03695540868883431, 210.87917058763182, 94.66877688567514, 0.559854963155523, 0.05596965447322385, 0.5283242004338682, 0.00521845863664777, 0.5825361227354474, 0.08446539070715221, 0.11689021676434563, 0.13976102706025856, 0.5470382456455533)

#Chile
    elif population==19112000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.209, 9.45, 0.12189814227575815, 0.3176712150977276, 3.2747880236725995, 2.2870256123322927, 2.616301565912923, 0.0020546519136984084, 189.40056620335562, 99.48645705730006, 1.466650585060447, 0.021038052575449228, 0.31224883418385746, 0.001, 0.7913912680992813, 0.05053444018856565, 0.053522073530619725, 0.08095106001296393, 0.5159229273728257)

#Argentina
    elif population==45191000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-22", population, 0.1, 7.45, 0.08635763991968512, 0.24016831898122043, 3.296873810196251, 1.8449140371135206, 2.3987918062419413, 0.3134163596668651, 170.21421045183092, 98.60479122366982, 1.575949656112387, 0.07024138665075999, 0.6899870942733037, 0.004665376840701445, 0.676654620109646, 0.05012945568487421, 0.24177936093552133, 0.12082843497689191, 0.6150111119305313)

#Turkey
    elif population==84336000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.32, 66.66, 0.11765616405381431, 0.3444335783685702, 3.830678734679756, 1.4907161032023748, 2.43660992174739, 0.05265450373764808, 164.69602502009252, 95.19409435753052, 0.7776206629160574, 0.04029420145619635, 0.45452736986468323, 0.001, 0.5653614227410103, 0.05415126318022234, 0.18247496324827325, 0.19996655839899904, 0.5148595959088139)

#Philippines
    elif population==109579000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.05, 2, 0.10562260236969818, 0.3194104248869056, 3.299647579477651, 2.2441551221448117, 1.8122963073421878, 0.008342557957959903, 174.69970939645148, 99.016243537992, 1.1038605465578257, 0.058182254264254364, 0.4633830030252964, 0.001, 0.5687625661136908, 0.07974302935252058, 0.13928956156834707, 0.1330445937580035, 0.5110357018120946)

#Indonesia
    elif population==273521000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-02-01", population, 0.05, 2, 0.09796291654534713, 0.5987154892600363, 3.426047936373679, 2.5172881366476103, 2.976736082966355, 0.19059102216001209, 186.10289505754824, 60.763807641787494, 1.435360283558129, 0.09932518178781065, 0.7014607582647638, 0.001, 0.6120677282851443, 0.199578793940744, 0.29892180316014905, 0.19960553957519753, 0.6244074696467922)

#Belgium
    elif population==11589000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.182, 27.9, 0.1255473398040381, 0.4185521887547508, 3.473732433041309, 2.952364885643484, 0.539497425798363, 0.01497222885720287, 188.97628856442424, 93.73818561276914, 0.5357193913697594, 0.061422657152317955, 0.5367539814435632, 0.0035715487909123863, 0.5482078761745521, 0.09765637673670724, 0.2110413108318685, 0.15424649543147645, 0.5301879477375873)

#Netherlands
    elif population==17133000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-30", population, 0.346, 18.4, 0.12660302074592797, 0.418038321197927, 3.478303054700837, 2.7156423289332325, 2.0427539587825634, 0.060388913176838116, 208.7978463026444, 88.5273093928572, 0.603615158903559, 0.060377821787094506, 0.5290072376164849, 0.004142475349822342, 0.5649483872909955, 0.09652744147389149, 0.13049962289177972, 0.13368758430492322, 0.5448541065847047)

#Sweden
    elif population==10096000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.15, 11.45, 0.11935216887770028, 0.4045891832326256, 3.451926327861535, 2.6784232278430085, 1.278591791862816, 0.04449119021711065, 250.62354788605077, 95.17511209119397, 0.9927297288307789, 0.04704933560312835, 0.594041520365077, 0.009999426037644919, 0.5639064565387321, 0.06189430915290856, 0.10064263127514855, 0.11585183465817393, 0.5210580034362773)

#Switzerland
    elif population==8655000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.18, 21, 0.1259686429715002, 0.41968580941504596, 3.5544085711980373, 2.467983604159466, 3.338744526447595, 0.8384977685218602, 183.77493212157773, 88.10626620773134, 0.45385324667280197, 0.04241031821947792, 0.4437394406367922, 0.004146398165406062, 0.5715519495454099, 0.059661872583196426, 0.13377059155523618, 0.17762791279483311, 0.5312684663286634)

#Poland
    elif population==37844000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-30", population, 0.15, 33.5, 0.10637703048458, 0.32974603519385104, 3.539726165440316, 2.014355012450462, 2.841188966266996, 0.125071990162543, 209.49254161780334, 94.608114539059, 1.0610407911310469, 0.0409815565746078, 0.45076776482709824, 0.0029999999999998045, 0.5849689645416305, 0.05901856852098479, 0.1102958057699774, 0.16605055587183343, 0.5463930909539617)

#Portugal
    elif population==10193000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-01", population, 0.1, 13.2, 0.13609496713888553, 0.29218754171896133, 3.9031424015248057, 2.9523991210145666, 2.4970499203423655, 0.003372712780499174, 150.2414970560147, 81.4575374392949, 0.8304722829405276, 0.027880589417283462, 0.5097133417703729, 0.001, 0.6211387774796723, 0.05000184566812434, 0.050034384844355906, 0.0655945349064284, 0.5382201149584575)

#Japan
    elif population==126398000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-12-10", population, 0.245, 49.33, 0.10481537665183834, 0.10470523620944183, 6.444480380775488, 3.801067396937808, 1.861536557347789, 0.10460802792068717, 204.525769773934, 119.32224243590701, 0.30001934973294153, 0.009810002872715539, 0.2073990292760525, 0.001, 0.6968838913227037, 0.0505496644325199, 0.050506921474361324, 0.085970654950504, 0.26925793078671956)

#South Korea
    elif population==51265000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2019-11-18", population, 0.2, 29.7, 0.1422339788146269, 0.2404778895069057, 3.5757704539841537, 2.7687168288544415, 2.6618353508448958, 0.051185598955605614, 216.89113345023233, 102.78887636752334, 0.7555272930647716, 0.00723713089641602, 0.4037326829637937, 0.001, 0.5825286547380502, 0.05645746499149452, 0.050081256063116156, 0.08304703477893667, 0.48557079629815997)

#South Africa
    elif population==59307000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-20", population, 0.1, 10.3, 0.12071980285386676, 0.34729025208195685, 3.270339511367968, 2.6715998948755817, 1.6334497216883226,  0.018614610809082943, 228.4178236383145, 94.648860946569, 1.7979434669736887, 0.014828130493937325, 0.316629721386868, 0.00866390123379665, 0.7179810876139996, 0.05032187105437112, 0.05584640002625346, 0.1505848862225384, 0.5501701329647415)

#Colombia
    elif population==50878000:
     dates, S, E, Is, C, R, D, Ia, Q, R_0_over_time, R_1_over_time, total_CFR, daily_CFR, total_Infected, daily_Infected, daily_treatment, daily_recovery, daily_death, B = Model(1.0, "2020-01-15", population, 0.01, 1.9, 0.11199961298012397, 0.3762278168269233, 3.2773766934980952, 3.3308901194825564, 1.6284573307617183, 0.7294341075556462, 224.71937579891951, 92.72934695623042, 1.397281021477436, 0.042324362367858824, 0.3850660407411915, 0.00998424857270472, 0.6830236472802009, 0.06712173923986227, 0.05087387336453491, 0.16227217896490007, 0.5454520743924544)

    return {  # return graph for compartments, graph for fatality rates, graph for reproduction rate, and graph for deaths over time
        'data': [
            {'x': dates, 'y': S.astype(int), 'type': 'line', 'name': 'Susceptible'}
        ],
        'layout': {'title': 'Susceptible'}
        }, {
        'data': [
            {'x': dates, 'y': total_Infected, 'type': 'line', 'name': 'Total infected'},
        ],
        'layout': {'title': 'Total Infected'}
        }, {
        'data': [
            {'x': dates, 'y': daily_Infected, 'type': 'line', 'name': 'Daily infected'}
        ],
        'layout': {'title': 'Daily Infected'}
#        }, {
 #       'data': [
  #          {'x': dates, 'y': E.astype(int), 'type': 'line', 'name': 'Exposed(E)'}
   #     ],
    #    'layout': {'title': 'Exposed(E)'}

        }, {
        'data': [
            {'x': dates, 'y': (Is+Ia).astype(int), 'type': 'line', 'name': 'Current infected'},
            {'x': dates, 'y': Is.astype(int), 'type': 'line', 'name': 'Symptomatic infected'},
            {'x': dates, 'y': Ia.astype(int), 'type': 'line', 'name': 'Asymptomatic infected'}
        ],
        'layout': {'title': 'Active cases'}

 #       }, {
  #      'data': [
  #          {'x': dates, 'y': Is.astype(int), 'type': 'line', 'name': 'Symptomatic infected(Is)'}
   #     ],
    #    'layout': {'title': 'Symptomatic infected(Is)'}
     #   }, {
      #  'data': [
      #      {'x': dates, 'y': Ia.astype(int), 'type': 'line', 'name': 'Asymptomatic infected(Ia)'}
      #  ],
      #  'layout': {'title': 'Asymptomatic infected(Ia)'}
        }, {
        'data': [
            {'x': dates, 'y': C.astype(int), 'type': 'line', 'name': 'Critical'}
        ],
        'layout': {'title': 'Critical'}
        }, {
        'data': [
            {'x': dates, 'y': (R+Q).astype(int), 'type': 'line', 'name': 'Recovered'}
        ],
        'layout': {'title': 'Recovered'}
        }, {
        'data': [
            {'x': dates, 'y': D.astype(int), 'type': 'line', 'name': 'Dead'}
        ],
        'layout': {'title': 'Dead'}
        }, {
        'data': [
  #          {'x': dates, 'y': daily_CFR, 'type': 'line', 'name': 'daily'},
            {'x': dates, 'y': total_CFR, 'type': 'line', 'name': 'total'}
        ],
        'layout': {'title': 'Case Fatality Rate over time (%)'}
        }, {
        'data': [
 #          {'x': dates, 'y': daily_treatment, 'type': 'line', 'color':'red', 'name': 'Daily treatment in ICU'},
            {'x': dates, 'y': daily_recovery, 'type': 'line', 'name': 'Recoveries in ICU'},
            {'x': dates, 'y': daily_death, 'type': 'line', 'name': 'Deaths in ICU'}
        ],
        'layout': {'title': 'Recoveries and Deaths in ICU'}
        }, {
        'data': [
            {'x': dates, 'y': [0] + [D[i]-D[i-1] for i in range(1, len(dates))], 'type': 'line', 'name': 'Daily deaths'},
            {'x': dates, 'y': [0] + [max(0, C[i]-B[i]) for i in range(1, len(dates))], 'type': 'line', 'name': 'Daily deaths without ICU'}
        ],
        'layout': {'title': 'Deaths per day vs Deaths over capacity'}
        }

############################################ the component-id callbacks ################################################

@app.callback(
    Output(component_id='page-a', component_property='style'),
    [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-a':
        return {'display': 'block'}
    else:
        return {'display': 'block'}




if __name__ == '__main__':
    app.run_server(debug=True)
