#!/usr/bin/env python
# coding: utf-8

# In[1]:


#District Wise CPR


# In[2]:


import os
import glob
import dash
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_html_components as html
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import plot, iplot
from dash.dependencies import Input, Output, State
import plotly.express as px
import cufflinks as cf
#import char_studio


# In[3]:


app = dash.Dash()


# In[4]:


df = pd.read_csv("F:\\Data Science\districtwise_cpr.csv")
d = pd.read_csv("F:\\Data Science\doubling_rate.csv")


# In[5]:



#df = df.columns.to_list


# In[6]:


df.head()


# In[7]:


#df.iplot(kind='scatter')
#df=df.set_index()
#df = df[col].values


# In[8]:


d.head()


# In[9]:


#df = df.columns
fig = go.Figure()
fig1 = go.Figure()

for col in df.columns:
    fig.add_trace(go.Scatter(y=df[col].values,
                             name = col,
                             mode = 'lines')
                 )
for col in d.columns:
    fig1.add_trace(go.Scatter(y=d[col].values,
                             name = col,
                             mode = 'lines')
                 )
fig1.show()

buttonlist = []
for col in df.columns:
    buttonlist.append(
        dict(
            args=['y',[df[str(col)]] ],
            label=str(col),
            method='restyle'
        )
      )
buttonlist1 = []
for col in d.columns:
    buttonlist1.append(
        dict(
            args=['y',[d[str(col)]] ],
            label=str(col),
            method='restyle'
        )
      )

fig.update_layout(
        
        title="Choose district",
        yaxis_title="District wise CPR",
        xaxis_title="Days",
        font=dict(
        family='Courier New, monospace',
        size=18,
        color='#ffffff'
        ),
        
        # Add dropdown
        updatemenus=[
            
            go.layout.Updatemenu(
                buttons=buttonlist,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="center",
                y=1.1,
                yanchor="top",
                
            ),
        ],
        
        
        autosize=True
    ),
fig1.update_layout(
        
        title="Choose district",
        yaxis_title="Rate Of Increase",
        xaxis_title="Days",
        font=dict(
        family='Courier New, monospace',
        size=18,
        color='#ffffff'
        ),
        
        # Add dropdown
        updatemenus=[
            
            go.layout.Updatemenu(
                buttons=buttonlist1,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="center",
                y=1.1,
                yanchor="top",
                
            ),
        ],
        
        
        autosize=True
    )


fig.layout.plot_bgcolor = '#D3D3D3'
fig.layout.paper_bgcolor = '#000000'
fig1.layout.plot_bgcolor = '#D3D3D3'
fig1.layout.paper_bgcolor = '#000000'


# In[10]:


app.layout = html.Div([
    html.H1('Covid Intelligence Unit',style={'text-align': 'center'}),
    html.Div('Optimizing the way to control Covid-19',style={'text-align': 'center'}),
    
    html.Div([
         
        dcc.Graph(figure=fig)
    ]),
    html.Div([
        dcc.Graph(figure=fig1)
    ]),
    
])


# In[ ]:


if __name__ == '__main__':
    app.run_server(port=2019)


# In[ ]:





# In[ ]:




