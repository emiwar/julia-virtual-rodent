import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import h5py
import numpy as np

#filename = "runs/test-2024-05-31T18:24:59.981.h5"
filename = "runs/test-2024-05-31T19:50:37.942.h5"

def create_simple_line(vals, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vals, name=title, line=dict(width=2)))
    fig.update_layout(
        #title=title,
        xaxis_title="Epoch",
        yaxis_title=title,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

def create_quantile_plot(quantiles, title):
    fig = go.Figure()
    for i in range(quantiles.shape[1]):
        alpha = 0.2#np.exp(-((i/quantiles.shape[1] - 0.5)**2)/4)
        fig.add_trace(go.Scatter(y=quantiles[:,i], line=dict(width=.5, color=f"rgba(0.0,0.0,1.0,{alpha})")))
    fig.update_layout(
        #title=title,
        xaxis_title="Epoch",
        yaxis_title=title,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CERULEAN])
f = h5py.File(filename, "r")
layout = [dbc.Row(dash.html.H1(filename))]
for group in f["training"].keys():
    layout.append(dbc.Row(dash.html.H4(group)))
    row = []
    for entry in f[f"training/{group}"].keys():
        if isinstance(f[f"training/{group}/{entry}"], h5py.Dataset):
            n_epochs = max(0, f['n_epochs'][0]-1)
            vals = f[f"training/{group}/{entry}"][:n_epochs]
            row.append(dbc.Col([dash.dcc.Graph(figure=create_simple_line(vals, entry))], width=4))
        elif "quantiles" in f[f"training/{group}/{entry}"]:
            n_epochs = max(0, f['n_epochs'][0]-1)
            quantiles = np.array(f[f"training/{group}/{entry}/quantiles"][:n_epochs, :])
            row.append(dbc.Col([dash.dcc.Graph(figure=create_quantile_plot(quantiles, entry))], width=4))
        else:
            print(f"Unknown entry type for {group}/{entry}")

    layout.append(dbc.Row(row))
app.layout = dbc.Container(layout, fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
