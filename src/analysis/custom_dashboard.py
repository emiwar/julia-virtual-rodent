import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import h5py
import numpy as np
import pathlib

log_y_plots = {"cumulative_reward", "actuator_force_sum_sqr",
               "critic_loss"} #"lifetime"}

def create_simple_line(vals, title):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(y=vals, name=title, line=dict(width=2)))
    if title in log_y_plots:
        fig.update_yaxes(type='log')
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
        if i == quantiles.shape[1]//2:
            fig.add_trace(go.Scattergl(y=quantiles[:,i], line=dict(width=1)))
        else:    
            fig.add_trace(go.Scattergl(y=quantiles[:,i], line=dict(width=.2, color="#1f77b4"), hoverinfo = 'none'))
    if title in log_y_plots:
        fig.update_yaxes(type='log')
    fig.update_layout(
        #title=title,
        xaxis_title="Epoch",
        yaxis_title=title,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig

def generate_layout():
    navbar = dbc.NavbarSimple(
        children=[
            dash.dcc.Dropdown(sorted([str(p) for p in pathlib.Path("runs/").glob("*.h5")]),
                            id="selected-run", clearable=False, maxHeight=500,
                            style={'width': 500})
        ],
        brand="Custom PPO implementation",
        brand_href="#",
        color="primary",
        dark=True,
    )
    return dash.html.Div([navbar, dash.html.Div(id='main')])

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CERULEAN])
app.layout = generate_layout

@dash.callback(
    dash.Output('main', 'children'),
    dash.Input('selected-run', 'value')
)
def load_run(filename):
    if filename is None:
        return "Select a run."
    if not filename.startswith("runs/"):
        return f"Invalid filename: {filename}"
    f = h5py.File(str(filename), "r")
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
    return dbc.Container(layout, fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
