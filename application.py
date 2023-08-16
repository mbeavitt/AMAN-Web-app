# Import statements
import pandas as pd
import numpy as np
import random

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.graph_objs as go

import os
import matplotlib.pyplot as plt
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq

import hdbscan

# Define a function to add Gaussian noise
def add_gaussian_noise(data, scale=0.05):
    noise = np.random.normal(0, scale * np.std(data), len(data))
    return data + noise

def resolve_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def performDR(features_cols, df, technique='umap', random_seed=42, components = list(range(0, 11))):

    df_filtered = df[df[features_cols].notnull().all(axis=1)]
    df_standardized = (df_filtered[features_cols] - df_filtered[features_cols].mean()) / df_filtered[features_cols].std()
    tsne_method = 'exact' if len(components) >= 4 else 'barnes_hut'
    dr = {'pca': PCA(n_components=max(components)+1), 'tsne': TSNE(n_components=max(components)+1, random_state=random_seed, method=tsne_method), 'umap': UMAP(n_components=max(components)+1, n_neighbors=20, min_dist=0.6, spread=2, random_state=random_seed)}[technique]
    
    dr_result = dr.fit_transform(df_standardized)
    
    if technique == 'pca':
        weights = dr.components_
        # Specify the path to the pickle file
        pickle_file = '~/temp_weights/weights.pkl'
        features_file = '~/temp_weights/features.pkl'

        # Use os.path.expanduser to handle the '~'
        pickle_file = os.path.expanduser(pickle_file)
        features_file = os.path.expanduser(features_file)

        # Pickle dump the weights
        with open(pickle_file, 'wb') as f:
            pickle.dump(weights, f)

        with open(features_file, 'wb') as f:
            pickle.dump(features_cols, f)

    dr_result_selected = dr_result[:, components]
    
    dr_df = pd.DataFrame(data=dr_result_selected, columns=['Component {}'.format(i+1) for i in components])
    
    return dr_df


def createFigure(title, point_size, dr_df, master_df, features_cols, label_col, plot_3d=True, hover_data=None, chosen_components = {'x_comp': 'Component 1', 'y_comp' : 'Component 2', 'z_comp' : 'Component 3'}):

    figure_width = 600*1.2
    figure_height = 400*1.2

    dr_df_subset = dr_df[list(chosen_components.values())].copy()

    plot_columns = list(dr_df_subset.columns)

    valid_rows = master_df[features_cols].notnull().all(axis=1)

    if label_col == "_HDBSCAN_":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        clusterer.fit(dr_df_subset)
        hdb_lab = clusterer.labels_
        labels_series = pd.Series(hdb_lab, index=master_df[valid_rows].index)
        master_df.loc[valid_rows, label_col] = labels_series

    labels_df = master_df[label_col][valid_rows].astype(str).fillna("Missing Data")
# optionally output to csv file... very useful..!
#    pd.concat([master_df['ID'][valid_rows], labels_df], axis=1).to_csv('~/Documents/mooney_files/data/labels.csv', index=False)

    labels = labels_df.tolist()
    labels_df = pd.concat([master_df['ID'][valid_rows], labels_df], axis=1)

    sorted_unique_labels = sorted(set(labels))
    label_mapping = {label: i+1 for i, label in enumerate(sorted_unique_labels)}
    labels_df[label_col] = labels_df[label_col].map(label_mapping)
    labels_df['count'] = labels_df.groupby(label_col).cumcount() + 1

    dr_df_subset['labels'] = labels
    dr_df_subset.loc[dr_df_subset['labels'].isin(['<NA>', 'nan']), 'labels'] = 'Missing Data'
    dr_df_subset = dr_df_subset.reset_index(drop=True)

    if hover_data is not None:
        hover_df = master_df[hover_data][valid_rows].reset_index(drop=True)
    else:
        hover_df = None

    unique_labels = sorted(set(dr_df_subset['labels']))

    # Making sure 'Missing Data' always appears last in the list if present
    if 'Missing Data' in unique_labels:
        unique_labels.remove('Missing Data')
        unique_labels.append('Missing Data')

    fig = go.Figure()
    color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    color_dict = {str(label): color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}
    for label in unique_labels:

        label_data = dr_df_subset[dr_df_subset['labels'] == label].copy()
        color = color_dict[str(label)]

        #print(label_data)
        if hover_df is not None:
            hover_data_filtered = hover_df.loc[label_data.index]
            hovertemplate_parts = ["<br>".join([f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_data)])]
        else:
            hover_data_filtered = None
            hovertemplate_parts = []

        if plot_3d:
            fig.add_trace(go.Scatter3d(x=label_data[plot_columns[0]], y=label_data[plot_columns[1]], z=label_data[plot_columns[2]],
                                    mode='markers', name=str(label),
                                    marker=dict(color=color, size=point_size),
                                    customdata=hover_data_filtered.values if hover_data_filtered is not None else [],
                                    hovertemplate=hovertemplate_parts[0] if hovertemplate_parts else ''))

            fig.update_layout(width=figure_width, height=figure_height, title=title, autosize=True, dragmode='select',
                            scene=dict(xaxis_title=plot_columns[0], yaxis_title=plot_columns[1], zaxis_title=plot_columns[2])) # Add this line
            fig.update_layout(legend_title_text=label_to_display_map[label_col])

        else:
            fig.add_trace(go.Scatter(x=label_data[plot_columns[0]], y=label_data[plot_columns[1]],
                                    mode='markers', name=str(label),
                                    marker=dict(color=color, size=point_size),
                                    customdata=hover_data_filtered.values if hover_data_filtered is not None else [],
                                    hovertemplate=hovertemplate_parts[0] if hovertemplate_parts else '',
                                    selectedpoints=[],
                                    selected=dict(marker=dict(color='orange')),
                                    unselected=dict(marker=dict(opacity=0.9))))

            fig.update_layout(width=figure_width, height=figure_height, title=title, autosize=True, dragmode='select')
            fig.update_xaxes(title_text=plot_columns[0])  # Add this line
            fig.update_yaxes(title_text=plot_columns[1])  # Add this line
            fig.update_layout(legend_title_text=label_to_display_map[label_col])


    return fig, labels_df




# Loading the dataframe from a pickle file
with open('master_df.pickle', 'rb') as f:
    master_df = pickle.load(f)

resolve_duplicate_columns(master_df)

# Setting problematic NAs to zero (biologically justified)
master_df['2-Unstim-Pop1-PMNs-CD66pos-CD66-MFI'] = master_df['2-Unstim-Pop1-PMNs-CD66pos-CD66-MFI'].fillna(0)
master_df['2-PMA-Pop1-PMNs-CD11cpost-CD11c-MFI'] = master_df['2-PMA-Pop1-PMNs-CD11cpost-CD11c-MFI'].fillna(0)
master_df['2-PMA-Pop1-PMNs-CD66pos-CD66-MFI'] = master_df['2-PMA-Pop1-PMNs-CD66pos-CD66-MFI'].fillna(0)

# Creating a table of contents
column_dict = {
    "Ungated FACS Data": [],
    "Population 1 FACS Data (Inactive Neutrophils)": [],
    "Population 2 FACS Data (Activated Neutrophils)": [],
    "Population 3 FACS Data (Dead Neutrophils)": [],
    "ROS assay Data": [],
    "Cell Count Data": [],
    "Cytokine Quantification Assay Data": [],
    "Biometric Data (Height, weight, etc.)": [],
    "Netosis Assay Data": [],
    "Netosis Assay Data (fold change)": [],
}
# Define the start and end column for each category in a list of tuples
range_list = [
    ("1-Unstim-Cells-FoP", "2-PMA-Pop3-PMNs-Hmox1pos-Hmox1-MFI"),
    ("1-FoP", "2-%ROS-lo"),
    ("RBC", "Gra%"),
    ("CD163 (BR28) (28)", "TFR(BR13) (13)"),
    ("Initial_weight", "End_haem"),
    ("media_netosis", "nts_netosis"),
    ("PMA_fc", "NTS_fc")
]
# For each start and end column, get all column names in this range and store in the dictionary
for i, (start, end) in enumerate(range_list):
    start_idx = list(master_df.columns).index(start)
    end_idx = list(master_df.columns).index(end)

    if i == 0:  # The first range contains multiple categories
        column_dict["Ungated FACS Data"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop' not in col]
        column_dict["Population 1 FACS Data (Inactive Neutrophils)"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop1' in col]
        column_dict["Population 2 FACS Data (Activated Neutrophils)"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop2' in col]
        column_dict["Population 3 FACS Data (Dead Neutrophils)"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop3' in col]
        column_dict["FACS (PMA only)"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'PMA' in col]
        column_dict["FACS (Unstim only)"] = [col for col in master_df.columns[start_idx : end_idx+1] if 'Unstim' in col]
        column_dict["Killing Assay Data (raw)"] = ['Unstim_phagocytosis', 'Unstim_killing', 'PMA_phagocytosis', 'PMA_killing']
    else:  # All other ranges correspond to one category
        key = list(column_dict.keys())[i+3]  # Skip the first four keys
        column_dict[key] = [col for col in master_df.columns[start_idx : end_idx+1]]

label_colnames = [
    'geo_cluster',      # geographic clusters of samples
#    'Group',             # one of three; Infected, Resolved, Control
    'Village',           # Village that patient lived in at time of sampling
    'Age',               # Age in years
    'Sex',               # male or female
    'killing_label',     # One of four; 'NonKiller', 'Killer-NonKiller', 'NonKiller-Killer', or 'Killer', based on Salmonella killing assay
    'days_after_start',   # days since start of sampling
    'visit_1_parasites_pa', # presence/absence of parasites on visit 1
    'visit_2_parasites_pa', # presence/absence of parasites on visit 2
    'ID', # Careful using this one to label clusters..!
    'RNAseq_done',
    'infection_status',
    'anemia_status',
    'killing_unstim',
    'killing_pma',
    'anemia_start',
    'anemia_end',
    'ph_HDBSCAN'
]

colour_colnames = [
    'geo_cluster',      # geographic clusters of samples
#    'Group',             # one of three; Infected, Resolved, Control
    'Village',           # Village that patient lived in at time of sampling
    'Age',               # Age in years
    'Sex',               # male or female
    'killing_label',     # One of four; 'NonKiller', 'Killer-NonKiller', 'NonKiller-Killer', or 'Killer', based on Salmonella killing assay
    'days_after_start',   # days since start of sampling
    'visit_1_parasites_pa', # presence/absence of parasites on visit 1
    'visit_2_parasites_pa', # presence/absence of parasites on visit 2
    'RNAseq_done',
    'infection_status',
    'anemia_status',
    'killing_unstim',
    'killing_pma',
    'anemia_start',
    'anemia_end',
    'ph_HDBSCAN'
]

label_to_display_map = {
    'geo_cluster': 'Geo Cluster',
#    'Group': 'Infection Status',
    'Village': 'Village',
    'Age': 'Age',
    'Sex': 'Sex',
    'killing_label': 'Killing Assay Label',
    'days_after_start': 'Days Since Start',
    'visit_1_parasites_pa': 'Parasites at Start? (y/n)',
    'visit_2_parasites_pa': 'Parasites at End? (y/n)',
    'ID': 'Patient ID',
    'RNAseq_done': 'RNA-seq Data? (y/n)',
    'infection_status': 'Infection Status',
    'anemia_status': 'Anemia Status',
    'killing_unstim': 'Killing Ability (Unstim ONLY)',
    'killing_pma': 'Killing Ability (PMA Stimulation ONLY)',
    'anemia_start': 'Anemic at start? (y/n)',
    'anemia_end': 'Anemic at end? (y/n)',
    '_HDBSCAN_': 'Assigned Using HDBSCAN',
    'ph_HDBSCAN': 'Legacy HDBSCAN (sanity check)'
}

num_to_component_map = {
     'Component 1':1,
     'Component 2':2,
     'Component 3':3,
     'Component 4':4,
     'Component 5':5,
     'Component 6':6,
     'Component 7':7,
     'Component 8':8,
     'Component 9':9,
     'Component 10':10
}

DR_styles = [
    "umap",
    "tsne",
    "pca"
]

dr_style_flag = "UMAP"

# master_df = synthetic_df
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

# Define the layout
app.layout = html.Div([

    html.H1("UMAP dimensionality reduction of the AMAN (Asymptomatic Malaria, Anemia and Neutrophils) dataset", style={'textAlign': 'center'}),

    html.Div([  # Div for hover_cols
        html.Label('Hover Data: (optional)'),
        dcc.Checklist(id='hover_cols', options=[{'label': label_to_display_map[l], 'value': l} for l in label_colnames], value=[],
                      style={'float': 'left', 'width': '100%'})],
        style={'width': '13%', 'float': 'left', 'margin-left': '15px'}
    ),

    html.Div([  # Div for features_cols and label_col
        html.Div([  # Subdiv for features_cols
            html.Label('Feature selection:'),
            dcc.Dropdown(id='features_cols', options=[{'label': k, 'value': k} for k in column_dict.keys()], multi=True)],
            style={'width': '40%', 'float': 'left', 'margin-right': '20px'}
        ),
        html.Div([  # Div for update_button
            html.Button('Update Plot', id='update_button')],
            style={'float': 'left', 'margin-top': '25px', 'margin-right': '50px'}
        ),
        html.Div([  # Subdiv for label_col
            html.Label('Colour by:'),
            dcc.Dropdown(id='label_col', options=[{'label': label_to_display_map[l], 'value': l} for l in colour_colnames], multi=False)],
            style={'width': '23%', 'float': 'left', 'margin-right': '40px'}
        ),

        html.Div([  # Subdiv for picking DR style
            html.Label('DR style:'),
            dcc.Dropdown(id='dr_style', options=[{'label': d, 'value': d} for d in DR_styles], multi=False, value="umap")],
            style={'width': '23%', 'float': 'left', 'margin-left': '575px'}
        ),
        html.Div([  # Div for 3D plot
            daq.ToggleSwitch(
                id='plot_3d',
                value=False,
                label="3D on/off",
                labelPosition='top'
            ),
            html.Div(id='plot_3d-switch-output')
        ]),
        html.Div([  # Div for HDBSCAN
            daq.ToggleSwitch(
                id='hdb_on',
                value=False,
                label='HDBSCAN on/off',
                labelPosition='top'
            ),
            html.Div(id='HDBSCAN on/off')
        ]),
        html.Div([  # Div for dr_plot
            dcc.Graph(id='dr_plot', config={'displayModeBar': True}, style={'width': '100%', 'height': '100vh', 'margin': '0 auto'},
                    responsive=True)],
            style={'clear': 'both'}
        ),



    ], style={'width': '70%', 'float': 'left', 'margin-bottom': '20px'}),  # Parent Div for features_cols and label_col

    html.Div([
        html.Label('Random seed:', style={'padding-right': '5px'}),
        dcc.Input(id='random_seed', type='text', value='4', style={'width': '50px'}),
        html.Div(id='error_message', style={'color': 'red'}),
        html.Div([
            html.Label('Components:'),
            html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}, children=[
                html.Span('x:', style={'margin-right': '10px', 'padding-left': '30px'}),
                dcc.Dropdown(
                    id='component_choice_x', 
                    options=[{'label': num_to_component_map[l], 'value': l} for l in num_to_component_map.keys()], 
                    value='Component 1',  # Default value for x
                    multi=False
                ),
            ]),
            html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}, children=[
                html.Span('y:', style={'margin-right': '10px', 'padding-left': '30px'}),
                dcc.Dropdown(
                    id='component_choice_y', 
                    options=[{'label': num_to_component_map[l], 'value': l} for l in num_to_component_map.keys()], 
                    value='Component 2',  # Default value for y
                    multi=False
                ),
            ]),
            html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}, children=[
                html.Span('z:', style={'margin-right': '10px', 'padding-left': '30px'}),
                dcc.Dropdown(
                    id='component_choice_z', 
                    options=[{'label': num_to_component_map[l], 'value': l} for l in num_to_component_map.keys()], 
                    value='Component 3',  # Default value for z
                    multi=False
                ),
            ]),
        ], style={'width': '30%'}),
    ], style={'width': '10%', 'float': 'left'}),

    html.Div([  # Div for point_size
        html.Label('Point Size:'),
        dcc.Slider(id='point_size', min=1, max=10, value=3, step=0.5, marks={i: str(i) for i in range(1, 11)})],
        style={'width': '40%', 'margin': '0 auto'}
    ),
    html.Div([  # Div for the map
        html.Label('Geo Cluster Map:'),
        html.Iframe(id='map', srcDoc=open('my_map.html', 'r').read(), width='100%', height='600')
    ], style={'width': '70%', 'float': 'left',  'margin-left': '260px', 'margin-top': '100px', 'margin-bottom': '100px'}),

    dcc.Store(id='intermediate_data'),

    html.Div(id='selected-data',
             style={
                'position': 'fixed',
                'top': '0',
                'right': '0',
                'bottom': '0',
                'width': '100px',
                'padding': '20px',
                'overflow-y': 'auto'
            })

], style={'width': '100%'})


# Create the initial plot object
fig = go.Figure()

@app.callback(
    Output('intermediate_data', 'data'),
    [Input('update_button', 'n_clicks')],
    [State('features_cols', 'value')],
    [State('dr_style', 'value')],
    [State('random_seed', 'value')]
)
def update_intermediate_data(n_clicks, features_cols, dr_style, value):

    global dr_style_flag

    if dr_style == 'umap':
        dr_style_flag = 'UMAP'
    elif dr_style == 'tsne':
        dr_style_flag = 't-SNE'
    elif dr_style == 'pca':
        dr_style_flag = 'PCA'

    if n_clicks is None:
        raise PreventUpdate

    cols = []
    for feature in features_cols:
        cols += column_dict[feature]

    plot_dataframe = performDR(cols, master_df, technique=dr_style, random_seed=int(value))

    data = plot_dataframe.to_dict('records')
    # Store both the data and features_cols in the dcc.Store
    return {"data": data, "features_cols": cols}


@app.callback(
    Output('error_message', 'children'),
    [Input('random_seed', 'value')]
)
def update_error_message(input_value):
    if input_value:
        try:
            val = int(input_value)
            return ''  # if the input is an integer, return an empty string
        except ValueError:
            return 'The input should be an integer!'
    else:
        return 'The input should not be empty!'

from dash.dependencies import Input, Output, State

# Callback to update dropdown options based on selected values
@app.callback(
    [Output('component_choice_x', 'options'),
     Output('component_choice_y', 'options'),
     Output('component_choice_z', 'options')],
    [Input('component_choice_x', 'value'),
     Input('component_choice_y', 'value'),
     Input('component_choice_z', 'value')]
)
def update_dropdown_options(x_val, y_val, z_val):
    all_options = [{'label': num_to_component_map[l], 'value': l} for l in num_to_component_map.keys()]
    
    # Remove selected values from options of other dropdowns
    x_options = [option for option in all_options if option['value'] != y_val and option['value'] != z_val]
    y_options = [option for option in all_options if option['value'] != x_val and option['value'] != z_val]
    z_options = [option for option in all_options if option['value'] != x_val and option['value'] != y_val]

    return x_options, y_options, z_options


@app.callback(
    Output('selected-data', 'children'),
    Input('dr_plot', 'selectedData')
)
def display_selected_data(selectedData):
    global labels_df

    if selectedData is not None:
        label_ids = [point['curveNumber']+1 for point in selectedData['points']]
        counts = [point['pointIndex']+1 for point in selectedData['points']]

        id_list = []

        for label_id, count in zip(label_ids, counts):
            id_list.extend(labels_df[(labels_df.iloc[:, 1] == label_id) & (labels_df['count'] == count)]['ID'])

        return [html.P(f"ID: {str(id)}") for id in id_list]
    return []


@app.callback(
    Output('dr_plot', 'figure'),
    [Input('intermediate_data', 'data'),
     Input('label_col', 'value'),
     Input('hover_cols', 'value'),
     Input('point_size', 'value'),
     Input('plot_3d', 'value'),
     Input('hdb_on', 'value'),
     Input('component_choice_x', 'value'),
     Input('component_choice_y', 'value'),
     Input('component_choice_z', 'value')]
)
def update_plot(intermediate_data, label_col, hover_cols, point_size, plot_3d, hdb_on, component_choice_x, component_choice_y, component_choice_z):

    global fig
    global labels_df

    comp_dict = {'x_comp': component_choice_x, 'y_comp' : component_choice_y, 'z_comp' : component_choice_z}

    if intermediate_data is None or label_col is None:
        return fig
    if hdb_on:
        data = intermediate_data['data']  
        features_cols = intermediate_data['features_cols']  

        plot_dataframe = pd.DataFrame(data)

        updated_fig, labels_df = createFigure(plot_3d=plot_3d, label_col="_HDBSCAN_", title=f'{dr_style_flag} dimensionality reduction plot', point_size=point_size, dr_df=plot_dataframe, master_df=master_df, features_cols=features_cols, hover_data=hover_cols, chosen_components = comp_dict)
        fig = updated_fig

        return fig

    else:
        data = intermediate_data['data']  
        features_cols = intermediate_data['features_cols'] 

        plot_dataframe = pd.DataFrame(data)

        updated_fig, labels_df = createFigure(plot_3d=plot_3d, label_col=label_col, title=f'{dr_style_flag} dimensionality reduction plot', point_size=point_size, dr_df=plot_dataframe, master_df=master_df, features_cols=features_cols, hover_data=hover_cols, chosen_components = comp_dict)
        fig = updated_fig

        return fig
# Run the app
if __name__ == '__main__':
    application.run(host='0.0.0.0', debug=True, use_reloader=False, port=8080)

