###################################################
# author: Wojciech Krolikowski (@detker)
# date: 31.07.2023
# version: 1.0
# about: web application presenting a 2D visualisation 
#           of polish words embedded vector space,
#           created with streamlit.
###################################################


import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu

from gensim.models import KeyedVectors

from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import curdoc
from bokeh.models import HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Greens9

# Computes euclidean_distance for two points in 2D space.
def euclidean_distance(A, B):
    return ((B[0]-A[0])**2 + (B[1]-A[1])**2) ** (1/2)

# Adding single word to local query.
def add_to_query(coords, word, query):
    word_idx = coords[coords['token'] == word].index[0]
    query.append(word_idx)

# Returning sample dataframe from 'single-words' query.
def create_sample(coords, query):
    return coords.iloc[query, :]

# Creating sample dataframe from 'word pairs' query.
def create_bisample(coords, query):
    # Slicing 'coords' sample containing query records.
    tmp = coords.iloc[query, :]
    tmp_x = tmp['x'].to_list()
    tmp_y = tmp['y'].to_list()
    tokens = tmp['token'].to_list()
    
    # Creating X, Y lists (for bokeh's multiline plot type) in specified format.
    X = [[tmp_x[i], tmp_x[i+1]] for i in range(0, len(tmp_x), 2)]
    Y = [[tmp_y[i], tmp_y[i+1]] for i in range(0, len(tmp_y), 2)]
    
    # Tokens merging and computing distances between word pairs.
    tokens_fixed = [tokens[i]+' - '+tokens[i+1] for i in range(0, len(tokens), 2)]
    distance = [euclidean_distance([X[i][0], Y[i][0]], [X[i][1], Y[i][1]]) for i in range(0, len(X))]
    
    # Creating 100 points between every pair of words to make bokeh's hover more flexible.
    X = [np.linspace(X[i][0], X[i][1], 100)[1:-1] for i in range(0, len(X))]
    Y = [np.linspace(Y[i][0], Y[i][1], 100)[1:-1] for i in range(0, len(Y))]
    
    # Inserting everything into a dataframe object.
    df = pd.DataFrame({'x': X, 'y': Y, 'distance': distance, 'tokens':tokens_fixed})
    
    return df, tmp

# Creating sample dataframe from 'k-nearest neighbours' queries.
def update_k_sample(sample, neighbours, coords):
    for neighbour, similarity in neighbours:
        # Getting the x and y values for each neighbour.
        x = coords[coords['token'] == neighbour]['x']
        y = coords[coords['token'] == neighbour]['y']
        # Inserting the data into a dataframe object for each neighbour
        new_row = pd.DataFrame({'x':x, 'y':y, 'token':neighbour, 'origin':word, 'similarity':similarity})
        # Concatenating every new record into a sample dataframe with all neighbours.
        sample = pd.concat([sample, new_row], axis=0, ignore_index=True)
        
    return sample

# Generates bokeh plot for each app feature.
def generate_plot(coords, opt):
    # Standard tools and hover init.
    hover = HoverTool(tooltips=[('word', '@token')])
    tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool()]
    
    # 'single words' plot.
    if opt==0:
        source = ColumnDataSource(coords)

        fig = figure(tools=tools, plot_width=500, plot_height=500, 
                    x_axis_label='x', y_axis_label='y')
        fig.scatter(x='x', y='y', source=source, size=10,
                fill_color='#40BF5A', line_color=None)
    
    # 'word pairs' plot.
    elif opt==1:
        coords_df, points_df = coords

        tools = [WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool()]
        source = ColumnDataSource(points_df)
        source2 = ColumnDataSource(coords_df)

        fig = figure(tools=tools, plot_width=500, plot_height=500, x_axis_label='x', y_axis_label='y')
        p2 = fig.multi_line(xs='x', ys='y', color='#40BF5A', line_width=3, source=source2)
        p1 = fig.circle(x='x', y='y', source=source, fill_color='white', line_color=None, size=10)
        
        # Adding hovers.
        hover2 = HoverTool(tooltips=[('words', '@tokens'), ('distance', '@distance')], renderers=[p2], attachment='left')
        fig.add_tools(hover2)
        hover = HoverTool(tooltips=[('word', '@token')], renderers=[p1], attachment='right')
        fig.add_tools(hover)
    
    # 'k-nearest neighbours' plot.
    elif opt==2:
        source = ColumnDataSource(coords)
        # Configuring color palette for data points.
        colors = factor_cmap('origin', palette=Greens9, factors=coords['origin'].unique())

        fig = figure(tools=tools, plot_width=500, plot_height=500, 
                    x_axis_label='x', y_axis_label='y')
        fig.scatter(x='x', y='y', source=source, size=10,
                line_color=None, fill_color=colors)
    
    # 'whole corpus' plot.
    else:
        source = ColumnDataSource(coords)
        hover = HoverTool(tooltips=[('word', '@token')], attachment='vertical')

        fig = figure(tools=tools,
                    x_axis_label='x', y_axis_label='y')
        fig.scatter(x='x', y='y', source=source, size=10,
                fill_color='#40BF5A', line_color=None)

    # Setting the plot's theme.
    doc = curdoc()
    doc.theme = 'dark_minimal'
    doc.add_root(fig)

    return fig

# Cleaning the query.
def reset_query(query, is_dual):
    print('reset!')
    st.session_state[query] = []
    # If 'k-nearest neighbour' also clean query with k values.
    if is_dual: st.session_state[str(query)+'k'] = []

# Load and cache the w2v model.
@st.cache_resource()
def load_model():
    return KeyedVectors.load('filtered_embedding.bin')


if __name__ == '__main__':

    # Hide streamlit menu and footer.
    hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Hyperlinks.
    embeddings_repo = '[<span style="color:#40BF5A;text-decoration:none">@sdadas</span>](https://github.com/sdadas/polish-nlp-resources)'
    project_repo = '[<span style="color:#40BF5A;text-decoration:none;font-weight:700">@detker</span>](https://github.com/detker/pl-word-embeddings-visualisation)'

    # Load dataframe with vector 2dim coordinates. 
    coords_df = pd.read_csv('coordsTSNE.csv')

    #######################################

    # App's header.
    st.markdown(project_repo, unsafe_allow_html=True)
    st.markdown("## n-dimensional vector word space - 2D visualisation")
    st.markdown("*polish language dataset, embedding done by " + embeddings_repo + ". dimensionality reduction using scikit t-SNE.*", unsafe_allow_html=True) #sorry :c

    # Menu's config.
    opt = option_menu(
        menu_title=None,
        options=['single words', 'word pairs', 'k-nearest neighbours', 'whole corpus'],
        icons=None,
        default_index=0,
        orientation='horizontal',
        styles={
            'nav-link': {'font-size':'12px', 'color': '#40BF5A', 'font-family':'monospace', 'margin':'0'},
            'nav-link-selected': {'color':'white'},
            'icon': {'display':'none'}
        }
    )

    if opt == 'single words':
        # Initialize query.
        if 'query' not in st.session_state:
            st.session_state['query'] = []
        
        st.markdown("<span style='color:#aeb0af'>write a word `x`, in order to display it and its relationships. write word `x` again to remove it from the chart.</span>", unsafe_allow_html=True)

        col1, col2 = st.columns([0.7, 0.2])

        with col2:
            word = st.text_input('input a word:')
            try:
                # If user have written the word.
                if word:
                    # If word does not exists already in a query.
                    if coords_df[coords_df['token'] == word].index[0] not in st.session_state['query']:
                        st.write('pushed', word, 'into vector space.')
                        add_to_query(coords_df, word, st.session_state['query'])
                    # If word does exists in query already.
                    else:
                        # Then remove it.
                        try: st.session_state['query'].remove(coords_df[coords_df['token'] == word].index[0])
                        except: pass
            except IndexError: st.write('this word does not exist in corpus. sorry!')

            # If user push reset button - reset the query.
            if st.button(label='reset'): reset_query('query', False)
            
            # Show the query but reversed.
            preview = coords_df.loc[st.session_state['query'], 'token']
            st.dataframe(preview.iloc[::-1], hide_index=True, use_container_width=True)

        with col1:
            # Create query sample and plot it.
            sample = create_sample(coords_df, st.session_state['query'])
            st.bokeh_chart(generate_plot(sample, 0))


    elif opt == 'word pairs':
        if 'query2' not in st.session_state:
            st.session_state['query2'] = []
        if 'ok' not in st.session_state:
            st.session_state['ok'] = None

        st.markdown("<span style='color:#aeb0af'>write words `x, y` separated by a comma, in order to display them and their relationships. write words `x, y` again in any order to remove them from the chart.</span>", unsafe_allow_html=True)

        col1, col2 = st.columns([0.7, 0.2])

        with col2:
            words = st.text_input('input words separated by a comma:')
            try:
                if words:
                    word1, word2 = words.split(',', maxsplit=1)
                    word1 = word1.strip()
                    word2 = word2.strip()
                    try:
                        if st.session_state['ok'] is None: st.session_state['ok'] = True
                        else: st.session_state['ok'] = not ({coords_df[coords_df['token'] == word1].index[0], coords_df[coords_df['token'] == word2].index[0]}.issubset(
                                set(st.session_state['query2'])))
                        
                        # If it's a first iteration for script or one of the words does not exists in a query.
                        if st.session_state['ok']:
                            st.write('pushed', word1, 'and', word2, 'into vector space.')
                            add_to_query(coords_df, word1, st.session_state['query2'])
                            add_to_query(coords_df, word2, st.session_state['query2'])
                        else:
                            try: 
                                st.session_state['query2'].remove(coords_df[coords_df['token'] == word1].index[0])
                                st.session_state['query2'].remove(coords_df[coords_df['token'] == word2].index[0])
                            except: pass
                    except IndexError: st.write('both(one) of the words does not exist(s) in corpus. sorry!')
            except ValueError: st.write('invalid input shape. please, try again!')
            
            if st.button(label='reset'): reset_query('query2', False)
            preview = coords_df.loc[st.session_state['query2'], 'token']
            st.dataframe(preview.iloc[::-1], hide_index=True, use_container_width=True)
            
        with col1:
            sample = create_bisample(coords_df, st.session_state['query2'])
            st.bokeh_chart(generate_plot(sample, 1))


    elif opt == 'k-nearest neighbours':
        if 'query3' not in st.session_state:
            st.session_state['query3'] = []
            st.session_state['query3k'] = []

        st.markdown("<span style='color:#aeb0af'>write a word `x` and an integer `k` separated by a comma, in order to display the word and its `k` neighbours. write word `x` and any integer `i` in correct order to remove it and its neigbours from the chart.</span>", unsafe_allow_html=True)
        
        # Loading the w2v model.
        w2v_model = load_model()
        col1, col2 = st.columns([0.7, 0.2])
        
        with col2:
            wordk = st.text_input('input word and k neighbours separated by a comma')
            try:
                if wordk:
                    word, k = wordk.split(',', maxsplit=1)
                    word = word.strip()
                    k = int(k.strip())
                    try: 
                        if coords_df[coords_df['token'] == word].index[0] not in st.session_state['query3']:
                            st.write('pushed', word, 'and', k, 'neighbours into vector space.')
                            add_to_query(coords_df, word, st.session_state['query3'])
                            st.session_state['query3k'].append(k)
                        else:
                            try: 
                                st.session_state['query3'].remove(coords_df[coords_df['token'] == word].index[0])
                                st.session_state['query3k'].remove(k)
                            except: pass 
                    except IndexError: st.write('word does not exists in the corpus. sorry!')
            except ValueError: st.write('invalid input shape. please, try again!')
            
            
            if st.button(label='reset'): reset_query('query3', True)
            preview = coords_df.loc[st.session_state['query3'], 'token']
            st.dataframe(preview.iloc[::-1], hide_index=True, use_container_width=True)
            
        with col1:
            sample = pd.DataFrame({'x': [], 'y': [], 'token': [], 'origin': [], 'similarity': []})
            # For every word in query.
            for idx, word in enumerate(coords_df.loc[st.session_state['query3'], 'token'].to_list()):
                # Computing most similar words.
                neighbours = w2v_model.most_similar(word, topn=st.session_state['query3k'][idx])
                # Appending the word itself.
                neighbours.append((word, None))
                sample = update_k_sample(sample, neighbours, coords_df)
            st.bokeh_chart(generate_plot(sample, 2))
            

    else:
        st.markdown("<span style='color:#aeb0af'>whole corpus. explore it as you wish :).</span>", unsafe_allow_html=True)
        # Plot the whole corpus.
        st.bokeh_chart(generate_plot(coords_df, 3))