import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# in real prod scenario, should do this inside the main.py of model folder for example, then save to anothe object (like those pickle files) adn then here just import it, but here is quite small data though
def get_clean_data():
    data = pd.read_csv("data/data.csv") # just need to run the csv file from data folder in this streamlit root folder so no need longer path
    
    # drop not-needed columns, cleaning etc
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    #from the diagnosis column, change value of M,B (ac tinh vs lanh tinh) to 0,1 by mapping from a dictionary
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

def add_sidebar():
    st.sidebar.header("Cell nuclei measurements ")

    data = get_clean_data()

    # make a list to create many sliders (taking Xn as input) in the streamlit app
    slider_labels = [
        ("Radius (mean)", "radius_mean"), # these 2 values in the list is the key* (1) label in the streamlit app is Radius (mean) - name of col in data is radius_mean, we can access the max value from this column
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),
        ("Radius (SE)", "radius_se"),
        ("Texture (SE)", "texture_se"),
        ("Perimeter (SE)", "perimeter_se"),
        ("Area (SE)", "area_se"),
        ("Smoothness (SE)", "smoothness_se"),
        ("Compactness (SE)", "compactness_se"),
        ("Concavity (SE)", "concavity_se"),
        ("Concave Points (SE)", "concave points_se"),
        ("Symmetry (SE)", "symmetry_se"),
        ("Fractal Dimension (SE)", "fractal_dimension_se"),
        ("Radius (Worst)", "radius_worst"),
        ("Texture (Worst)", "texture_worst"),
        ("Perimeter (Worst)", "perimeter_worst"),
        ("Area (Worst)", "area_worst"),
        ("Smoothness (Worst)", "smoothness_worst"),
        ("Compactness (Worst)", "compactness_worst"),
        ("Concavity (Worst)", "concavity_worst"),
        ("Concave Points (Worst)", "concave points_worst"),
        ("Symmetry (Worst)", "symmetry_worst"),
        ("Fractal Dimension (Worst)", "fractal_dimension_worst")
    ]


    # create a dictionary to store and pair the value inside each slider to visualize
    input_dict = {}


    # loop through those labels above and create a slider for each of them
    # key = ("Radius (mean)", "radius_mean") -- 1 key from the list above
    # for each key in the list slider_labels created above, create the slider

    for label, key in slider_labels:
        # the slider is inside the sidebar so put like this
        input_dict[key] = st.sidebar.slider(
            label=label, #take the first element from the key to use as label
            min_value=float(0),
            max_value=float(data[key].max()), # parse in the key from the slider_labels list above, and take the max value - convert to float to be sure
            value=float(data[key].mean()) # default
        )   
    return input_dict

# to scale all values in the radar chart in a way that they are somehow between 0 n 1 -- acn do with scikit learn or now do by hand
def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for  key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    # use example from plotly just change to use our data https://plotly.com/python/radar-chart/ -- Multiple Trace Radar Chart -- need to instal plotly
    categories = ['Radius', 'Texture', 'Perimeter','Area','Smoothness','Compactness','Concavity','Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    # do this for each of the categories, with this dataset there are mean, SE, worst
    
    # mean
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], 
            input_data['smoothness_mean'], input_data['compactness_mean'], input_data['concavity_mean'], 
            input_data['concave points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself', # for color
        name='Mean Value'
    ))
    
    #SE
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'], 
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'], 
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard error'
    ))
    
    # worst
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'], 
            input_data['smoothness_worst'], input_data['compactness_worst'], input_data['concavity_worst'], 
            input_data['concave points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst value'
    ))


    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1] #as we have scaled data already
        )),
    showlegend=True
    )

    return fig


def add_predictions(input_data):
    #import the model from pkl
    model = pickle.load(open("model/model.pkl", "rb")) # rb = read mode n binary mode
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    #convert the input data which is now a dictionary with key/value pair to a single numpy array 
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    # Adding header
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")


    # as we scale value to between 0 and 1, if 0 then it is lanh tinh Benign
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    # probability of Benign OR Malicious
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


    # st.write(input_array)


def main():
    # set page configuration for our app
    st.set_page_config(
        page_title="Breast cancer predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()


    # create a container to contain different elements
    with st.container():
        #create H1 header
        st.title("Breast cancer predictor")

        #create p element - paragraph
        st.write("Please connect this app to your cytology app to help diagnose breast cancer from the tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurement by hand using the sliders in the sidebar")

    # create the columns
    col1, col2 = st.columns([4,1]) # how many cols do you want and the ratio between these cols -- the first col is 4 times bigger than the second col

    #write inside the columns using with func from py
    with col1:
        #st.write("This is column 1")
        # parse the function taking the input data as argument for visualization
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        # st.write("This is column 2")
        # parse the function for prediction
        add_predictions(input_data)

# always need this to test/ensure this is the correct file that is executed 
if __name__ == '__main__':
    main()

# to run this in a browser
# streamlit run app/main.py