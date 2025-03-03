import  streamlit as st
import pickle
import numpy as np

dataset = pickle.load(open("T20_dataset.pkl", "rb"))
pipe = pickle.load(open("t20_pipe.pkl", "rb"))



st.title("T20 SCORE PREDICTION")


#innings
inn = st.selectbox("Innings"  , dataset["innings"].unique())



#city
city = st.selectbox("City"  , dataset["City"].unique())


#batting and bowling teams
col1 , col2  = st.columns(2)
with col1:
    bat = st.selectbox("Batting Team" , dataset["Batting Team"].unique())

with col2:
    ball = st.selectbox("Bowling Team", dataset["Bowling Team"].unique())



#current score and wicket left
col1 , col2  = st.columns(2)
with col1:
    c_score = st.number_input("Current Score" , max_value= 300 , min_value=0  )

with col2:
    w_remain = st.number_input("Wickets Remaining" , min_value=0 , max_value=10)



#current score and wicket left
col1 , col2  = st.columns(2)
with col1:
    r_5 = st.number_input("Runs Scored in Last 5 overs" , min_value=0 , max_value=150)

with col2:
    w_5 = st.number_input("Wickets Lost in Last 5 overs" , min_value=0 , max_value=10)



overs_comp = st.number_input("Overs Completed" ,min_value=0.0, max_value=20.0)


if st.button("Predict Score" ) :

    #crr
    runs = c_score
    overs = overs_comp

    overs_dec = overs - int(overs)

    x = overs_dec * 10
    y = x * 1.66
    z = y / 10

    overs = int(overs) + z
    crr = runs / overs

    query = np.array([bat , ball , city , c_score , crr , w_remain , r_5 , w_5 , inn , overs_comp])

    query = query.reshape(1, 10)

    st.title("Score : " + str(int((pipe.predict(query)[0]))) )


