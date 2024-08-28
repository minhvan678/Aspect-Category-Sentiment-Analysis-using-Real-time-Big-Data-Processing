import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Aspect Category Sentiment Analysis on Traveloka",
    page_icon="✅",
    layout="wide",
)

# if not "sleep_time" in st.session_state:
#     st.session_state.sleep_time = 2

# if not "auto_refresh" in st.session_state:
#     st.session_state.auto_refresh = True

# auto_refresh = st.sidebar.checkbox('Auto Refresh?', st.session_state.auto_refresh)

# if auto_refresh:
#     number = st.sidebar.number_input('Refresh rate in seconds', value=st.session_state.sleep_time)
#     st.session_state.sleep_time = number

if not "aspect" in st.session_state:
    st.session_state.aspect = 1
if not "hotel" in st.session_state:
    st.session_state.hotel = 'diamond bay hotel nha trang'

# @st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

def change():
    st.session_state.aspect = st.session_state.kind_of_aspect
    st.session_state.region = st.session_state.name_of_region



df = load_data('./export_data_test.csv')

# dashboard title
st.title("Dashboard Aspect Category Sentiment Analysis on Traveloka")

# top-level filters

hotel_dict = {'Classy Boutique Hotel' : 'Classy Boutique Hotel',
              'Milan Homestay - The Song Vung Tau' : 'Milan Homestay - The Song Vung Tau',
              'Seashore Hotel & Apartment' : 'Seashore Hotel & Apartment',
              'Classy Holiday Hotel & Spa' : 'Classy Holiday Hotel & Spa',
              'Khách sạn The Grace Dalat': 'Khách sạn The Grace Dalat'}

# hotel_dict = {'Khách sạn Poetic Huế' : 'Khách sạn Poetic Huế',
#               'Volga Hotel Nha Trang' : 'Volga Hotel Nha Trang',
#               'Au Lac Charner Hotel' : 'Au Lac Charner Hotel',
#               'Hotel La Perle' : 'Hotel La Perle',
#               'Quy Nhon Hotel': 'Quy Nhon Hotel'}

hotel = st.selectbox("Select the Hotel", list(hotel_dict.keys()), on_change=change, key='name_of_region')


aspect = st.selectbox("Select the Aspect", pd.unique(df["aspect"]), on_change=change, key='kind_of_aspect')
# creating a single-element container
placeholder = st.empty()

# dataframe filter


df_select = df[(df['hotel_name'] == hotel_dict[hotel]) & (df['aspect'] == aspect)]
df_select.reset_index(drop=True, inplace=True)
data_chart = df_select.groupby(by = ['time','hotel_name','aspect']).sum().reset_index()

df_select_hotel = df[(df['hotel_name'] == hotel_dict[hotel])]
all_data_chart = df_select_hotel.groupby(by = ['time','hotel_name']).sum().reset_index()

hotel_sum_pos = all_data_chart['Positive'].sum()
hotel_sum_neu = all_data_chart['Neutral'].sum()
hotel_sum_neg = all_data_chart['Negative'].sum()

sum_pos = data_chart['Positive'].sum()
sum_neu = data_chart['Neutral'].sum()
sum_neg = data_chart['Negative'].sum()


with placeholder.container():
    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)
    try:
        kpi1.metric(
            label="Positive ⏳",
            value=sum_pos,
            delta=int(data_chart['Positive'].iloc[-1] - data_chart['Positive'].iloc[-2]),
        )
    except:
        kpi1.metric(
            label="Positive ⏳",
            value=sum_pos,
            delta=int(0),
        )

    try:
        kpi2.metric(
            label="Neutral ⏳",
            value=sum_neu,
            delta=int(data_chart['Neutral'].iloc[-1] - data_chart['Neutral'].iloc[-2]),
        )
    except:
        kpi2.metric(
            label="Neutral ⏳",
            value=sum_neu,
            delta=int(0),
        )
    try:
        kpi3.metric(
            label="Negative ⏳",
            value=sum_neg,
            delta=int(data_chart['Negative'].iloc[-1] - data_chart['Negative'].iloc[-2]),
        )
    except:
        kpi3.metric(
            label="Negative ⏳",
            value=sum_neg,
            delta=int(0),
        )

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### Total Sentiment in Hotel")
        fig = go.Figure(data=(go.Pie(labels=['Positive','Neutral','Negative'], values=[hotel_sum_pos, hotel_sum_neu, hotel_sum_neg])),)
        fig.update_layout(width=700, height=500, legend=dict(font=dict(size=20)), legend_traceorder='normal')
        fig.update_traces(marker=dict(colors=['#4169E1',"#FFA500","#32CD32"]))
        st.write(fig)
        
    with fig_col2:
        st.markdown("### Positive Aspect Sentiment in Hotel")
        fig2 = px.line(data_frame=data_chart, x = 'time', y=['Positive'], labels={"value": "count", "index": "time"},
                       width=900, height=600,color_discrete_sequence=['#FF6961'])
        fig2.update_layout(showlegend=False)
        # fig2.update_layout(legend=dict(font=dict(size=20)))
        # fig2.update_traces(marker=dict())
        st.write(fig2)

    fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        st.markdown("### Neutral Aspect Sentiment in Hotel")
        fig3 = px.line(data_frame=data_chart, x = 'time', y=['Neutral'], labels={"value": "count", "index": "time"},
                       width=900, height=600,color_discrete_sequence=['#FFF44C'])
        fig3.update_layout(showlegend=False)
        # fig3.update_layout(legend=dict(font=dict(size=20)))
        # fig2.update_traces(marker=dict())
        st.write(fig3)
        
    with fig_col4:
        st.markdown("### Negative Aspect Sentiment in Hotel")
        fig4 = px.line(data_frame=data_chart, x = 'time', y=['Negative'], labels={"value": "count", "index": "time"},
                       width=900, height=600,color_discrete_sequence=["#288EEB"])
        fig4.update_layout(showlegend=False)
        # fig4.update_layout(legend=dict(font=dict(size=20)))
        # fig2.update_traces(marker=dict())
        st.write(fig4)


# if auto_refresh:
#     time.sleep(number)
#     st.experimental_rerun()


time.sleep(25)
st.experimental_rerun()
