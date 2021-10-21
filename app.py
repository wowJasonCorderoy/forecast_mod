import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime

# random comment
#from traitlets.traitlets import default

st.set_page_config(page_title="Forecast Dashboard",
page_icon=":chart_with_upwards_trend:",
layout="wide")

### define session state
if 'df_changelog' not in st.session_state:
    st.session_state['df_changelog'] = pd.DataFrame(columns=['change_request_dt', 'date_from', 'date_to','variable','operation','operand','explanation'])
 
n_weeks = 52

today = date.today()
dates_list = [today+timedelta(days=x*7+7) for x in range(n_weeks)]

@st.cache()
def load_data(n=n_weeks):
    asp_ts = 10+np.random.normal(0,0.5,n)
    #qty_ts = 1000+np.random.normal(0,100,n)
    qty_ts = 1000*(1/(asp_ts/np.mean(asp_ts))+np.random.normal(0,0.05,n))
    sales_ts = asp_ts*qty_ts
    return asp_ts, qty_ts, sales_ts

@st.cache()
def gen_df():
    asp_ts,qty_ts,sales_ts = load_data()
    df = pd.DataFrame({'date':dates_list,'asp':asp_ts, 'qty':qty_ts, 'sales':sales_ts})
    #df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df

df = gen_df()

# post changes to forecast df
@st.cache()
def make_changelog_Changes(df=df, changelog=st.session_state.df_changelog):
    copy_df = df.copy()
    for i in range(len(st.session_state.df_changelog)):
        change_i = st.session_state.df_changelog.iloc[i]
        # split into a to be changed and not to be changed df
        changes_df = copy_df[(copy_df['date'] >= change_i['date_from']) & (copy_df['date'] <= change_i['date_to'])]
        no_changes_df = copy_df[~((copy_df['date'] >= change_i['date_from']) & (copy_df['date'] <= change_i['date_to']))]
        # implement the change in the data
        changes_df[change_i['variable']] = [eval(str(x)+change_i['operation']+str(change_i['operand'])) for x in changes_df[change_i['variable']]]
        # update the copy_df to the newly adjusted one
        copy_df = pd.concat([changes_df, no_changes_df])
        # update sales given qty and asp
        copy_df['sales'] = copy_df['qty']*copy_df['asp']
    return copy_df

df_changes = make_changelog_Changes()

df_combined = df.merge(df_changes.rename(columns={'sales':'adj_sales','asp':'adj_asp','qty':'adj_qty'}),how='left')
#df_combined_long = df_combined.melt(id_vars=['date'])

# f_date = st.sidebar.multiselect(
#     "Select date:",
#     df['date'].unique(),
# )

btn_refreshData = st.sidebar.button('refresh data')

st.sidebar.markdown("**Make adjustments:**")
f_date = st.sidebar.slider(
    "Select date:",
    value=(df['date'].min(), df['date'].max()),
    step=timedelta(days=7)
)

f_var = st.sidebar.selectbox(
    "Select variable(s):",
    options=['asp', 'qty'],
    #default=['asp'],
)

f_op = st.sidebar.selectbox(
    "Select operation:",
    options=['+', '-', '*', '/'],
    #default=['asp'],
)

f_val = st.sidebar.number_input(
    "Enter Operand:",
    value=1.00,
    step=0.01,
)

if ( (f_date != []) and (f_var != []) ):
    #st.sidebar.text(f"Function:\n{f_var} {f_op} {f_val}\n to be applied to dates:\n{[x.strftime('%Y-%m-%d') for x in f_date]}")
    st.sidebar.text(f"Function:\n{f_var} {f_op} {f_val}\n to be applied to dates:\n{f_date[0]} to {f_date[1]}")

f_explained = st.sidebar.text_input("Write an explanation")

# on submit update change log
# if changelog doesn't exist then create it else append submitted changes.

def onClick_submit_btn():
#if submit_btn:
    c_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    df_current_changes = (
        pd.DataFrame({'change_request_dt':[c_datetime]}).
        merge(pd.DataFrame({'date_from':[f_date[0]], 'date_to':[f_date[1]]}), how='cross').
        merge(pd.DataFrame({'variable':[f_var]}), how='cross').
        merge(pd.DataFrame({'operation':[f_op]}), how='cross').
        merge(pd.DataFrame({'operand':[f_val]}), how='cross').
        merge(pd.DataFrame({'explanation':[f_explained]}), how='cross')
        )
    # have to work with session state so that dateframe persists between reruns
    st.session_state.df_changelog = pd.concat([st.session_state.df_changelog, df_current_changes], ignore_index=True)

submit_btn = st.sidebar.button('submit change', on_click=onClick_submit_btn)

f_plotLines = st.multiselect('plot selection',
#options=['sales','qty','asp', 'adj_sales', 'adj_qty', 'adj_asp'],
options=['sales','qty','asp'],
default=['sales','qty','asp']
)

f_include_adjustments = st.checkbox('Incl. changelog adjustments')

if f_include_adjustments:
    f_plotLines= f_plotLines+['adj_'+x for x in f_plotLines]

if not f_include_adjustments:
    f_plotLines= [x for x in f_plotLines if 'adj_' not in x]

f_multiPlot_normalize = st.checkbox('Normalize')

def normalize_func(dat):
    return [x/(max(dat)-min(dat)) for x in dat]

if not f_multiPlot_normalize:
    fig_multi_data = df_combined.copy()
else: # normlize time!
    fig_multi_data = df_combined.copy()
    for i in f_plotLines:
        dat_tmp = fig_multi_data[i]
        maxi = max(dat_tmp)
        mini = min(dat_tmp)
        fig_multi_data[i] = [(x-mini)/(maxi-mini) for x in dat_tmp]
    # fig_multi_data['sales'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['sales']]
    # fig_multi_data['qty'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['qty']]
    # fig_multi_data['asp'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['asp']]

# If plot multiple plot is selected then 1 plot per 'plot selection' else just 1 plot
multiple_plots_checkbox = st.checkbox('1 plot per series')
if not multiple_plots_checkbox:
    fig_multi = px.line(fig_multi_data,
                x='date',
                y=f_plotLines,
                #hover_name='asp',
                color_discrete_map={'asp':'royalblue','qty':'orange','sales':'firebrick'},
                #line_dash_map={'asp':'dash','qty':'dash','sales':'dash','adj_asp':'dot','adj_qty':'dot','adj_sales':'dot'},
                title=f'')
    for i in range(len(fig_multi.data)):
        fig_multi.data[i].update(mode='markers+lines')
    fig_multi.update_layout(width=1000,height=350)
    st.plotly_chart(fig_multi)
else:
    for i in f_plotLines:
        if i.startswith('adj_'):
            continue
        i_c = [i]+['adj_'+i]
        fig_multi = px.line(fig_multi_data,#[['date',i]],
                x='date',
                y=i_c,
                #hover_name='asp',
                title=i)
        for i in range(len(fig_multi.data)):
            fig_multi.data[i].update(mode='markers+lines')
        fig_multi.update_layout(width=1000,height=350)
        st.plotly_chart(fig_multi)

st.sidebar.text("-"*15)
#st.sidebar.text("#Clean changelog:")
st.sidebar.markdown("**Clean changelog:**")

def try_or(fn, default):
    try:
        return fn()
    except:
        return default

d_index = st.sidebar.number_input(
    "Enter index to delete:",
    step=1,
    min_value=0,
    max_value=try_or(lambda: np.max(list(st.session_state.df_changelog.index)),0),
    format="%i",
)

def dl_ch_i():
    if dl_ch_btn:
        try:
            st.session_state.df_changelog = st.session_state.df_changelog.drop(d_index)
        except:
            pass

show_ch_btn = st.checkbox("show change log")
dl_ch_btn = st.sidebar.button("delete index from changelog", on_click=dl_ch_i)
st.sidebar.text("-"*15)

if show_ch_btn:
    st.dataframe(st.session_state.df_changelog)

st.download_button('Download changelog', data=st.session_state.df_changelog.to_csv(index=False), file_name='changelog.csv')

# st.dataframe(df_changes.sort_values('date'))

## hide streamlit style
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
