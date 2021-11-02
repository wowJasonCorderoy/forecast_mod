import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime

# from traitlets.traitlets import default

# st.set_page_config(
#     page_title="Forecast Dashboard",
#     page_icon=":chart_with_upwards_trend:",
#     layout="wide",
# )


st.set_page_config(
    page_title="Forecast Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

### define session state
if "df_changelog" not in st.session_state:
    st.session_state["df_changelog"] = pd.DataFrame(
        columns=[
            "change_request_dt",
            "date_from",
            "date_to",
            "variable",
            "operation",
            "operand",
            "explanation",
            "dept",
            "cat",
            "subcat",
            "segment",
            "article",
        ]
    )


# @st.cache()
# def gen_df():
    # dtypes = {"asp": np.float64, "sales": np.float64, "qty": np.float64}
    # parse_dates = ["date"]
    # df = pd.read_csv("./data/data.csv", dtype=dtypes, parse_dates=parse_dates)
    # return df

# df_all = gen_df()

with st.expander("File Upload", True):
    uploaded_file=st.file_uploader("Choose a (csv) file (Must contain the following columns 'article', 'category', 'subcategory', 'segment', 'department', 'pricefamily', 'date', 'sales', 'qty', 'asp')",
    type='csv',)

    if uploaded_file is not None:
        dtypes = {"asp": np.float64, "sales": np.float64, "qty": np.float64}
        parse_dates = ["date"]
        df_all = pd.read_csv(uploaded_file, dtype=dtypes, parse_dates=parse_dates)

if "all_f_dept" not in st.session_state:
    st.session_state["all_f_dept"] = df_all["department"].unique()
    st.session_state["all_f_cat"] = df_all["category"].unique()
    st.session_state["all_f_subcat"] = df_all["subcategory"].unique()
    st.session_state["all_f_segment"] = df_all["segment"].unique()
    st.session_state["all_f_article"] = df_all["article"].unique()
    # Initialise filtered values as all possible. e.g. f_cat = filtered categories whereas all_f_cat = all possible selections.
    st.session_state["f_dept"] = df_all["department"].unique()
    st.session_state["f_cat"] = df_all["category"].unique()
    st.session_state["f_subcat"] = df_all["subcategory"].unique()
    st.session_state["f_segment"] = df_all["segment"].unique()
    st.session_state["f_article"] = df_all["article"].unique()

# post changes to forecast df
@st.cache()
def make_changelog_Changes(df=df_all, changelog=st.session_state.df_changelog):
    copy_df = df_all.copy()
    for i in range(len(st.session_state.df_changelog)):
        change_i = st.session_state.df_changelog.iloc[i]
        # split into a to be changed and not to be changed df
        # changes_df = copy_df[(copy_df['date'] >= change_i['date_from']) & (copy_df['date'] <= change_i['date_to'])]
        # no_changes_df = copy_df[~((copy_df['date'] >= change_i['date_from']) & (copy_df['date'] <= change_i['date_to']))]
        changes_df = copy_df[
            (
                (copy_df["department"].isin(st.session_state["f_dept"]))
                & (copy_df["category"].isin(st.session_state["f_cat"]))
                & (copy_df["subcategory"].isin(st.session_state["f_subcat"]))
                & (copy_df["segment"].isin(st.session_state["f_segment"]))
                & (copy_df["article"].isin(st.session_state["f_article"]))
                & (copy_df["date"] >= pd.to_datetime(change_i["date_from"]))
                & (copy_df["date"] <= pd.to_datetime(change_i["date_to"]))
            )
        ]
        # no_changes_df = copy_df[~((copy_df['date'] >= pd.to_datetime( change_i['date_from'])) & (copy_df['date'] <= pd.to_datetime(change_i['date_to'])))]
        no_changes_df = copy_df[
            ~(
                (copy_df["department"].isin(st.session_state["f_dept"]))
                & (copy_df["category"].isin(st.session_state["f_cat"]))
                & (copy_df["subcategory"].isin(st.session_state["f_subcat"]))
                & (copy_df["segment"].isin(st.session_state["f_segment"]))
                & (copy_df["article"].isin(st.session_state["f_article"]))
                & (copy_df["date"] >= pd.to_datetime(change_i["date_from"]))
                & (copy_df["date"] <= pd.to_datetime(change_i["date_to"]))
            )
        ]
        # implement the change in the data
        changes_df[change_i["variable"]] = [
            eval(str(x) + change_i["operation"] + str(change_i["operand"]))
            for x in changes_df[change_i["variable"]]
        ]
        # update the copy_df to the newly adjusted one
        copy_df = pd.concat([changes_df, no_changes_df])
        # update sales given qty and asp
        copy_df["sales"] = copy_df["qty"] * copy_df["asp"]
    return copy_df


df_changes = make_changelog_Changes()

# df_combined combines original df and changes
df_combined = df_all.merge(
    df_changes.rename(
        columns={"sales": "adj_sales", "asp": "adj_asp", "qty": "adj_qty"}
    ),
    how="left",
)
# df_combined_long = df_combined.melt(id_vars=['date'])


#btn_refreshData = st.sidebar.button("refresh data")

# st.sidebar.markdown("**Filter Data:**")

with st.expander("Filter data:"):
    st.session_state["f_dept"] = st.multiselect(
        "Select Department:",
        options=st.session_state["all_f_dept"],  # top level options = all
        default=st.session_state["all_f_dept"],
    )

    # update category options based on department selections
    st.session_state["f_cat"] = df_all[
        df_all["department"].isin(st.session_state["f_dept"])
    ]["category"].unique()

    st.session_state["f_cat"] = st.multiselect(
        "Select Category:",
        options=st.session_state["f_cat"],
        default=st.session_state["f_cat"],
    )

    # update subcategory options based on department and category selections
    st.session_state["f_subcat"] = df_all[
        (df_all["department"].isin(st.session_state["f_dept"]))
        & (df_all["category"].isin(st.session_state["f_cat"]))
    ]["subcategory"].unique()

    st.session_state["f_subcat"] = st.multiselect(
        "Select Sub-Category:",
        options=st.session_state["f_subcat"],
        default=st.session_state["f_subcat"],
    )

    # update segment options based on department and category and subcat selections
    st.session_state["f_segment"] = df_all[
        (df_all["department"].isin(st.session_state["f_dept"]))
        & (df_all["category"].isin(st.session_state["f_cat"]))
        & (df_all["subcategory"].isin(st.session_state["f_subcat"]))
    ]["segment"].unique()

    st.session_state["f_segment"] = st.multiselect(
        "Select Segment:",
        options=st.session_state["f_segment"],
        default=st.session_state["f_segment"],
    )

    # update article options based on department and category and cubcat and segment selections
    st.session_state["f_article"] = df_all[
        (df_all["department"].isin(st.session_state["f_dept"]))
        & (df_all["category"].isin(st.session_state["f_cat"]))
        & (df_all["subcategory"].isin(st.session_state["f_subcat"]))
        & (df_all["segment"].isin(st.session_state["f_segment"]))
    ]["article"].unique()

    st.session_state["f_article"] = st.multiselect(
        "Select Article:",
        options=st.session_state["f_article"],
        default=st.session_state["f_article"],
    )


# df_filtered is the df based on dropdown selections
df_filtered = df_combined[
    (df_combined["department"].isin(st.session_state["f_dept"]))
    & (df_combined["category"].isin(st.session_state["f_cat"]))
    & (df_combined["subcategory"].isin(st.session_state["f_subcat"]))
    & (df_combined["segment"].isin(st.session_state["f_segment"]))
    & (df_combined["article"].isin(st.session_state["f_article"]))
]

# df is the aggregated dataframe for plotting
df = (
    df_filtered.groupby("date")
    .agg({"qty": "sum", "sales": "sum", "adj_qty": "sum", "adj_sales": "sum"})
    .reset_index()
    .sort_values("date")
    .assign(
        asp=lambda df: df.sales / df.qty, adj_asp=lambda df: df.adj_sales / df.adj_qty
    )
)

with st.expander("Scenario modelling:"):
    #st.sidebar.markdown("**Make adjustments:**")
    f_date = st.slider(
        "Select date:",
        # value=(min(df['date']), max(df['date'])),
        value=(datetime.date(min(df["date"])), datetime.date(max(df["date"]))),
        step=timedelta(days=7),
    )

    f_var = st.selectbox(
        "Select variable(s):",
        options=["asp", "qty"],
        # default=['asp'],
    )

    f_op = st.selectbox(
        "Select operation:",
        options=["*", "/", "+", "-"],
        # default=['asp'],
    )

    f_val = st.number_input(
        "Enter Operand:",
        value=1.00,
        step=0.01,
    )

    #if (f_date != []) and (f_var != []):
        ## st.sidebar.text(f"Function:\n{f_var} {f_op} {f_val}\n to be applied to dates:\n{[x.strftime('%Y-%m-%d') for x in f_date]}")
        #st.sidebar.text(
            #f"Function:\n{f_var} {f_op} {f_val}\n to be applied to dates:\n{f_date[0]} to {f_date[1]}"
        #)

    f_explained = st.text_input("Write an explanation")

# on submit update change log
# if changelog doesn't exist then create it else append submitted changes.


    def onClick_submit_btn():
        # if submit_btn:
        c_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_current_changes = (
            pd.DataFrame({"change_request_dt": [c_datetime]})
            .merge(
                pd.DataFrame({"date_from": [f_date[0]], "date_to": [f_date[1]]}),
                how="cross",
            )
            .merge(pd.DataFrame({"variable": [f_var]}), how="cross")
            .merge(pd.DataFrame({"operation": [f_op]}), how="cross")
            .merge(pd.DataFrame({"operand": [f_val]}), how="cross")
            .merge(pd.DataFrame({"explanation": [f_explained]}), how="cross")
            .merge(pd.DataFrame({"dept": [st.session_state["f_dept"]]}), how="cross")
            .merge(pd.DataFrame({"cat": [st.session_state["f_cat"]]}), how="cross")
            .merge(pd.DataFrame({"subcat": [st.session_state["f_subcat"]]}), how="cross")
            .merge(pd.DataFrame({"segment": [st.session_state["f_segment"]]}), how="cross")
            .merge(pd.DataFrame({"article": [st.session_state["f_article"]]}), how="cross")
        )
        # have to work with session state so that dateframe persists between reruns
        st.session_state.df_changelog = pd.concat(
            [st.session_state.df_changelog, df_current_changes], ignore_index=True
        )

    submit_btn = st.button("submit change", on_click=onClick_submit_btn)

with st.expander("Plot controls:"):
    f_plotLines = st.multiselect(
        "plot selection",
        # options=['sales','qty','asp', 'adj_sales', 'adj_qty', 'adj_asp'],
        options=["sales", "qty", "asp"],
        default=["sales", "qty", "asp"],
    )

    f_include_adjustments = st.checkbox("Incl. changelog adjustments", value=True)

    if f_include_adjustments:
        f_plotLines = f_plotLines + ["adj_" + x for x in f_plotLines]

    if not f_include_adjustments:
        f_plotLines = [x for x in f_plotLines if "adj_" not in x]

    f_multiPlot_normalize = st.checkbox("Normalize")

    multiple_plots_checkbox = st.checkbox("1 plot per series", value=True)


def normalize_func(dat):
    return [x / (max(dat) - min(dat)) for x in dat]


if not f_multiPlot_normalize:
    fig_multi_data = df.copy()
else:  # normlize time!
    fig_multi_data = df.copy()
    for i in f_plotLines:
        dat_tmp = fig_multi_data[i]
        maxi = max(dat_tmp)
        mini = min(dat_tmp)
        fig_multi_data[i] = [(x - mini) / (maxi - mini) for x in dat_tmp]
    # fig_multi_data['sales'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['sales']]
    # fig_multi_data['qty'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['qty']]
    # fig_multi_data['asp'] = [x/(max(dat)-min(dat)) for x in fig_multi_data['asp']]

# If plot multiple plot is selected then 1 plot per 'plot selection' else just 1 plot

with st.expander("View line plots:", True):
    if not multiple_plots_checkbox:
        fig_multi = px.line(
            fig_multi_data,
            x="date",
            y=f_plotLines,
            # hover_name='asp',
            color_discrete_map={"asp": "royalblue", "qty": "orange", "sales": "firebrick"},
            # line_dash_map={'asp':'dash','qty':'dash','sales':'dash','adj_asp':'dot','adj_qty':'dot','adj_sales':'dot'},
            title=f"",
        )
        for i in range(len(fig_multi.data)):
            fig_multi.data[i].update(mode="markers+lines")
        fig_multi.update_layout(width=1000, height=350)
        st.plotly_chart(fig_multi)
    else:
        for i in f_plotLines:
            if i.startswith("adj_"):
                continue
            if f_include_adjustments:
                i_c = [i] + ["adj_" + i]
            else:
                i_c = [i]
            fig_multi = px.line(
                fig_multi_data,  # [['date',i]],
                x="date",
                y=i_c,
                # hover_name='asp',
                title=i,
            )
            for i in range(len(fig_multi.data)):
                fig_multi.data[i].update(mode="markers+lines")
            fig_multi.update_layout(width=1000, height=350)
            st.plotly_chart(fig_multi)


# if show_ch_btn:
# st.dataframe(st.session_state.df_changelog)


with st.expander("Changelog:"):

    uploaded_file_changelog=st.file_uploader("Upload changelog (JSON)", type='json')

    if uploaded_file_changelog is not None:
        st.session_state.df_changelog = pd.read_json(uploaded_file_changelog)

    st.download_button(
        "Download changelog (JSON)",
        #data=st.session_state.df_changelog.to_csv(index=True),
        data=st.session_state.df_changelog.to_json(),
        file_name="changelog.json",
    )

    # show_ch_btn = st.checkbox("show change log")

    # if show_ch_btn:
    # st.dataframe(st.session_state.df_changelog)

    st.dataframe(st.session_state.df_changelog)

#with st.expander("Edit Changelog:"):

    def try_or(fn, default):
        try:
            return fn()
        except:
            return default

    d_index = st.number_input(
        "Enter index to delete:",
        step=1,
        min_value=0,
        max_value=try_or(lambda: np.max(list(st.session_state.df_changelog.index)), 0),
        format="%i",
    )

    def dl_ch_i():
        if dl_ch_btn:
            try:
                st.session_state.df_changelog = st.session_state.df_changelog.drop(
                    d_index
                )
            except:
                pass

    # show_ch_btn = st.checkbox("show change log")
    dl_ch_btn = st.button("delete index from changelog", on_click=dl_ch_i)


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

