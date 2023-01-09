import json

import numpy as np
import streamlit as st
import pandas as pd
import altair as alt


from load import load_tasks_data
from difficulty import (
    get_difficulty_per_trial,
    convert_difficulties_to_quantiles,
    quantize_difficulties,
)
from accuracy import get_accuracy_per_trial, get_accuracy_per_model
from agent_characteristic import get_logistic_agent_characteristic, get_auc


get_accuracy_per_model_with_cache = st.cache(get_accuracy_per_model)

tasks = load_tasks_data()


def render_dashboard():
    st.selectbox("Task", options=tasks, key="task")

    task = st.session_state["task"]

    st.multiselect("Models", options=tasks[task]["models"], key="models")

    instance_accuracy_per_model = get_accuracy_per_model_with_cache(task)

    model_accuracy_per_trial = get_accuracy_per_trial(instance_accuracy_per_model)

    st.selectbox("Plot type", options=["Logistic fit", "Binned"], key="plot_type")

    if st.session_state["plot_type"] == "Binned":
        st.number_input("Number of bins", value=5, min_value=2, key="num_bins")

    st.selectbox(
        "X-axis", options=["Difficulty quantile", "Raw difficulty"], key="x_axis"
    )

    instance_difficulties = get_difficulty_per_trial(
        model_accuracy_per_trial, exclude_models=st.session_state["models"]
    )

    if st.session_state["x_axis"] == "Difficulty quantile":
        difficulty_column_name = "Difficulty quantile"

    else:
        difficulty_column_name = "Difficulty"

    if st.session_state["plot_type"] == "Logistic fit":
        create_logistic_acc_plot(
            instance_accuracy_per_model, instance_difficulties, difficulty_column_name
        )

    elif st.session_state["plot_type"] == "Binned":
        create_binned_acc_plot(
            instance_accuracy_per_model, instance_difficulties, difficulty_column_name
        )
    create_auc_plot(instance_accuracy_per_model, instance_difficulties)

def create_auc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
):
    df = get_df_for_auc_plot(instance_accuracy_per_model, instance_difficulties)
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="log(params)",
            y="AUC",
        )
    )
    st.altair_chart(chart, use_container_width=True)

def get_df_for_auc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
):
    with open("models.json") as f:
        params_per_model = json.load(f)
    df_dict = {"log(params)": [], "AUC": []}
    for model_name in st.session_state["models"]:
        correct = {
            result["id"]: result["is_correct"]
            for result in instance_accuracy_per_model[model_name]
        }
        xs, ys = get_logistic_agent_characteristic(
            instance_difficulties,
            correct,
            quantiles=st.session_state["x_axis"] == "Difficulty quantile",
        )
        auc = get_auc(xs, ys)
        df_dict["log(params)"].append(np.log10(params_per_model[model_name]))
        df_dict["AUC"].append(auc)
    return pd.DataFrame(df_dict)


def create_logistic_acc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
    difficulty_column_name: str,
):
    df = get_df_for_logistic_acc_plot(
        instance_accuracy_per_model, instance_difficulties, difficulty_column_name
    )
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=difficulty_column_name,
            y="P(correct)",
            color="model",
        )
    )
    st.altair_chart(chart, use_container_width=True)


def get_df_for_logistic_acc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
    difficulty_column_name: str,
):
    df_dict = {difficulty_column_name: [], "model": [], "P(correct)": []}
    for model_name in st.session_state["models"]:
        correct = {
            result["id"]: result["is_correct"]
            for result in instance_accuracy_per_model[model_name]
        }
        xs, ys = get_logistic_agent_characteristic(
            instance_difficulties,
            correct,
            quantiles=st.session_state["x_axis"] == "Difficulty quantile",
        )
        df_dict[difficulty_column_name] += xs.tolist()
        df_dict["model"] += [model_name] * len(xs)
        df_dict["P(correct)"] += ys.tolist()
    return pd.DataFrame(df_dict)


def create_binned_acc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
    difficulty_column_name: str,
):
    df = get_df_for_binned_acc_plot(
        instance_accuracy_per_model,
        instance_difficulties,
        difficulty_column_name,
    )
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=difficulty_column_name,
            y=alt.Y("mean(correct)", title="P(correct)"),
            color="model",
        )
    )
    error_bars = (
        alt.Chart(df)
        .mark_errorband(extent="ci")
        .encode(
            x=difficulty_column_name,
            y=alt.Y("correct", title="P(correct)"),
            color="model",
        )
    )
    st.altair_chart(chart + error_bars, use_container_width=True)


def get_df_for_binned_acc_plot(
    instance_accuracy_per_model: dict[str, list[dict[str, int]]],
    instance_difficulties: dict[str, float],
    difficulty_column_name: str,
):
    df_dict = {
        difficulty_column_name: [],
        "model": [],
        "correct": [],
    }
    for model_name in st.session_state["models"]:
        correct = {
            result["id"]: result["is_correct"]
            for result in instance_accuracy_per_model[model_name]
        }
        ys = list(correct.values())
        if st.session_state["x_axis"] == "Raw difficulty":
            xs = quantize_difficulties(
                instance_difficulties, num_bins=st.session_state["num_bins"]
            )
        elif st.session_state["x_axis"] == "Difficulty quantile":
            xs = convert_difficulties_to_quantiles(
                instance_difficulties, num_bins=st.session_state["num_bins"]
            )
        df_dict[difficulty_column_name] += xs
        df_dict["model"] += [model_name] * len(xs)
        df_dict["correct"] += ys
    return pd.DataFrame(df_dict)


render_dashboard()
