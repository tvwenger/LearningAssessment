"""
dashboard.py
Build model summary dashboards:
- Model summary
- Class summary
- Student summaries

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import argparse
import os
import arviz as az
import numpy as np
import cloudpickle as pickle

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


def model_summary(
    model, trend_plot_width=800, trend_plot_height=400, skip_pairplot=False
):
    if not os.path.isdir(model.outdir):
        os.mkdir(model.outdir)
    if not os.path.isdir(os.path.join(model.outdir, "model")):
        os.mkdir(os.path.join(model.outdir, "model"))

    # Generate summary of models
    for objective in model.objectives:
        print(f"Summary for objective: {objective}")
        summary = az.summary(
            model.models[objective].trace,
            var_names=[
                var_name
                for var_name in model.var_names
                if var_name in model.models[objective].trace.posterior
            ],
        )
        print(summary)
        if not skip_pairplot:
            axes = az.plot_pair(
                model.models[objective].trace,
                var_names=[
                    var_name
                    for var_name in model.var_names
                    if var_name in model.models[objective].trace.posterior
                ],
                kind="kde",
                marginals=True,
                backend_kwargs={"figsize": (6, 6), "layout": "constrained"},
            )
            fig = axes.ravel()[0].figure
            fig.suptitle(f"{objective}, Max(r_hat) = {summary["r_hat"].max()}")
            fig.savefig(os.path.join(model.outdir, "model", f"{objective}_pair.png"))
            plt.close(fig)

    # Plot eta posterior distributions
    fig = go.Figure()
    count, index = np.histogram(
        model.models[model.objectives[0]].prior.prior["eta"].data.flatten(),
        bins=25,
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=count / count.sum(),
            line=dict(color="black", width=1, shape="hvh"),
            name="Prior",
        )
    )
    for objective in model.objectives:
        count, index = np.histogram(
            model.models[objective].trace.posterior["eta"].data.flatten(),
            bins=25,
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=count / count.sum(),
                line=dict(width=1, shape="hvh"),
                name=f"{objective}",
            )
        )
    fig.update_layout(
        title=dict(text=r"GP Scatter Posterior Distributions"),
        xaxis=dict(title=dict(text=r"GP Scatter")),
        yaxis=dict(title=dict(text="Probability Density")),
        autosize=False,
        width=trend_plot_width,
        height=trend_plot_height,
    )
    fig.write_html(
        os.path.join(model.outdir, "model", "eta_posteriors.html"),
        include_plotlyjs=False,
        full_html=False,
    )

    # Plot ell posterior distributions
    fig = go.Figure()
    count, index = np.histogram(
        model.models[model.objectives[0]].prior.prior["ell"].data.flatten(),
        bins=25,
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=count / count.sum(),
            line=dict(color="black", width=1, shape="hvh"),
            name="Prior",
        )
    )
    for objective in model.objectives:
        count, index = np.histogram(
            model.models[objective].trace.posterior["ell"].data.flatten(),
            bins=25,
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=count / count.sum(),
                line=dict(width=1, shape="hvh"),
                name=f"{objective}",
            )
        )
    fig.update_layout(
        title=dict(text=r"GP Lengthscale Posterior Distributions"),
        xaxis=dict(title=dict(text=r"GP Lengthscale")),
        yaxis=dict(title=dict(text="Probability Density")),
        autosize=False,
        width=trend_plot_width,
        height=trend_plot_height,
    )
    fig.write_html(
        os.path.join(model.outdir, "model", "ell_posteriors.html"),
        include_plotlyjs=False,
        full_html=False,
    )


def student_summary(
    model,
    trend_plot_width=800,
    trend_plot_height=400,
    prob_plot_width=400,
    prob_plot_height=400,
    percentiles=[5.0, 95.0],
    score_labels=["No Evidence", "Below Mastery", "Mastery"],
):
    if not os.path.isdir(model.outdir):
        os.mkdir(model.outdir)
    if not os.path.isdir(os.path.join(model.outdir, "students")):
        os.mkdir(os.path.join(model.outdir, "students"))

    model.mastery = {}
    model.mastery_timeline = {}
    for i, student in enumerate(model.students):
        if not os.path.isdir(os.path.join(model.outdir, "students", f"{student}")):
            os.mkdir(os.path.join(model.outdir, "students", f"{student}"))
        model.mastery[student] = {}
        model.mastery_timeline[student] = {}

        for objective in model.objectives:
            last_idx = model.last_assignment[objective]
            last_score_pred = (
                1
                + model.models[objective]
                .posterior_predictive.sel(student=student)
                .score.data
            )
            last_mastery = np.zeros_like(last_score_pred[last_idx])
            last_mastery[last_score_pred[last_idx] >= model.mastery_score] = 1
            last_p_mastery = last_mastery.sum() / last_mastery.shape[0]

            model.mastery[student][objective] = last_p_mastery > model.mastery_threshold
            score_pred = (
                1
                + model.models[objective]
                .posterior_predictive.sel(student=student)
                .score_pred.data
            )
            mastery = np.zeros_like(score_pred)
            mastery[score_pred >= model.mastery_score] = 1
            p_mastery = mastery.sum(axis=1) / mastery.shape[1]
            model.mastery_timeline[student][objective] = (
                p_mastery > model.mastery_threshold
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=model.models[objective].posterior_predictive.norm_date
                    * model.date_range
                    + model.date_offset,
                    y=np.percentile(
                        0.25
                        + (model.max_score - 1.0)  # offset for missing assignments
                        * model.models[objective]
                        .posterior_predictive.sel(student=student)
                        .p_pred,
                        [percentiles[0]],
                        axis=1,
                    )[0],
                    mode="lines",
                    line=dict(color="lightblue", width=1),
                    fill=None,
                    name=f"{int(percentiles[0])}th Percentile",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=model.models[objective].posterior_predictive.norm_date
                    * model.date_range
                    + model.date_offset,
                    y=np.percentile(
                        0.25
                        + (model.max_score - 1.0)  # offset for missing assignments
                        * model.models[objective]
                        .posterior_predictive.sel(student=student)
                        .p_pred,
                        [percentiles[1]],
                        axis=1,
                    )[0],
                    mode="lines",
                    line=dict(color="lightblue", width=1),
                    fill="tonexty",
                    fillcolor="rgba(173, 216, 230, 0.25)",
                    name=f"{int(percentiles[1])}th Percentile",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=model.models[objective].posterior_predictive.norm_date
                    * model.date_range
                    + model.date_offset,
                    y=np.percentile(
                        0.25
                        + (model.max_score - 1.0)  # offset for missing assignments
                        * model.models[objective]
                        .posterior_predictive.sel(student=student)
                        .p_pred,
                        [50.0],
                        axis=1,
                    )[0],
                    mode="lines",
                    line=dict(color="darkblue", width=3),
                    name="Median",
                )
            )
            samples = np.random.choice(
                model.models[objective].posterior_predictive.sample, 30, replace=False
            )
            for sample in samples:
                fig.add_trace(
                    go.Scatter(
                        x=model.models[objective].posterior_predictive.norm_date
                        * model.date_range
                        + model.date_offset,
                        y=(
                            0.25
                            + (model.max_score - 1.0)  # offset for missing assignments
                            * model.models[objective]
                            .posterior_predictive.sel(student=student, sample=sample)
                            .p_pred
                        ),
                        mode="lines",
                        line=dict(color="darkblue", width=1),
                        opacity=0.1,
                        showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=model.models[objective].posterior_predictive.norm_date
                    * model.date_range
                    + model.date_offset,
                    y=p_mastery,
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="Probability of Mastery",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.array([0.0, 1.0]) * model.date_range + model.date_offset,
                    y=np.array([model.mastery_threshold, model.mastery_threshold]),
                    mode="lines",
                    line=dict(color="red", width=3, dash="dash"),
                    name="Mastery Threshold",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.array(
                        [
                            model.norm_dates[objective].values[
                                model.last_assignment[objective]
                            ],
                            model.norm_dates[objective].values[
                                model.last_assignment[objective]
                            ],
                        ]
                    )
                    * model.date_range
                    + model.date_offset,
                    y=np.array([0.0, 1.0]),
                    mode="lines",
                    line=dict(color="orange", width=3, dash="dot"),
                    name="Last Assessment (Prediction Point)",
                ),
                secondary_y=True,
            )

            # offset like dates by a small amount
            ismissing = model.scores[objective][i] == 0.0
            offset = np.linspace(
                -0.01, 0.01, len(model.norm_dates[objective][~ismissing])
            )
            fig.add_trace(
                go.Scatter(
                    x=(model.norm_dates[objective][~ismissing] + offset)
                    * model.date_range
                    + model.date_offset,
                    y=-0.75 + model.scores[objective][i][~ismissing],
                    mode="markers",
                    name="Scores",
                    marker=dict(color="black", size=10, symbol="x"),
                    customdata=model.assignments[objective][~ismissing],
                    hovertemplate="<br>%{customdata}",
                ),
                secondary_y=False,
            )
            offset = np.linspace(
                -0.01, 0.01, len(model.norm_dates[objective][ismissing])
            )
            fig.add_trace(
                go.Scatter(
                    x=(model.norm_dates[objective][ismissing] + offset)
                    * model.date_range
                    + model.date_offset,
                    y=np.zeros_like(model.norm_dates[objective][ismissing]),
                    mode="markers",
                    name="Missing/No Assessment",
                    marker=dict(color="black", size=10, symbol="circle"),
                    customdata=model.assignments[objective][ismissing],
                    hovertemplate="<br>%{customdata}",
                ),
                secondary_y=False,
            )
            fig.update_layout(
                title=dict(text=f"Student: {student}, Objective: {objective}"),
                xaxis=dict(title=dict(text="Date")),
                yaxis=dict(
                    title=dict(text="Score"),
                    range=[-0.1, model.max_score - 0.65],
                    tickmode="array",
                    tickvals=[0.0] + list(np.arange(model.max_score) + 0.25),
                    ticktext=score_labels,
                ),
                autosize=False,
                width=trend_plot_width,
                height=trend_plot_height,
            )
            fig.update_yaxes(
                title=dict(text="Probability of Mastery"),
                range=[-0.1, 1.1],
                color="red",
                secondary_y=True,
                showgrid=False,
            )
            fig.write_html(
                os.path.join(
                    model.outdir, "students", f"{student}", f"{objective}_trend.html"
                ),
                include_plotlyjs=False,
                full_html=False,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=last_mastery,
                    histnorm="probability density",
                    name="Predictive Probability",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0.5, 1.5],
                    y=[model.mastery_threshold, model.mastery_threshold],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Mastery Threshold",
                )
            )
            fig.update_layout(
                title=dict(
                    text=(
                        f"Student: {student}, Objective: {objective}<br>"
                        + f"Predictive Probability on {model.last_assignment_date[objective]}"
                    )
                ),
                xaxis=dict(
                    title=dict(text="Score"),
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["Below Mastery", "At or Above Mastery"],
                ),
                yaxis=dict(title=dict(text="Probability")),
                autosize=False,
                width=prob_plot_width,
                height=prob_plot_height,
                bargap=0.2,
            )
            fig.write_html(
                os.path.join(
                    model.outdir, "students", f"{student}", f"{objective}_prob.html"
                ),
                include_plotlyjs=False,
                full_html=False,
            )

        # Histogram of mastered objectives
        fig = go.Figure()
        for objective in model.objectives:
            last_idx = model.last_assignment[objective]
            if model.mastery[student][objective]:
                fig.add_trace(
                    go.Histogram(
                        x=[1.0],
                        name=f"{objective}",
                    )
                )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=[0.0],
                        name=f"{objective}",
                    )
                )
        fig.update_layout(
            title=dict(
                text=(
                    f"Student: {student}, Objective Mastery<br>"
                    + f"Predictive Probability on {datetime.now().date()}"
                )
            ),
            xaxis=dict(
                title=dict(text="Score"),
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Below Mastery", "At or Above Mastery"],
            ),
            yaxis=dict(title=dict(text="Number of Objectives")),
            autosize=False,
            width=prob_plot_width,
            height=prob_plot_height,
            barmode="stack",
            bargap=0.2,
        )
        fig.write_html(
            os.path.join(model.outdir, "students", f"{student}", "mastery.html"),
            include_plotlyjs=False,
            full_html=False,
        )


def class_summary(
    model,
    trend_plot_width=800,
    trend_plot_height=400,
    score_labels=["No Evidence", "Below Mastery", "Mastery"],
):
    if not os.path.isdir(model.outdir):
        os.mkdir(model.outdir)
    if not os.path.isdir(os.path.join(model.outdir, "class")):
        os.mkdir(os.path.join(model.outdir, "class"))

    # Histogram of mastered objectives
    students_mastery = np.zeros(len(model.students))
    for i, student in enumerate(model.students):
        for objective in model.objectives:
            if model.mastery[student][objective] > model.mastery_threshold:
                students_mastery[i] += 1
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=students_mastery,
            name="Class Mastery",
            xbins=dict(start=-0.5, end=len(model.objectives) + 0.5, size=1),
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "Class Mastery Distribution<br>"
                + f"Predictive Probability on {datetime.now().date()}"
            )
        ),
        xaxis=dict(
            title=dict(text="Number of Objectives"),
            range=[0.5, len(model.objectives) + 0.5],
        ),
        yaxis=dict(title=dict(text="Number of Students")),
        autosize=False,
        width=trend_plot_width,
        height=trend_plot_height,
    )
    fig.write_html(
        os.path.join(model.outdir, "class", "class_mastery.html"),
        include_plotlyjs=False,
        full_html=False,
    )

    for objective in model.objectives:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        num_mastery = np.zeros_like(model.norm_date_axis)
        for i, student in enumerate(model.students):
            fig.add_trace(
                go.Scatter(
                    x=model.models[objective].posterior_predictive.norm_date
                    * model.date_range
                    + model.date_offset,
                    y=np.percentile(
                        1.0
                        + (model.max_score - 1.0)  # offset for missing assignments
                        * model.models[objective]
                        .posterior_predictive.sel(student=student)
                        .p_pred,
                        [50.0],
                        axis=1,
                    )[0],
                    mode="lines",
                    name=f"{student}",
                )
            )
            num_mastery += model.mastery_timeline[student][objective]
        fig.add_trace(
            go.Scatter(
                x=model.models[objective].posterior_predictive.norm_date
                * model.date_range
                + model.date_offset,
                y=np.percentile(
                    1.0
                    + (model.max_score - 1.0)  # offset for missing assignments
                    * model.models[objective].posterior_predictive.p_pred,
                    [50.0],
                    axis=(0, 2),
                )[0],
                mode="lines",
                line=dict(color="black", width=5),
                name="Class Median",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=model.models[objective].posterior_predictive.norm_date
                * model.date_range
                + model.date_offset,
                y=num_mastery / len(model.students),
                mode="lines",
                line=dict(color="red", width=3),
                name="Fraction Mastered",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(
                    [
                        model.norm_dates[objective].values[
                            model.last_assignment[objective]
                        ],
                        model.norm_dates[objective].values[
                            model.last_assignment[objective]
                        ],
                    ]
                )
                * model.date_range
                + model.date_offset,
                y=np.array([0.0, 1.0]),
                mode="lines",
                line=dict(color="orange", width=3, dash="dot"),
                name="Last Assessment (Prediction Point)",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title=dict(text="Class Mastery Fraction"),
            range=[-0.1, 1.1],
            color="red",
            secondary_y=True,
            showgrid=False,
        )
        fig.update_layout(
            title=dict(text=f"Objective: {objective}"),
            xaxis=dict(title=dict(text="Date")),
            yaxis=dict(
                title=dict(text="Score"),
                range=[0.9, model.max_score + 0.1],
                tickmode="array",
                tickvals=np.arange(model.max_score) + 1.0,
                ticktext=score_labels[1:],
            ),
            autosize=False,
            width=trend_plot_width,
            height=trend_plot_height,
        )
        fig.write_html(
            os.path.join(model.outdir, "class", f"{objective}.html"),
            include_plotlyjs=False,
            full_html=False,
        )


def build_dashboard(
    model,
    classname,
    trend_plot_width=800,
    prob_plot_width=400,
):
    with open(os.path.join(model.outdir, "index.html"), "w") as f:
        f.write("<html><head><title>Learning Assessment Dashboard</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>Learning Assessment Dashboard</h1>\n")
        f.write(f"<h2>Class: {classname}</h2>\n")
        f.write(f"<h3>Updated: {datetime.now().date()}</h3>\n")
        f.write("<a href='model/index.html'>Model Summary</a><br>\n")
        f.write("<a href='class/index.html'>Class Summary</a><br>\n")
        f.write("<h3>Student Summaries</h3>\n")
        f.write("<ul>\n")
        for student in model.students:
            f.write(
                f'<li><a href="students/{student}/index.html">Student: {student}</a></li>\n'
            )
        f.write("</ui>\n")
        f.write("</body></html>\n")

    with open(os.path.join(model.outdir, "model", "index.html"), "w") as f:
        f.write("<html><head><title>Model Summary</title></head>\n")
        f.write("<body>\n")
        f.write(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>\n'
        )
        f.write("<a href='../index.html'>Back to Dashboard</a><br>\n")
        f.write("<h1>Model Summary</h1>\n")
        with open(
            os.path.join(model.outdir, "model", "ell_posteriors.html"), "r"
        ) as f_in:
            div = f_in.read()
            f.write(
                div.replace(
                    "<div>",
                    f'<div style="display:inline-block; width:{trend_plot_width+50}px;">',
                )
                + "\n"
            )
        with open(
            os.path.join(model.outdir, "model", "eta_posteriors.html"), "r"
        ) as f_in:
            div = f_in.read()
            f.write(
                div.replace(
                    "<div>",
                    f'<div style="display:inline-block; width:{trend_plot_width+50}px;">',
                )
                + "\n"
            )
        for objective in model.objectives:
            f.write(f'<img src="{objective}_pair.png" />')
        f.write("</body></html>\n")

    with open(os.path.join(model.outdir, "class", "index.html"), "w") as f:
        f.write("<html><head>\n")
        f.write("<title>Class Summary</title>\n")
        f.write(
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js'></script>\n"
        )
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("<a href='../index.html'>Back to Dashboard</a><br>\n")
        f.write("<h1>Class Summary</h1>\n")
        with open(
            os.path.join(model.outdir, "class", "class_mastery.html"), "r"
        ) as f_in:
            div = f_in.read()
            f.write(
                div.replace(
                    "<div>",
                    f'<div style="display:inline-block; width:{trend_plot_width+50}px;">',
                )
                + "<br/>\n"
            )
        for objective in model.objectives:
            with open(
                os.path.join(model.outdir, "class", f"{objective}.html"), "r"
            ) as f_in:
                div = f_in.read()
                f.write(
                    div.replace(
                        "<div>",
                        f'<div style="display:inline-block; width:{trend_plot_width+50}px;">',
                    )
                    + "\n"
                )
        f.write("</body></html>\n")

    for student in model.students:
        with open(
            os.path.join(model.outdir, "students", f"{student}", "index.html"), "w"
        ) as f:
            f.write(f"<html><head><title>Student: {student}</title></head>\n")
            f.write("<body>\n")
            f.write(
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>\n'
            )
            f.write("<a href='../../index.html'>Back to Dashboard</a><br>\n")
            f.write(f"<h1>Student: {student}</h1>\n")
            with open(
                os.path.join(model.outdir, "students", f"{student}", "mastery.html"),
                "r",
            ) as f_in:
                div = f_in.read()
                f.write(
                    div.replace(
                        "<div>",
                        f'<div style="display:inline-block; width:{prob_plot_width+50}px;">',
                    )
                    + "<br/>\n"
                )
            for objective in model.objectives:
                with open(
                    os.path.join(
                        model.outdir,
                        "students",
                        f"{student}",
                        f"{objective}_trend.html",
                    ),
                    "r",
                ) as f_in:
                    div = f_in.read()
                    f.write(
                        div.replace(
                            "<div>",
                            f'<div style="display:inline-block; width:{trend_plot_width+50}px;">',
                        )
                        + "\n"
                    )
                with open(
                    os.path.join(
                        model.outdir,
                        "students",
                        f"{student}",
                        f"{objective}_prob.html",
                    ),
                    "r",
                ) as f_in:
                    div = f_in.read()
                    f.write(
                        div.replace(
                            "<div>",
                            f'<div style="display:inline-block; width:{prob_plot_width+50}px;">',
                        )
                        + "<br/>\n"
                    )
            f.write("</body></html>\n")


def main():
    PARSER = argparse.ArgumentParser(
        description="Build assessment model dashboards",
        prog="dashboard.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("input_file", type=str, help="Model pickle file")
    PARSER.add_argument(
        "--score_labels",
        type=str,
        nargs="+",
        default=["No Evidence", "Below Mastery", "Mastery"],
    )
    PARSER.add_argument(
        "--trend_plot_width", type=int, default=800, help="Trend plot width (pix)"
    )
    PARSER.add_argument(
        "--trend_plot_height", type=int, default=400, help="Trend plot height (pix)"
    )
    PARSER.add_argument(
        "--prob_plot_width", type=int, default=400, help="Probability plot width (pix)"
    )
    PARSER.add_argument(
        "--prob_plot_height",
        type=int,
        default=400,
        help="Probability plot height (pix)",
    )
    PARSER.add_argument(
        "--skip_pairplot",
        action="store_true",
        default=False,
        help="Skip pair-plot generation",
    )
    PARSER.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=[5.0, 95.0],
        help="Percentiles for trend plots",
    )
    ARGS = vars(PARSER.parse_args())

    with open(ARGS["input_file"], "rb") as f:
        models = pickle.load(f)

    print(f"Building dashboards for {ARGS['input_file']}")
    model_summary(
        models,
        trend_plot_width=ARGS["trend_plot_width"],
        trend_plot_height=ARGS["trend_plot_height"],
        skip_pairplot=ARGS["skip_pairplot"],
    )
    student_summary(
        models,
        trend_plot_width=ARGS["trend_plot_width"],
        trend_plot_height=ARGS["trend_plot_height"],
        prob_plot_width=ARGS["prob_plot_width"],
        prob_plot_height=ARGS["prob_plot_height"],
        percentiles=ARGS["percentiles"],
        score_labels=ARGS["score_labels"],
    )
    class_summary(
        models,
        trend_plot_width=ARGS["trend_plot_width"],
        trend_plot_height=ARGS["trend_plot_height"],
        score_labels=ARGS["score_labels"],
    )
    build_dashboard(
        models,
        models.outdir,
        trend_plot_width=ARGS["trend_plot_width"],
        prob_plot_width=ARGS["prob_plot_width"],
    )
