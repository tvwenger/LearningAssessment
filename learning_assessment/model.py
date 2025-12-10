"""
model.py
Model definition.

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import argparse
import os
import arviz as az
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from canvasapi import Canvas

from learning_assessment.loader import load


class ObjectiveModels:
    def __init__(
        self,
        datapath,
        outdir,
        max_score=2,
        mastery_score=2.0,
        mastery_threshold=0.6,
        score_labels=["No Evidence", "Below Mastery", "Mastery"],
        date_samples=50,
        trend_plot_width=800,
        trend_plot_height=400,
        prob_plot_width=400,
        prob_plot_height=400,
    ):
        # Load data
        data = load(datapath)
        self.date_offset = data["Date"].min()
        self.date_range = data["Date"].max() - data["Date"].min()
        data["Date Norm"] = (data["Date"] - self.date_offset) / self.date_range
        self.data = data

        # Unique students and objectives
        self.students = sorted(data["id"].unique())
        self.objectives = sorted(
            [
                col
                for col in data.columns
                if col not in ["id", "Date", "Assignment", "Date Norm"]
            ]
        )

        self.norm_date_axis = np.linspace(0.0, 1.0, date_samples)
        self.max_score = max_score
        self.outdir = outdir
        self.mastery_score = mastery_score
        self.mastery_threshold = mastery_threshold
        self.score_labels = score_labels
        self.trend_plot_width = trend_plot_width
        self.trend_plot_height = trend_plot_height
        self.prob_plot_width = prob_plot_width
        self.prob_plot_height = prob_plot_height
        self.var_names = ["eta", "ell"]

        self.assignments = {}
        self.scores = {}
        self.norm_dates = {}
        self.last_assignment = {}
        self.last_assignment_date = {}
        for objective in self.objectives:
            # Build data array (shape: students, assignments)
            assignments = None
            scores = None
            norm_dates = None
            for i, student in enumerate(self.students):
                sub_data = self.data.loc[self.data["id"] == student]
                score = sub_data[objective]
                # Drop NaNs (not assessed)
                nan = score.isna()
                if assignments is None:
                    assignments = sub_data["Assignment"][~nan]
                    norm_dates = sub_data["Date Norm"][~nan]
                    scores = np.ones((len(self.students), len(assignments))) * np.nan
                    scores[i] = score[~nan]
                else:
                    # Checks
                    assert len(assignments) == len(sub_data["Assignment"][~nan])
                    assert np.all(norm_dates.values == sub_data["Date Norm"][~nan])
                    scores[i] = score[~nan]
            self.scores[objective] = scores.copy()
            self.assignments[objective] = assignments
            self.norm_dates[objective] = norm_dates

            # norm_date of last assignment
            self.last_assignment[objective] = np.nanargmax(norm_dates.values)
            self.last_assignment_date[objective] = sub_data["Date"][~nan].values[
                self.last_assignment[objective]
            ]

        # Storage for results
        self.gps = {}
        self.models = {}
        self.mastery = {}

    def build(
        self,
        intercept=0.0,
        prior_ell=[2.0, 2.0],
        prior_eta=1.0,
        cov_func="matern52",
        missing="replace",
    ):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "model")):
            os.mkdir(os.path.join(self.outdir, "model"))

        # Loop over learning objectives
        for objective in self.objectives:
            print(f"Building model for objective: {objective}")

            # Drop or replace zeros for sampling (missing assessments)
            scores = self.scores[objective].copy()
            if missing == "drop":
                scores[scores == 0.0] = np.nan
            elif missing == "replace":
                scores[scores == 0.0] = 1.0
            else:
                raise ValueError(f"Invalid missing value handling: {missing}")

            with pm.Model(
                coords={
                    "student": self.students,
                    "assignment": self.assignments[objective],
                    "norm_date": self.norm_date_axis,
                }
            ) as self.models[objective]:
                # gp parameters
                # ell = pm.Beta("ell", alpha=prior_ell[0], beta=prior_ell[1])
                ell = pm.LogitNormal("ell", mu=prior_ell[0], sigma=prior_ell[1])
                eta = pm.HalfNormal("eta", sigma=prior_eta)
                # eta = pm.Gamma("eta", alpha=2.0, beta=1.0)

                if cov_func == "matern52":
                    cov = eta**2.0 * pm.gp.cov.Matern52(1, ell)
                elif cov_func == "matern32":
                    cov = eta**2.0 * pm.gp.cov.Matern32(1, ell)
                elif cov_func == "expquad":
                    cov = eta**2.0 * pm.gp.cov.ExpQuad(1, ell)
                else:
                    raise ValueError(f"Unsupported cov_func: {cov_func}")

                mean_func = pm.gp.mean.Constant(intercept)

                self.gps[objective] = pm.gp.Latent(mean_func=mean_func, cov_func=cov)
                f = self.gps[objective].prior(
                    "f",
                    X=self.norm_dates[objective].values[:, None],
                    dims=["student", "assignment"],
                )

                # likelihood
                p = pm.Deterministic(
                    "p", pm.math.invlogit(f), dims=["student", "assignment"]
                )
                _ = pm.Binomial(
                    "score",
                    n=int(self.max_score) - 1,  # offset for missing assignments
                    p=p,
                    observed=scores - 1,  # offset for missing assignments
                    dims=["student", "assignment"],
                )

            print(f"Sampling prior for objective: {objective}")
            # Sample prior
            with self.models[objective]:
                self.models[objective].prior = pm.sample_prior_predictive()

    def sample(self, thin_predictive=10):
        # Loop over learning objectives
        for objective in self.objectives:
            print(f"Sampling posterior for objective: {objective}")
            with self.models[objective]:
                self.models[objective].trace = pm.sample()

            print(f"Sampling posterior predictive for objective: {objective}")
            with self.models[objective]:
                f_pred = self.gps[objective].conditional(
                    "f_pred",
                    self.norm_date_axis[:, None],
                    dims=["student", "norm_date"],
                )
                p_pred = pm.Deterministic(
                    "p_pred",
                    pm.math.invlogit(f_pred),
                    dims=["student", "norm_date"],
                )
                _ = pm.Binomial(
                    "score_pred",
                    n=int(self.max_score) - 1,  # offset for missing assignments
                    p=p_pred,
                    dims=["student", "norm_date"],
                )
                self.models[objective].posterior_predictive = az.extract(
                    pm.sample_posterior_predictive(
                        self.models[objective].trace.sel(
                            draw=slice(None, None, thin_predictive)
                        ),
                        var_names=["score", "p_pred", "score_pred"],
                    ).posterior_predictive
                )

    def model_summary(self, skip_pairplot=False):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "model")):
            os.mkdir(os.path.join(self.outdir, "model"))

        # Generate summary of models
        for objective in self.objectives:
            print(f"Summary for objective: {objective}")
            summary = az.summary(
                self.models[objective].trace,
                var_names=[
                    var_name
                    for var_name in self.var_names
                    if var_name in self.models[objective].trace.posterior
                ],
            )
            print(summary)
            if not skip_pairplot:
                axes = az.plot_pair(
                    self.models[objective].trace,
                    var_names=[
                        var_name
                        for var_name in self.var_names
                        if var_name in self.models[objective].trace.posterior
                    ],
                    kind="kde",
                    marginals=True,
                    backend_kwargs={"figsize": (6, 6), "layout": "constrained"},
                )
                fig = axes.ravel()[0].figure
                fig.suptitle(f"{objective}, Max(r_hat) = {summary["r_hat"].max()}")
                fig.savefig(os.path.join(self.outdir, "model", f"{objective}_pair.png"))
                plt.close(fig)

        # Plot eta posterior distributions
        fig = go.Figure()
        count, index = np.histogram(
            self.models[self.objectives[0]].prior.prior["eta"].data.flatten(),
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
        for objective in self.objectives:
            count, index = np.histogram(
                self.models[objective].trace.posterior["eta"].data.flatten(),
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
            width=self.trend_plot_width,
            height=self.trend_plot_height,
        )
        fig.write_html(
            os.path.join(self.outdir, "model", "eta_posteriors.html"),
            include_plotlyjs=False,
            full_html=False,
        )

        # Plot ell posterior distributions
        fig = go.Figure()
        count, index = np.histogram(
            self.models[self.objectives[0]].prior.prior["ell"].data.flatten(),
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
        for objective in self.objectives:
            count, index = np.histogram(
                self.models[objective].trace.posterior["ell"].data.flatten(),
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
            width=self.trend_plot_width,
            height=self.trend_plot_height,
        )
        fig.write_html(
            os.path.join(self.outdir, "model", "ell_posteriors.html"),
            include_plotlyjs=False,
            full_html=False,
        )

    def student_summary(self, percentiles=[5.0, 95.0]):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "students")):
            os.mkdir(os.path.join(self.outdir, "students"))

        for i, student in enumerate(self.students):
            if not os.path.isdir(os.path.join(self.outdir, "students", f"{student}")):
                os.mkdir(os.path.join(self.outdir, "students", f"{student}"))
            self.mastery[student] = {}

            for objective in self.objectives:
                last_idx = self.last_assignment[objective]
                last_score_pred = (
                    1
                    + self.models[objective]
                    .posterior_predictive.sel(student=student)
                    .score.data
                )
                last_mastery = np.zeros_like(last_score_pred[last_idx])
                last_mastery[last_score_pred[last_idx] >= self.mastery_score] = 1
                last_p_mastery = last_mastery.sum() / last_mastery.shape[0]

                self.mastery[student][objective] = (
                    last_p_mastery > self.mastery_threshold
                )
                score_pred = (
                    1
                    + self.models[objective]
                    .posterior_predictive.sel(student=student)
                    .score_pred.data
                )
                mastery = np.zeros_like(score_pred)
                mastery[score_pred >= self.mastery_score] = 1
                p_mastery = mastery.sum(axis=1) / mastery.shape[1]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=self.models[objective].posterior_predictive.norm_date
                        * self.date_range
                        + self.date_offset,
                        y=np.percentile(
                            0.25
                            + (self.max_score - 1.0)  # offset for missing assignments
                            * self.models[objective]
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
                        x=self.models[objective].posterior_predictive.norm_date
                        * self.date_range
                        + self.date_offset,
                        y=np.percentile(
                            0.25
                            + (self.max_score - 1.0)  # offset for missing assignments
                            * self.models[objective]
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
                        x=self.models[objective].posterior_predictive.norm_date
                        * self.date_range
                        + self.date_offset,
                        y=np.percentile(
                            0.25
                            + (self.max_score - 1.0)  # offset for missing assignments
                            * self.models[objective]
                            .posterior_predictive.sel(student=student)
                            .p_pred,
                            [50.0],
                            axis=1,
                        )[0],
                        mode="lines",
                        line=dict(color="darkblue", width=3),
                        opacity=1.0,
                        name="Median",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.models[objective].posterior_predictive.norm_date
                        * self.date_range
                        + self.date_offset,
                        y=p_mastery,
                        mode="lines",
                        line=dict(color="red", width=3),
                        opacity=1.0,
                        name="Probability of Mastery",
                    ),
                    secondary_y=True,
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.array([0.0, 1.0]) * self.date_range + self.date_offset,
                        y=np.array([self.mastery_threshold, self.mastery_threshold]),
                        mode="lines",
                        line=dict(color="red", width=3, dash="dash"),
                        opacity=1.0,
                        name="Mastery Threshold",
                    ),
                    secondary_y=True,
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.array(
                            [
                                self.norm_dates[objective].values[
                                    self.last_assignment[objective]
                                ],
                                self.norm_dates[objective].values[
                                    self.last_assignment[objective]
                                ],
                            ]
                        )
                        * self.date_range
                        + self.date_offset,
                        y=np.array([0.0, 1.0]),
                        mode="lines",
                        line=dict(color="orange", width=3, dash="dot"),
                        opacity=1.0,
                        name="Last Assessment (Prediction Point)",
                    ),
                    secondary_y=True,
                )

                # offset like dates by a small amount
                ismissing = self.scores[objective][i] == 0.0
                offset = np.linspace(
                    -0.01, 0.01, len(self.norm_dates[objective][~ismissing])
                )
                fig.add_trace(
                    go.Scatter(
                        x=(self.norm_dates[objective][~ismissing] + offset)
                        * self.date_range
                        + self.date_offset,
                        y=-0.75 + self.scores[objective][i][~ismissing],
                        mode="markers",
                        name="Scores",
                        marker=dict(color="black", size=10, symbol="x"),
                        customdata=self.assignments[objective][~ismissing],
                        hovertemplate="<br>%{customdata}",
                    ),
                    secondary_y=False,
                )
                offset = np.linspace(
                    -0.01, 0.01, len(self.norm_dates[objective][ismissing])
                )
                fig.add_trace(
                    go.Scatter(
                        x=(self.norm_dates[objective][ismissing] + offset)
                        * self.date_range
                        + self.date_offset,
                        y=np.zeros_like(self.norm_dates[objective][ismissing]),
                        mode="markers",
                        name="Missing/No Assessment",
                        marker=dict(color="black", size=10, symbol="circle"),
                        customdata=self.assignments[objective][ismissing],
                        hovertemplate="<br>%{customdata}",
                    ),
                    secondary_y=False,
                )
                fig.update_layout(
                    title=dict(text=f"Student: {student}, Objective: {objective}"),
                    xaxis=dict(title=dict(text="Date")),
                    yaxis=dict(
                        title=dict(text="Score"),
                        range=[-0.1, self.max_score - 0.65],
                        tickmode="array",
                        tickvals=[0.0] + list(np.arange(self.max_score) + 0.25),
                        ticktext=self.score_labels,
                    ),
                    autosize=False,
                    width=self.trend_plot_width,
                    height=self.trend_plot_height,
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
                        self.outdir, "students", f"{student}", f"{objective}_trend.html"
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
                        y=[self.mastery_threshold, self.mastery_threshold],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Mastery Threshold",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=(
                            f"Student: {student}, Objective: {objective}<br>"
                            + f"Predictive Probability on {self.last_assignment_date[objective]}"
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
                    width=self.prob_plot_width,
                    height=self.prob_plot_height,
                    bargap=0.2,
                )
                fig.write_html(
                    os.path.join(
                        self.outdir, "students", f"{student}", f"{objective}_prob.html"
                    ),
                    include_plotlyjs=False,
                    full_html=False,
                )

            # Histogram of mastered objectives
            fig = go.Figure()
            for objective in self.objectives:
                last_idx = self.last_assignment[objective]
                if self.mastery[student][objective]:
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
                width=self.prob_plot_width,
                height=self.prob_plot_height,
                barmode="stack",
                bargap=0.2,
            )
            fig.write_html(
                os.path.join(self.outdir, "students", f"{student}", "mastery.html"),
                include_plotlyjs=False,
                full_html=False,
            )

    def class_summary(self):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "class")):
            os.mkdir(os.path.join(self.outdir, "class"))

        # Histogram of mastered objectives
        students_mastery = np.zeros(len(self.students))
        for i, student in enumerate(self.students):
            for objective in self.objectives:
                if self.mastery[student][objective] > self.mastery_threshold:
                    students_mastery[i] += 1
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=students_mastery,
                name="Class Mastery",
                xbins=dict(start=-0.5, end=len(self.objectives) + 0.5, size=1),
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
                range=[0.5, len(self.objectives) + 0.5],
            ),
            yaxis=dict(title=dict(text="Number of Students")),
            autosize=False,
            width=self.trend_plot_width,
            height=self.trend_plot_height,
        )
        fig.write_html(
            os.path.join(self.outdir, "class", "class_mastery.html"),
            include_plotlyjs=False,
            full_html=False,
        )

        for objective in self.objectives:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            num_mastery = np.zeros_like(self.norm_date_axis)
            for i, student in enumerate(self.students):
                fig.add_trace(
                    go.Scatter(
                        x=self.models[objective].posterior_predictive.norm_date
                        * self.date_range
                        + self.date_offset,
                        y=np.percentile(
                            1.0
                            + (self.max_score - 1.0)  # offset for missing assignments
                            * self.models[objective]
                            .posterior_predictive.sel(student=student)
                            .p_pred,
                            [50.0],
                            axis=1,
                        )[0],
                        mode="lines",
                        name=f"{student}",
                    )
                )
                num_mastery += self.mastery[student][objective]
            fig.add_trace(
                go.Scatter(
                    x=self.models[objective].posterior_predictive.norm_date
                    * self.date_range
                    + self.date_offset,
                    y=np.percentile(
                        1.0
                        + (self.max_score - 1.0)  # offset for missing assignments
                        * self.models[objective].posterior_predictive.p_pred,
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
                    x=self.models[objective].posterior_predictive.norm_date
                    * self.date_range
                    + self.date_offset,
                    y=num_mastery / len(self.students),
                    mode="lines",
                    line=dict(color="red", width=3),
                    opacity=1.0,
                    name="Fraction Mastered",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.array(
                        [
                            self.norm_dates[objective].values[
                                self.last_assignment[objective]
                            ],
                            self.norm_dates[objective].values[
                                self.last_assignment[objective]
                            ],
                        ]
                    )
                    * self.date_range
                    + self.date_offset,
                    y=np.array([0.0, 1.0]),
                    mode="lines",
                    line=dict(color="orange", width=3, dash="dot"),
                    opacity=1.0,
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
                    range=[0.9, self.max_score + 0.1],
                    tickmode="array",
                    tickvals=np.arange(self.max_score) + 1.0,
                    ticktext=self.score_labels[1:],
                ),
                autosize=False,
                width=self.trend_plot_width,
                height=self.trend_plot_height,
            )
            fig.write_html(
                os.path.join(self.outdir, "class", f"{objective}.html"),
                include_plotlyjs=False,
                full_html=False,
            )

    def build_dashboard(self, classname):
        with open(os.path.join(self.outdir, "index.html"), "w") as f:
            f.write("<html><head><title>Learning Assessment Dashboard</title></head>\n")
            f.write("<body>\n")
            f.write("<h1>Learning Assessment Dashboard</h1>\n")
            f.write(f"<h2>Class: {classname}</h2>\n")
            f.write(f"<h3>Updated: {datetime.now().date()}</h3>\n")
            f.write("<a href='model/index.html'>Model Summary</a><br>\n")
            f.write("<a href='class/index.html'>Class Summary</a><br>\n")
            f.write("<h3>Student Summaries</h3>\n")
            f.write("<ul>\n")
            for student in self.students:
                f.write(
                    f'<li><a href="students/{student}/index.html">Student: {student}</a></li>\n'
                )
            f.write("</ui>\n")
            f.write("</body></html>\n")

        with open(os.path.join(self.outdir, "model", "index.html"), "w") as f:
            f.write("<html><head><title>Model Summary</title></head>\n")
            f.write("<body>\n")
            f.write(
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>\n'
            )
            f.write("<a href='../index.html'>Back to Dashboard</a><br>\n")
            f.write("<h1>Model Summary</h1>\n")
            with open(
                os.path.join(self.outdir, "model", "ell_posteriors.html"), "r"
            ) as f_in:
                div = f_in.read()
                f.write(
                    div.replace(
                        "<div>",
                        f'<div style="display:inline-block; width:{self.trend_plot_width+50}px;">',
                    )
                    + "\n"
                )
            with open(
                os.path.join(self.outdir, "model", "eta_posteriors.html"), "r"
            ) as f_in:
                div = f_in.read()
                f.write(
                    div.replace(
                        "<div>",
                        f'<div style="display:inline-block; width:{self.trend_plot_width+50}px;">',
                    )
                    + "\n"
                )
            for objective in self.objectives:
                f.write(f'<img src="{objective}_pair.png" />')
            f.write("</body></html>\n")

        with open(os.path.join(self.outdir, "class", "index.html"), "w") as f:
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
                os.path.join(self.outdir, "class", "class_mastery.html"), "r"
            ) as f_in:
                div = f_in.read()
                f.write(
                    div.replace(
                        "<div>",
                        f'<div style="display:inline-block; width:{self.trend_plot_width+50}px;">',
                    )
                    + "<br/>\n"
                )
            for objective in self.objectives:
                with open(
                    os.path.join(self.outdir, "class", f"{objective}.html"), "r"
                ) as f_in:
                    div = f_in.read()
                    f.write(
                        div.replace(
                            "<div>",
                            f'<div style="display:inline-block; width:{self.trend_plot_width+50}px;">',
                        )
                        + "\n"
                    )
            f.write("</body></html>\n")

        for student in self.students:
            with open(
                os.path.join(self.outdir, "students", f"{student}", "index.html"), "w"
            ) as f:
                f.write(f"<html><head><title>Student: {student}</title></head>\n")
                f.write("<body>\n")
                f.write(
                    '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>\n'
                )
                f.write("<a href='../../index.html'>Back to Dashboard</a><br>\n")
                f.write(f"<h1>Student: {student}</h1>\n")
                with open(
                    os.path.join(self.outdir, "students", f"{student}", "mastery.html"),
                    "r",
                ) as f_in:
                    div = f_in.read()
                    f.write(
                        div.replace(
                            "<div>",
                            f'<div style="display:inline-block; width:{self.prob_plot_width+50}px;">',
                        )
                        + "<br/>\n"
                    )
                for objective in self.objectives:
                    with open(
                        os.path.join(
                            self.outdir,
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
                                f'<div style="display:inline-block; width:{self.trend_plot_width+50}px;">',
                            )
                            + "\n"
                        )
                    with open(
                        os.path.join(
                            self.outdir,
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
                                f'<div style="display:inline-block; width:{self.prob_plot_width+50}px;">',
                            )
                            + "<br/>\n"
                        )
                f.write("</body></html>\n")


def main():
    PARSER = argparse.ArgumentParser(
        description="Build and sample assessment model",
        prog="model.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("canvas_url", type=str, help="Canvas URL")
    PARSER.add_argument("course_id", type=int, help="Course ID")
    PARSER.add_argument("--max_score", type=int, default=2, help="Maximum score")
    PARSER.add_argument(
        "--mastery_score", type=float, default=2.0, help="Mastery score"
    )
    PARSER.add_argument(
        "--mastery_threshold",
        type=float,
        default=0.6,
        help="Mastery threshold probability",
    )
    PARSER.add_argument(
        "--score_labels",
        type=str,
        nargs="+",
        default=["No Evidence", "Below Mastery", "Mastery"],
    )
    PARSER.add_argument(
        "--date_samples",
        type=int,
        default=50,
        help="Number of date samples for visualizations",
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
        "--intercept", type=float, default=-2.0, help="Gaussian process mean"
    )
    PARSER.add_argument(
        "--prior_ell",
        type=float,
        nargs="+",
        default=[-0.5, 0.25],
        help="Gaussian process lengthscale prior",
    )
    PARSER.add_argument(
        "--prior_eta",
        type=float,
        default=1.0,
        help="Gaussian process scatter prior",
    )
    PARSER.add_argument(
        "--cov_func",
        type=str,
        default="matern32",
        help="Gaussian process covariance function",
    )
    PARSER.add_argument(
        "--missing",
        type=str,
        default="replace",
        help="Missing assessment strategy",
    )
    PARSER.add_argument(
        "--thin_predictive",
        type=int,
        default=1,
        help="Thin predictive samples",
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

    canvas = Canvas(ARGS["canvas_url"], os.environ["CANVAS_TOKEN"])
    course = canvas.get_course(ARGS["course_id"])
    outdir = course.name
    datapath = os.path.join(outdir, "assessments")

    models = ObjectiveModels(
        datapath,
        outdir,
        max_score=ARGS["max_score"],
        mastery_score=ARGS["mastery_score"],
        score_labels=ARGS["score_labels"],
        date_samples=ARGS["date_samples"],
        trend_plot_width=ARGS["trend_plot_width"],
        trend_plot_height=ARGS["trend_plot_height"],
        prob_plot_width=ARGS["prob_plot_width"],
        prob_plot_height=ARGS["prob_plot_height"],
    )
    models.build(
        intercept=ARGS["intercept"],
        prior_ell=ARGS["prior_ell"],
        prior_eta=ARGS["prior_eta"],
        cov_func=ARGS["cov_func"],
        missing=ARGS["missing"],
    )
    models.sample(thin_predictive=ARGS["thin_predictive"])
    models.model_summary(skip_pairplot=ARGS["skip_pairplot"])
    models.student_summary()
    models.class_summary()
    models.build_dashboard(course.name)
