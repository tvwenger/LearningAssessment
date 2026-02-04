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
import cloudpickle as pickle

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
        date_samples=50,
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


def main():
    PARSER = argparse.ArgumentParser(
        description="Build and sample assessment model",
        prog="model.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("canvas_url", type=str, help="Canvas URL")
    PARSER.add_argument("course_id", type=int, help="Course ID")
    PARSER.add_argument("output_file", type=str, help="Output model pickle file")
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
        "--date_samples",
        type=int,
        default=50,
        help="Number of date samples for visualizations",
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
        date_samples=ARGS["date_samples"],
    )
    models.build(
        intercept=ARGS["intercept"],
        prior_ell=ARGS["prior_ell"],
        prior_eta=ARGS["prior_eta"],
        cov_func=ARGS["cov_func"],
        missing=ARGS["missing"],
    )
    models.sample(thin_predictive=ARGS["thin_predictive"])
    with open(ARGS["output_file"], "wb") as f:
        pickle.dump(models, f)
