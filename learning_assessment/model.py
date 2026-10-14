"""
model.py
Model definition.

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import argparse
import logging
import os
import arviz as az
import pymc as pm
import numpy as np
import cloudpickle as pickle
import pandas as pd

from learning_assessment.loader import load

_LOGGER = logging.getLogger(__name__)


class ObjectiveModels:
    def __init__(
        self,
        datapath,
        outdir,
        max_score=2,
        mastery_score=2.0,
        mastery_threshold=0.6,
        date_samples=50,
        start_date=None,
        end_date=None,
    ):
        # Load data
        data = load(datapath)
        if start_date is None:
            start_date = data["Date"].min()
        else:
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = data["Date"].max()
        else:
            end_date = pd.to_datetime(end_date)
        self.date_offset = start_date
        self.date_range = end_date - start_date
        data["Date Norm"] = (data["Date"] - self.date_offset) / self.date_range
        self.data = data

        # Unique students and objectives
        self.students = sorted(data["id"].unique())
        self.student_names = {
            student: self.data.loc[self.data["id"] == student, "name"].values[0]
            for student in self.students
        }
        self.objectives = sorted(
            [
                col
                for col in data.columns
                if col not in ["id", "name", "Date", "Assignment", "Date Norm"]
            ]
        )

        # catch dropped students
        drop_students = []
        for objective in self.objectives:
            num_assignments = np.zeros(len(self.students), dtype=int)
            for i, student in enumerate(self.students):
                sub_data = self.data.loc[self.data["id"] == student]
                score = sub_data[objective]
                num_assignments[i] = len(sub_data["Assignment"][~score.isna()])
            for i, student in enumerate(self.students):
                if num_assignments[i] != np.median(num_assignments):
                    drop_students.append(student)
        for student in np.unique(drop_students):
            _LOGGER.warning(
                f"ALERT: {self.student_names[student]} is missing assessments and will be dropped."
            )
        self.students = [
            student for student in self.students if student not in drop_students
        ]

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
        self.median_norm_date_diff = {}
        for objective in self.objectives:
            # Build data array (shape: students, assignments)
            assignments = None
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

            # minimum norm_date difference between assessments
            # excluding zero
            norm_dates_diff = np.diff(np.sort(norm_dates.values))
            norm_dates_diff = norm_dates_diff[norm_dates_diff > 0]
            if len(norm_dates_diff) == 0:
                norm_dates_diff = [1.0]
            self.median_norm_date_diff[objective] = np.median(norm_dates_diff)
            _LOGGER.info(
                f"Objective: {objective} "
                + f"median norm_date diff: {self.median_norm_date_diff[objective]:.4f}"
            )

        # Storage for results
        self.gps = {}
        self.models = {}

    def build(
        self,
        intercept=0.0,
        prior_ell=[2.0, 2.0],
        prior_eta=[1.5, 2.0],
        cov_func="matern52",
        missing="replace",
    ):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "model")):
            os.mkdir(os.path.join(self.outdir, "model"))

        # Loop over learning objectives
        for objective in self.objectives:
            _LOGGER.info(f"Building model for objective: {objective}")

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
                # ell = pm.Beta("ell", mu=prior_ell[0], sigma=prior_ell[1])
                # ell = pm.LogitNormal("ell", mu=prior_ell[0], sigma=prior_ell[1])
                # ell = pm.Truncated(
                #    "ell",
                #    pm.Beta.dist(mu=prior_ell[0], sigma=prior_ell[1]),
                #    lower=self.mean_norm_date_diff[objective]
                # )
                ell_norm = pm.Gamma("ell_norm", mu=prior_ell[0], sigma=prior_ell[1])
                # ell_norm = pm.HalfNormal("ell_norm", sigma=prior_ell[0])
                ell = pm.Deterministic(
                    "ell", ell_norm * self.median_norm_date_diff[objective]
                )
                # ell = pm.Uniform("ell", lower=prior_ell[0], upper=prior_ell[1])
                # ell = pm.HalfNormal("ell", sigma=prior_ell[0])
                # eta = pm.HalfNormal("eta", sigma=prior_eta[0])
                eta = pm.Gamma("eta", mu=prior_eta[0], sigma=prior_eta[1])

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

            _LOGGER.info(f"Sampling prior for objective: {objective}")
            # Sample prior
            with self.models[objective]:
                self.models[objective].prior = pm.sample_prior_predictive()

    def sample(self, thin_predictive=10):
        # Loop over learning objectives
        for objective in self.objectives:
            _LOGGER.info(f"Sampling posterior for objective: {objective}")
            with self.models[objective]:
                self.models[objective].trace = pm.sample()

            _LOGGER.info(f"Sampling posterior predictive for objective: {objective}")
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
    PARSER.add_argument("outdir", type=str, help="Output directory")
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
        nargs="+",
        default=[1.5, 2.0],
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
        "--start_date",
        type=str,
        default=None,
        help="Start date for model",
    )
    PARSER.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for model",
    )
    ARGS = vars(PARSER.parse_args())

    outdir = ARGS["outdir"]
    datapath = os.path.join(outdir, "assessments")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    models = ObjectiveModels(
        datapath,
        outdir,
        max_score=ARGS["max_score"],
        mastery_score=ARGS["mastery_score"],
        date_samples=ARGS["date_samples"],
        start_date=ARGS["start_date"],
        end_date=ARGS["end_date"],
    )
    models.build(
        intercept=ARGS["intercept"],
        prior_ell=ARGS["prior_ell"],
        prior_eta=ARGS["prior_eta"],
        cov_func=ARGS["cov_func"],
        missing=ARGS["missing"],
    )
    models.sample(thin_predictive=ARGS["thin_predictive"])
    with open(os.path.join(outdir, ARGS["output_file"]), "wb") as f:
        pickle.dump(models, f)
