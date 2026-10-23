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
import pytensor.tensor as pt

from learning_assessment.loader import load

_LOGGER = logging.getLogger(__name__)


def _gamma_prior_parameters(name, values):
    """Validate a Gamma prior parameterized by its mean and standard deviation."""
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values: mean and sigma")
    mu, sigma = (float(value) for value in values)
    if mu <= 0.0 or sigma <= 0.0:
        raise ValueError(f"{name} mean and sigma must both be positive")
    return mu, sigma


def _fixed_cutpoints(max_score):
    """Return unit-spaced cutpoints for ``max_score`` modeled ratings."""
    return np.arange(max_score - 1, dtype=float) - 0.5 * (max_score - 2)


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
        self.date_range_days = self.date_range / pd.Timedelta(days=1)
        if self.date_range_days <= 0.0:
            raise ValueError("end_date must be later than start_date")
        data["Date Norm"] = (data["Date"] - self.date_offset) / self.date_range
        self.data = data

        # Unique students and objectives
        self.students = sorted(data["id"].unique())
        self.student_index = {
            student: index for index, student in enumerate(self.students)
        }
        self.student_names = {
            student: self.data.loc[self.data["id"] == student, "name"].values[0]
            for student in self.students
        }
        self.objectives = sorted(data["Objective"].unique())

        self.norm_date_axis = np.linspace(0.0, 1.0, date_samples)
        if int(max_score) != max_score or max_score < 2:
            raise ValueError("max_score must be an integer greater than or equal to 2")
        self.max_score = int(max_score)
        self.cutpoints = _fixed_cutpoints(self.max_score)
        self.outdir = outdir
        self.mastery_score = mastery_score
        self.mastery_threshold = mastery_threshold
        self.var_names = ["eta", "ell"]

        self.observations = {}
        self.assessments = {}
        self.last_assignment = {}
        self.last_assignment_date = {}
        self.last_assignment_norm_date = {}
        for objective in self.objectives:
            objective_data = data.loc[data["Objective"] == objective].copy()
            assessment_data = (
                objective_data[["Assignment", "Date", "Date Norm"]]
                .drop_duplicates()
                .sort_values(["Date", "Assignment"], ignore_index=True)
            )
            if assessment_data["Assignment"].duplicated().any():
                raise ValueError(
                    f"Objective {objective} has duplicate assignment names on "
                    "different dates"
                )

            assessment_data["Assignment Index"] = np.arange(len(assessment_data))
            assignment_index = assessment_data.set_index("Assignment")[
                "Assignment Index"
            ]
            objective_data["Student Index"] = objective_data["id"].map(
                self.student_index
            )
            objective_data["Assignment Index"] = objective_data["Assignment"].map(
                assignment_index
            )
            objective_data = objective_data.sort_values(
                ["Date", "Assignment", "id"], ignore_index=True
            )
            if objective_data.duplicated(["id", "Assignment"]).any():
                raise ValueError(
                    f"Objective {objective} contains duplicate student-assignment "
                    "observations"
                )

            self.observations[objective] = objective_data
            self.assessments[objective] = assessment_data

            # Index and date of the last assessment for this objective.
            self.last_assignment[objective] = int(
                assessment_data["Date Norm"].to_numpy().argmax()
            )
            last_assessment = assessment_data.iloc[self.last_assignment[objective]]
            self.last_assignment_date[objective] = last_assessment["Date"]
            self.last_assignment_norm_date[objective] = last_assessment["Date Norm"]

        # Storage for results
        self.gps = {}
        self.models = {}
        self.likelihood_observations = {}

    def build(
        self,
        intercept=-2.0,
        prior_ell=(2.0, 2.0),
        prior_eta=(1.5, 2.0),
        cov_func="matern32",
    ):
        ell_mu, ell_sigma = _gamma_prior_parameters("prior_ell", prior_ell)
        eta_mu, eta_sigma = _gamma_prior_parameters("prior_eta", prior_eta)

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, "model")):
            os.mkdir(os.path.join(self.outdir, "model"))

        # Loop over learning objectives
        for objective in self.objectives:
            _LOGGER.info(f"Building model for objective: {objective}")

            # Zero and NaN both represent No Evidence / a missing submission
            # and are excluded from the likelihood. Students or objectives not
            # listed in an assessment CSV have no long-form row and therefore
            # are also absent from the likelihood.
            observations = self.observations[objective].copy()
            is_missing = observations["Score"].isna() | (observations["Score"] == 0.0)
            observations = observations.loc[~is_missing].reset_index(drop=True)
            self.likelihood_observations[objective] = observations

            scores = observations["Score"].to_numpy()
            if np.any(scores < 1.0) or np.any(scores > self.max_score):
                raise ValueError(
                    f"Objective {objective} has scores outside [1, {self.max_score}]"
                )
            if not np.allclose(scores, np.round(scores)):
                raise ValueError(f"Objective {objective} has non-integer scores")

            assessment_data = self.assessments[objective]
            assignment_names = assessment_data["Assignment"].to_numpy()

            coords = {
                "student": self.students,
                "assignment": assignment_names,
                "observation": np.arange(len(observations)),
                "norm_date": self.norm_date_axis,
            }

            with pm.Model(coords=coords) as self.models[objective]:
                # gp parameters
                ell = pm.Gamma("ell", mu=ell_mu, sigma=ell_sigma)
                ell_normalized = ell / self.date_range_days
                eta = pm.Gamma("eta", mu=eta_mu, sigma=eta_sigma)

                if cov_func == "matern52":
                    cov = eta**2.0 * pm.gp.cov.Matern52(1, ell_normalized)
                elif cov_func == "matern32":
                    cov = eta**2.0 * pm.gp.cov.Matern32(1, ell_normalized)
                elif cov_func == "expquad":
                    cov = eta**2.0 * pm.gp.cov.ExpQuad(1, ell_normalized)
                else:
                    raise ValueError(f"Unsupported cov_func: {cov_func}")

                mean_func = pm.gp.mean.Constant(intercept)

                self.gps[objective] = pm.gp.Latent(mean_func=mean_func, cov_func=cov)
                f = self.gps[objective].prior(
                    "f",
                    X=assessment_data["Date Norm"].to_numpy()[:, None],
                    dims=["student", "assignment"],
                )

                # Map the student-assignment GP values to the long-form rows.
                f_observed = pm.Deterministic(
                    "f_observed",
                    f[
                        observations["Student Index"].to_numpy(dtype=int),
                        observations["Assignment Index"].to_numpy(dtype=int),
                    ],
                    dims="observation",
                )

                _ = pm.OrderedLogistic(
                    "score",
                    eta=f_observed,
                    cutpoints=self.cutpoints,
                    observed=scores.astype(int) - 1,
                    dims="observation",
                    compute_p=False,
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
                _ = pm.OrderedLogistic(
                    "score_assessment_pred",
                    eta=self.models[objective]["f"],
                    cutpoints=self.cutpoints,
                    dims=["student", "assignment"],
                    compute_p=False,
                )
                f_pred = self.gps[objective].conditional(
                    "f_pred",
                    self.norm_date_axis[:, None],
                    dims=["student", "norm_date"],
                )
                _ = pm.OrderedLogistic(
                    "score_pred",
                    eta=f_pred,
                    cutpoints=self.cutpoints,
                    dims=["student", "norm_date"],
                    compute_p=True,
                )
                _ = pm.Deterministic(
                    "p_pred",
                    pt.sum(
                        self.models[objective]["score_pred_probs"]
                        * np.arange(self.max_score),
                        axis=-1,
                    )
                    / (self.max_score - 1),
                    dims=["student", "norm_date"],
                )
                posterior_predictive = pm.sample_posterior_predictive(
                    self.models[objective].trace.sel(
                        draw=slice(None, None, thin_predictive)
                    ),
                    var_names=[
                        "score",
                        "score_assessment_pred",
                        "p_pred",
                        "score_pred",
                    ],
                )
                self.models[objective].posterior_predictive = az.extract(
                    posterior_predictive,
                    group="posterior_predictive",
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
        nargs=2,
        metavar=("MU", "SIGMA"),
        default=[2.0, 2.0],
        help="Mean and standard deviation (days) of the Gamma prior on ell",
    )
    PARSER.add_argument(
        "--prior_eta",
        type=float,
        nargs=2,
        metavar=("MU", "SIGMA"),
        default=[1.5, 2.0],
        help="Mean and standard deviation of the Gamma prior on eta",
    )
    PARSER.add_argument(
        "--cov_func",
        type=str,
        default="matern32",
        choices=["matern32", "matern52", "expquad"],
        help="Gaussian process covariance function",
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
        mastery_threshold=ARGS["mastery_threshold"],
        date_samples=ARGS["date_samples"],
        start_date=ARGS["start_date"],
        end_date=ARGS["end_date"],
    )
    models.build(
        intercept=ARGS["intercept"],
        prior_ell=ARGS["prior_ell"],
        prior_eta=ARGS["prior_eta"],
        cov_func=ARGS["cov_func"],
    )
    models.sample(thin_predictive=ARGS["thin_predictive"])
    with open(os.path.join(outdir, ARGS["output_file"]), "wb") as f:
        pickle.dump(models, f)
