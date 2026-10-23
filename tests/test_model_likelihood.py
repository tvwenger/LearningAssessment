"""Tests for the ordered-logistic likelihood and prior parameterization."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
from learning_assessment import model as model_module


class ObjectiveLikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "id": [1, 2, 1, 2, 1, 2],
                "name": ["Ada", "Emmy"] * 3,
                "Date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-01",
                        "2026-01-08",
                        "2026-01-08",
                        "2026-01-15",
                        "2026-01-15",
                    ]
                ),
                "Assignment": ["A1", "A1", "A2", "A2", "A3", "A3"],
                "Objective": ["Objective"] * 6,
                "Score": [1, 2, 2, 1, 2, 2],
            }
        )

    def build(self, max_score, scores=None, prior_ell=(2.0, 2.0), data=None):
        if data is None:
            data = self.data.copy()
            data["Score"] = scores
        else:
            data = data.copy()
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        outdir = Path(temporary_directory.name) / "output"

        with (
            patch.object(model_module, "load", return_value=data),
            patch.object(pm, "sample_prior_predictive", return_value=None),
        ):
            models = model_module.ObjectiveModels(
                "unused", str(outdir), max_score=max_score
            )
            models.build(prior_ell=prior_ell)
        return models

    def test_two_rating_rubric_uses_fixed_ordered_logistic(self):
        models = self.build(2, [1, 2, 2, 1, 2, 2])
        model = models.models["Objective"]

        # PyMC represents OrderedLogistic as a categorical random variable.
        self.assertEqual(type(model["score"].owner.op).__name__, "CategoricalRV")
        self.assertNotIn("cutpoints", model.named_vars)
        np.testing.assert_array_equal(models.cutpoints, [0.0])
        self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))

    def test_three_rating_rubric_uses_same_likelihood(self):
        models = self.build(3, [1, 2, 2, 1, 3, 2])
        model = models.models["Objective"]

        self.assertEqual(type(model["score"].owner.op).__name__, "CategoricalRV")
        np.testing.assert_array_equal(models.cutpoints, [-0.5, 0.5])
        self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))

    def test_four_rating_rubric_uses_the_same_unit_spaced_cutpoints(self):
        models = self.build(4, [1, 2, 3, 2, 4, 3])
        model = models.models["Objective"]

        self.assertEqual(type(model["score"].owner.op).__name__, "CategoricalRV")
        np.testing.assert_array_equal(models.cutpoints, [-1.0, 0.0, 1.0])
        self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))

    def test_zero_and_nan_are_both_dropped(self):
        models = self.build(2, [0, 1, np.nan, 0, 2, 2])
        observations = models.likelihood_observations["Objective"]

        self.assertFalse(observations["Score"].isna().any())
        self.assertFalse((observations["Score"] == 0.0).any())
        self.assertEqual(len(observations), 3)
        np.testing.assert_array_equal(observations["Score"], [1.0, 2.0, 2.0])

    def test_absent_csv_row_does_not_create_likelihood_observation(self):
        data = self.data.drop(index=3)
        models = self.build(2, data=data)
        observations = models.likelihood_observations["Objective"]

        self.assertEqual(len(observations), 5)
        student_assignments = set(zip(observations["id"], observations["Assignment"]))
        self.assertNotIn((2, "A2"), student_assignments)

    def test_lengthscale_prior_is_in_calendar_days(self):
        models = self.build(2, [1, 2, 2, 1, 2, 2], prior_ell=(10.0, 2.0))
        model = models.models["Objective"]
        ell = pm.draw(model["ell"], draws=1000, random_seed=42)

        self.assertEqual(models.date_range_days, 14.0)
        self.assertAlmostEqual(ell.mean(), 10.0, delta=0.25)
        self.assertAlmostEqual(ell.std(), 2.0, delta=0.25)

    def test_gamma_prior_parameters_must_be_positive_pairs(self):
        self.assertEqual(
            model_module._gamma_prior_parameters("prior", [2, 0.5]),
            (2.0, 0.5),
        )
        for invalid in ([1], [1, 2, 3], [0, 1], [1, -1]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    model_module._gamma_prior_parameters("prior", invalid)


if __name__ == "__main__":
    unittest.main()
