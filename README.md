# LearningAssessment
A Gaussian Process Model for Student Learning Assessment

This package allows an instructor to "fit" a statistical model to objective-based assessment data in order to quantify student mastery. The model is a Gaussian process with several tunable parameters.

- [LearningAssessment](#learningassessment)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Setup: Canvas Assignments](#setup-canvas-assignments)
  - [Setup: Canvas API Token](#setup-canvas-api-token)
  - [Download Assessment Data](#download-assessment-data)
  - [Building the Model](#building-the-model)
  - [Distributing to Students](#distributing-to-students)
- [The Dashboard](#the-dashboard)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)

# How it works

Consider a single course objective. Given a series of rubric assessments with ratings like "No Evidence", "Below Mastery", and "Mastery", we want to know what is the probability that a student has achieved "Mastery". We model these data using a Gaussian process (GP) prior. A GP prior is a prior over possible functions.

The GP is defined by three parameters: `intercept` is the prior mean of the latent score; `ell` is the correlation timescale in calendar days; and `eta` is the latent-score scatter. `intercept` is fixed by the user. The other parameters, `ell` and `eta`, are shared by all student GPs for a given objective and are inferred from the data.

The timescale is assigned the prior `ell ~ Gamma(mu=2, sigma=2`, where `mu` and `sigma` denote the distribution mean and standard deviation in units of days. The scatter is assigned `eta ~ Gamma(mu=1.5, sigma=2)`. These defaults can be changed with `--prior_ell MU_DAYS SIGMA_DAYS` and `--prior_eta MU SIGMA`.

From the GP prior we predict a latent score, `f`, for each student and assessment. Every rubric is modeled with the same ordered-logistic likelihood. Score 0 (or `NaN`) represents No Evidence / a missing submission and is always excluded from the likelihood. A student or objective not listed in an assessment CSV is structurally absent and likewise contributes no likelihood observation. The likelihood models the `max_score` valid ratings numbered `1, ..., max_score`. For example, `max_score=2` models 1 (Below Mastery) and 2 (Mastery).

Posterior samples from this statistical model are drawn using Markov Chain Monte Carlo (MCMC).

Finally, we produce several dashboards that are useful for both the instructor and the students.
1. Model dashboard: assess model performance, compare prior and posterior distributions, etc.
2. Class dashboard: assess overall class mastery.
3. Student dashboard: assess student mastery.

# Installation
```bash
conda env create -f environment.yml
conda activate learning
python -m pip install .
```

# Usage

## Setup: Canvas Assignments

Assignments and rubric grades are downloaded automatically from Canvas. It is therefore imperative that the Canvas assignments follow a strict format. The following must be true for all assignments on Canvas:
1. Each assignment must have a "due" date.
2. Each assignment must be associated with a rubric.
3. An assignment is only downloaded if the due date is past and all submissions have been graded.

## Setup: Canvas API Token

You will need to have a Canvas API Token in order to use this package. [Follow these instructions to get a token](https://community.canvaslms.com/t5/Canvas-Basics-Guide/How-do-I-manage-API-access-tokens-in-my-user-account/ta-p/615312).

Once you have a Canvas API token, save it as a system variable. For example, in bash:
```bash
export CANVAS_TOKEN="insert token here"
```

## Download Assessment Data

Download assessment data as follows:
```bash
download_assessments <canvas_url> <course_id>
```
where `<canvas_url>` is the URL for your institution's Canvas page, and `<course_id>` is the course ID for which you wish to download assessments. You can determine both by navigating to your Canvas course and looking at the URL. For example, a course page at `https://canvas.csuchico.edu/courses/00000` has `<canvas_url> = https://canvas.csuchico.edu` and `<course_id> = 00000`.

For example:
```bash
download_assessments https://canvas.csuchico.edu 00000
```

This program downloads all assessment data into `./<course_name>/assessments` where `<course_name>` is the Canvas course name. The assessment data for each assignment is named like `<assignment_name> <due_date>.csv`. *Note that this is a destructive process; any existing data in `./<course_name>/assessments` is overwritten.*

## Building the Model

To build and fit the model, use `assessments_model`. There are many parameters.
```bash
assessments_model --help
```

The important parameters for the model specification are:
1. `max_score`: the largest valid rating and the number of likelihood categories. Valid modeled ratings must be consecutive integers `1, ..., max_score`; score 0 is reserved for No Evidence / missing.
2. `mastery_score`: the lowest rating treated as mastery when summarizing posterior predictions.
3. `mastery_threshold`: the posterior probability threshold used to classify mastery.
4. `intercept`: the latent GP prior mean (default `-2`). With `max_score=2` and fixed cutpoint 0, an intercept of 0 corresponds to a 50% prior probability of rating 2 at the GP mean.
5. `prior_ell MU_DAYS SIGMA_DAYS`: the mean and standard deviation, in calendar days, of the Gamma prior on the GP correlation timescale (default `2 2`).
6. `prior_eta MU SIGMA`: the mean and standard deviation of the Gamma prior on latent-score scatter (default `1.5 2`).
7. `cov_func`: the Gaussian process kernel: `matern32` (default), `matern52`, or `expquad`.

Missing observations are not configurable: score 0, `NaN`, and absent CSV rows are always excluded from the likelihood.

The output of this program is a dashboard that can be viewed in a web browser. The dashboard shows summaries for the (1) model, (2) class, and (3) each student.

## Distributing to Students

This package can also distribute the student learning assessment dashboards directly to the students via Canvas message. To do so,
```bash
send_assessments <canvas_url> <course_id>
```

For example:
```bash
send_assessments https://canvas.csuchico.edu 00000
```

# The Dashboard

TBD (under construction)

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/LearningAssessment).

# License and Copyright

Copyright(C) 2025 by Trey V. Wenger

This code is licensed under MIT license (see LICENSE for details)
