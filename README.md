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

The GP is defined by three parameters: `intercept` is the null-hypothesis normalized score, `ell` is the normalized timescale (0 being the first assessment due date and 1 being the most recent assessment due date) over which we expect scores to be correlated, and `eta` is the normalized score scatter. `intercept` is a tunable parameter set by the user. The other parameters, `ell` and `eta`, are treated as hyperparameters. That is, these parameters are assumed to govern the GPs for all students, and they are inferred from the data. The assumed prior on `ell` is a logit-normal distribution, and on `eta` is a Half-normal distribution.

From the GP prior we predict the normalized score, `f`, for each student and each assignment. The normalized score is transformed into a probability of a given rating using a logistic transform. We assume a Binomial likelihood when comparing the predicted ratings to the actual ratings.

Posterior samples from this statistical model are drawn using Monte Carlo Markov Chains (MCMC).

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
```
usage: model.py [-h] [--max_score MAX_SCORE] [--mastery_score MASTERY_SCORE] [--mastery_threshold MASTERY_THRESHOLD] [--score_labels SCORE_LABELS [SCORE_LABELS ...]]
                [--date_samples DATE_SAMPLES] [--trend_plot_width TREND_PLOT_WIDTH] [--trend_plot_height TREND_PLOT_HEIGHT] [--prob_plot_width PROB_PLOT_WIDTH]
                [--prob_plot_height PROB_PLOT_HEIGHT] [--intercept INTERCEPT] [--prior_ell PRIOR_ELL [PRIOR_ELL ...]] [--prior_eta PRIOR_ETA] [--cov_func COV_FUNC]
                [--missing MISSING] [--thin_predictive THIN_PREDICTIVE] [--skip_pairplot] [--percentiles PERCENTILES PERCENTILES]
                canvas_url course_id
```

The important parameters for the model specification are:
1. `max_score`: the maximum "points" per rubric item.
2. `mastery_threshold`: the probability threshold for "mastery".
3. `intercept`: the Gaussian process prior mean. A value of 0 corresponds to a 50% prior probability of mastery. Use this to tune the model based on your prior expectation for mastery in the absence of evidence. The mastery score will 
4. `prior_ell`: the shape of the Gaussian process prior normalized lengthscale.
5. `prior_eta`: the shape of the Gaussian process prior scatter.
6. `cov_func`: the Gaussian process kernel.
7. `missing`: either `replace` (replace missing scores with the lowest non-zero score) or `drop` (do not include missing scores in the model).

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