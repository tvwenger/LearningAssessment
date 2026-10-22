"""
download_assessments.py
Download assessments.

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import argparse

import numpy as np
import os
import glob
import warnings
from datetime import datetime, timezone

from canvasapi import Canvas
import pandas as pd


def download(canvas_url, course_id, overwrite=False):
    canvas = Canvas(canvas_url, os.environ["CANVAS_TOKEN"])
    course = canvas.get_course(course_id)
    outdir = course.name

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(os.path.join(outdir, "assessments")):
        os.mkdir(os.path.join(outdir, "assessments"))

    # cache/log file
    log_file = os.path.join(outdir, "assessments.log")
    log_data = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                log_data.append(line.strip())

    existing_files = glob.glob(os.path.join(outdir, "assessments/*.csv"))
    if overwrite:
        log_data = []
        for file in existing_files:
            os.delete(file)
        existing_files = []

    new_files = []
    assignments = course.get_assignments()
    for assignment in assignments:
        # Get assignment due date
        try:
            due_date = assignment.due_at_date
            date = due_date.astimezone(tz=None).date()
        except AttributeError:
            warnings.warn(f"ALERT: Assignment {assignment} is missing the due date.")
            continue
        if due_date > datetime.now(timezone.utc):
            warnings.warn(f"ALERT: Assignment {assignment} is not due yet.")
            continue

        fname = os.path.join(
            outdir, "assessments", f"{assignment.name.replace(":", "")} {date}.csv"
        )
        if os.path.exists(fname) and fname in log_data:
            warnings.warn(f"Skipping already downloaded assignment {assignment}")
            new_files.append(fname)
            continue
        elif os.path.exists(fname):
            warnings.warn(f"ALERT: Overwriting downloaded assignment {assignment}")
            os.remove(fname)

        # Check that assignment is graded
        if hasattr(assignment, "needs_grading_count") and assignment.needs_grading_count > 0:
            warnings.warn(f"ALERT: Assignment {assignment} has ungraded submissions.")
            continue

        # get rubric items
        rubric_items = {
            rubric["id"]: rubric["description"] for rubric in assignment.rubric
        }

        # Loop over submissions and get rubric assessment data
        all_good = True
        submissions = assignment.get_submissions(include=["rubric_assessment"])
        assessment_data = []
        for submission in submissions:
            student_assessment = {"id": submission.user_id, "name": course.get_user(submission.user_id)}
            try:
                for rubric_id, assessment in submission.rubric_assessment.items():
                    if "points" not in assessment.keys():
                        warnings.warn(
                            f"ALERT: Assignment {assignment} has ungraded rubric item(s)."
                            + f" (student ID: {submission.user_id}, {course.get_user(submission.user_id)})."
                        )
                        student_assessment[rubric_items[rubric_id]] = np.nan
                        all_good = False
                    else:
                        student_assessment[rubric_items[rubric_id]] = assessment[
                            "points"
                        ]
            except AttributeError:
                if not submission.missing:
                    warnings.warn(
                        f"ALERT: Assignment {assignment} is missing rubric assessment"
                        + f" (student ID: {submission.user_id}, {course.get_user(submission.user_id)})."
                    )
                    all_good = False
                for rubric_id in rubric_items.keys():
                    student_assessment[rubric_items[rubric_id]] = np.nan
            assessment_data.append(student_assessment)
        assessment_data = pd.DataFrame(assessment_data)

        # Save to disk
        assessment_data.to_csv(fname, index=False)
        new_files.append(fname)
        print(f"Saving assessment data to {fname}")

        if all_good:
            log_data.append(fname)

    for file in existing_files:
        if file not in new_files:
            warnings.warn("ALERT: Deleting unused assessment file {file}")
            os.remove(file)

    with open(log_file, "w") as f:
        for fname in log_data:
            f.write(fname + "\n")


def main():
    PARSER = argparse.ArgumentParser(
        description="Download learning assessments from Canvas",
        prog="download_assessments.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("canvas_url", type=str, help="Canvas URL")
    PARSER.add_argument("course_id", type=int, help="Course ID")
    PARSER.add_argument("--overwrite", action="store_true", default=False, help="Overwrite all assessments")
    ARGS = vars(PARSER.parse_args())
    download(ARGS["canvas_url"], ARGS["course_id"], overwrite=ARGS["overwrite"])
