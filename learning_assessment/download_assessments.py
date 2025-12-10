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


def download(canvas_url, course_id):
    canvas = Canvas(canvas_url, os.environ["CANVAS_TOKEN"])
    course = canvas.get_course(course_id)
    outdir = course.name

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(os.path.join(outdir, "assessments")):
        os.mkdir(os.path.join(outdir, "assessments"))

    # delete existing assessments
    files = glob.glob(os.path.join(outdir, "assessments/*.csv"))
    for file in files:
        os.remove(file)

    canvas = Canvas(canvas_url, os.environ["CANVAS_TOKEN"])
    course = canvas.get_course(course_id)

    assignments = course.get_assignments()
    for assignment in assignments:
        # Check that assignment is graded
        if assignment.needs_grading_count > 0:
            warnings.warn(f"ALERT: Assignment {assignment} has ungraded submissions.")
            continue

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

        # get rubric items
        rubric_items = {
            rubric["id"]: rubric["description"] for rubric in assignment.rubric
        }

        # Loop over submissions and get rubric assessment data
        submissions = assignment.get_submissions(include=["rubric_assessment"])
        assessment_data = []
        for submission in submissions:
            student_assessment = {"id": submission.user_id}
            if submission.missing:
                for rubric_id in rubric_items.keys():
                    student_assessment[rubric_items[rubric_id]] = np.nan
            else:
                try:
                    for rubric_id, assessment in submission.rubric_assessment.items():
                        if "points" not in assessment.keys():
                            warnings.warn(
                                f"ALERT: Assignment {assignment} has ungraded rubric item(s)."
                            )
                            student_assessment[rubric_items[rubric_id]] = np.nan
                        else:
                            student_assessment[rubric_items[rubric_id]] = assessment[
                                "points"
                            ]
                except AttributeError:
                    warnings.warn(
                        f"ALERT: Assignment {assignment} is missing rubric assessment."
                    )
                    for rubric_id in rubric_items.keys():
                        student_assessment[rubric_items[rubric_id]] = np.nan
            assessment_data.append(student_assessment)
        assessment_data = pd.DataFrame(assessment_data)

        # Save to disk
        fname = os.path.join(
            outdir, "assessments", f"{assignment.name.replace(":", "")} {date}.csv"
        )
        assessment_data.to_csv(fname, index=False)
        print(f"Saving assessment data to {fname}")


def main():
    PARSER = argparse.ArgumentParser(
        description="Download learning assessments from Canvas",
        prog="download_assessments.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("canvas_url", type=str, help="Canvas URL")
    PARSER.add_argument("course_id", type=int, help="Course ID")
    ARGS = vars(PARSER.parse_args())
    download(ARGS["canvas_url"], ARGS["course_id"])
