"""
send_assessments.py
Distribute assessment dashboards to students

Copyright(C) 025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import argparse
import os
import warnings
from canvasapi import Canvas


def send(canvas_url, course_id):
    canvas = Canvas(canvas_url, os.environ["CANVAS_TOKEN"])
    course = canvas.get_course(course_id)
    outdir = course.name
    user = canvas.get_current_user()
    folders = user.get_folders()
    for folder in folders:
        if folder.name == "conversation attachments":
            break
    else:
        raise ValueError("attachments folder not found!")

    course = canvas.get_course(course_id)
    students = course.get_users(enrollment_type=["student"])

    for student in students:
        dashboard = os.path.join(outdir, "students", f"{student.id}", "index.html")
        if not os.path.exists(dashboard):
            warnings.warn(f"No dashboard found for {student.id}")
            continue

        new_dashboard = os.path.join(
            outdir,
            "students",
            f"{student.id}",
            f"assessment_dashboard_{student.id}.html",
        )
        with open(new_dashboard, "w") as fout:
            with open(dashboard, "r") as fin:
                for line in fin:
                    if "Back to Dashboard" not in line:
                        fout.write(line)

        success, uploaded_file = folder.upload(new_dashboard)
        if not success:
            warnings.warn(f"Failed to upload dashboard for {student.id}")
            continue

        canvas.create_conversation(
            recipients=[student.id],
            subject="Updated Learning Assessment Dashboard",
            body="Please open the attached file in your web browser to view the dashboard.",
            attachment_ids=[uploaded_file["id"]],
            context_code=f"course_{course_id}",
            group_conversation=True,
            bulk_message=True,
        )
        print(f"Sent dashboard to {student.name} ({student.id})")


def main():
    PARSER = argparse.ArgumentParser(
        description="Distribute learning assessment dashboards to students",
        prog="send_assessments.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("canvas_url", type=str, help="Canvas URL")
    PARSER.add_argument("course_id", type=int, help="Course ID")
    ARGS = vars(PARSER.parse_args())
    send(ARGS["canvas_url"], ARGS["course_id"])
