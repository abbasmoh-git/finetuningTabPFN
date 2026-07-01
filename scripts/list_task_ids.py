"""
list_task_ids.py
----------------
Print all classification task IDs from the TabArena benchmark suite (457).
Used by submit_all.sh to determine which jobs to submit.

Usage:
    python scripts/list_task_ids.py
    python scripts/list_task_ids.py --suite_id 457
"""

import argparse
import openml

parser = argparse.ArgumentParser()
parser.add_argument("--suite_id", type=int, default=457)
args = parser.parse_args()

suite = openml.study.get_suite(args.suite_id)
for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id)
    if task.task_type == "Supervised Classification":
        print(task_id)
