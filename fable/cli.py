"""Fable CLI — `fable run "do something"`"""

import argparse
import json
import os
import sys
from fable.core import FableRunner


def main():
    parser = argparse.ArgumentParser(
        prog="fable",
        description="AI-powered automation. Describe a task, Fable runs it.",
    )
    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="Run an automation task")
    run_cmd.add_argument("task", help="Task description in plain language")
    run_cmd.add_argument("--dry-run", action="store_true", help="Plan only, don't execute")

    plan_cmd = sub.add_parser("plan", help="Show steps without executing")
    plan_cmd.add_argument("task", help="Task description in plain language")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    runner = FableRunner(api_key=api_key)

    if args.command in ("plan", "run") and (args.command == "plan" or args.dry_run):
        steps = runner.plan(args.task)
        print(json.dumps(steps, indent=2))
        return

    if args.command == "run":
        results = runner.run(args.task)
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['step']['description']}")
            print(f"    → {r['result']}")


if __name__ == "__main__":
    main()
