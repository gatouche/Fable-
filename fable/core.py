"""Core automation engine — parses intent via Claude, executes steps."""

import json
import anthropic

SYSTEM_PROMPT = """You are Fable, an automation assistant.
When given a task in natural language, you return a JSON list of steps to execute.

Each step has this shape:
{
  "action": "click|type|open|wait|browser_goto|file_write|shell",
  "params": { ... action-specific params ... },
  "description": "human-readable description of this step"
}

Only return valid JSON. No prose, no markdown fences."""


class FableRunner:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6"

    def plan(self, task: str) -> list[dict]:
        """Ask Claude to break the task into executable steps."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}],
        )
        raw = message.content[0].text
        return json.loads(raw)

    def run(self, task: str) -> list[dict]:
        """Plan and execute a task end-to-end."""
        steps = self.plan(task)
        results = []
        for step in steps:
            result = self._execute(step)
            results.append({"step": step, "result": result})
        return results

    def _execute(self, step: dict) -> str:
        action = step.get("action")
        params = step.get("params", {})

        if action == "shell":
            import subprocess
            out = subprocess.run(
                params["command"], shell=True, capture_output=True, text=True
            )
            return out.stdout or out.stderr

        if action == "file_write":
            path = params["path"]
            content = params["content"]
            with open(path, "w") as f:
                f.write(content)
            return f"Written to {path}"

        if action == "browser_goto":
            # Playwright integration point
            return f"[browser] Would navigate to {params.get('url')}"

        if action == "wait":
            import time
            time.sleep(params.get("seconds", 1))
            return "waited"

        return f"[stub] action '{action}' not yet implemented"
