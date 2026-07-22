# Fable — AI-Powered Automation

Fable is a smart automation toolkit powered by Claude (Anthropic). It lets you automate repetitive tasks on your computer using natural language — no code required.

Inspired by abandoned RPA projects like `automagica`, rebuilt from scratch with modern AI at the core.

## What it does

- Automate desktop tasks (clicks, file ops, keyboard, browser)
- Describe what you want in plain language → Fable figures out the steps
- Schedule, monitor, and chain automations together

## Architecture

```
User (natural language)
        ↓
   Claude API (intent parsing + step generation)
        ↓
   Fable Runner (executes steps on the OS)
        ↓
   Result / Feedback loop
```

## Stack

- Python 3.11+
- Anthropic SDK (`claude-sonnet-4-6` for fast tasks, `claude-opus-4-8` for complex reasoning)
- `pyautogui` for desktop control
- `playwright` for browser automation
- FastAPI for the local REST API

## Quick Start

```bash
pip install fable-ai
fable run "Download all PDF attachments from my Gmail and save them to ~/Documents"
```

## Monetization model

| Tier | Price | Limits |
|------|-------|--------|
| Free | $0 | 50 automations/month |
| Pro | $19/mo | Unlimited, scheduling |
| Teams | $49/mo | Multi-user, audit logs |
| Enterprise | Custom | On-prem, SSO |

## Status

Early development. Core runner in progress.
