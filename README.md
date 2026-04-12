---
title: SupportOpsEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - evaluation
---

# SupportOpsEnv

SupportOpsEnv is a multi-step environment for evaluating agents on realistic customer support operations. The agent behaves like a support analyst: it reviews ticket summaries, requests missing context, assigns priority, chooses the correct internal route, selects a resolution, escalates when needed, and finalizes the case. This models a genuine workflow used by support operations, trust and safety, monetization, and account-recovery teams.

The environment is designed to score well against OpenEnv-style hackathon criteria:

- Real-world task simulation instead of a toy game
- Three deterministic tasks with easy, medium, and hard difficulty
- Dense reward shaping across the trajectory
- Typed observation, action, and reward models
- Reproducible OpenAI baseline runner
- Reproducible rule-based baseline runner that works with no API key
- Dockerized deployment on Hugging Face Spaces

## Environment Motivation

Support queue triage is one of the clearest real-world benchmarks for agent quality:

- Humans perform it every day
- It requires multi-step reasoning, not one-shot classification
- Progress can be measured deterministically
- It exposes practical agent failure modes such as premature resolution, wrong escalation, and poor prioritization

## Observation Space

`Observation` is a Pydantic model with:

- `task_id`: active task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `title`: task title
- `instruction`: natural-language objective
- `queue_mode`: whether the task contains multiple tickets
- `tickets`: list of ticket observations
- `remaining_steps`: steps left in the episode
- `available_actions`: valid action names
- `current_queue_order`: current queue ranking, if any
- `score_hint`: latest intermediate grader snapshot

Each ticket observation contains:

- `ticket_id`
- `summary`
- `visible_context`
- `discovered_context`
- `selected_priority`
- `selected_route`
- `selected_resolution`
- `escalation_team`

## Action Space

`Action` is a Pydantic model with:

- `action_type`
- `target`
- `value`

Supported `action_type` values:

| `action_type`    | `target`   | `value` example                        |
|------------------|------------|----------------------------------------|
| `inspect_ticket` | ticket ID  | `""`                                   |
| `request_context`| ticket ID  | `"tax_status"`                         |
| `set_priority`   | ticket ID  | `"urgent"` / `"high"` / `"normal"` / `"low"` |
| `set_route`      | ticket ID  | `"account_security"` / `"billing_refunds"` / `"monetization_compliance"` / `"policy_appeals"` |
| `set_resolution` | ticket ID  | `"temporary_lock_and_manual_recovery"` / `"request_tax_renewal"` / `"approve_refund"` / `"expedited_human_review"` |
| `escalate`       | ticket ID  | `"security_specialist"`                |
| `rank_queue`     | `"queue"`  | `"T2,T1,T3"`                           |
| `finalize`       | ticket ID  | `""`                                   |

## Reward Design

`RewardModel` is a Pydantic model with:

- `value`: scalar reward for this step
- `components`: dict of named sub-rewards
- `rationale`: human-readable explanation

Reward shaping is dense, not sparse:

- positive reward for discovering required context keys
- positive reward for correct intermediate decisions (priority, route, resolution)
- positive reward for correct queue ranking progress
- terminal reward from the deterministic grader score
- penalties for invalid actions, redundant actions, and wasted steps

This creates a learning or evaluation signal over the full trajectory, not just at episode end.

## Tasks

### Easy: Account Takeover Triage

Objective: correctly handle an urgent suspected account takeover with unauthorized ad spend.

Success criteria:

- request the right security and billing context
- assign `urgent`
- route to `account_security`
- choose `temporary_lock_and_manual_recovery`
- escalate to `security_specialist`

### Medium: Monetization Payout Hold

Objective: investigate a missing creator payout and avoid unsafe release of funds.

Success criteria:

- discover tax-expiry and compliance-hold context
- assign `high`
- route to `monetization_compliance`
- choose `request_tax_renewal`
- avoid unnecessary escalation

### Hard: Mixed Support Queue Triage

Objective: prioritize and resolve a heterogeneous queue of three tickets under SLA pressure.

Success criteria:

- correctly rank the queue by urgency
- assign route and priority for each ticket independently
- choose correct resolutions per ticket
- escalate only the security-critical case

## Graders

Each task has a deterministic grader that returns a score in `[0.0, 1.0]`.

- Easy grader weights context discovery, priority, route, resolution, and escalation
- Medium grader weights context and policy-safe resolution more heavily
- Hard grader scores per-ticket handling and queue ranking independently

Programmatic graders live in [`support_ops_env/graders/`](./support_ops_env/graders/).

## Baseline Scores

### Rule-based baseline (no API key required)

The deterministic rule-based baseline always takes the optimal action sequence and is used as a sanity check that the graders are correct and reachable:

| Task                    | Score |
|-------------------------|-------|
| `easy_account_takeover` | 1.000 |
| `medium_payout_hold`    | 1.000 |
| `hard_queue_triage`     | 1.000 |
| **average**             | **1.000** |

### LLM baseline (GPT-4.1-mini)

These are the reproducible scores from the OpenAI baseline runner. They demonstrate that the environment provides a genuine challenge to frontier models, particularly on the hard task:

| Task                    | Score | Notes |
|-------------------------|-------|-------|
| `easy_account_takeover` | ~0.20 | Model skips mandatory set_priority / set_route / set_resolution before finalize |
| `medium_payout_hold`    | ~0.35 | Correct context discovery but premature finalize |
| `hard_queue_triage`     | ~0.13 | Multi-ticket ranking and per-ticket mandatory actions not completed |
| **average**             | **~0.23** | |

The gap between the rule baseline and the LLM baseline confirms the reward function produces genuine signal and the hard task challenges frontier models.

## Setup

```bash
cd support_ops_env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the local tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run the app locally:

```bash
python app.py
```

Run the default no-API baseline:

```bash
python scripts/run_rule_baseline.py
```

Run the OpenAI baseline:

```bash
export OPENAI_API_KEY=your_key_here
python scripts/run_baseline.py --model gpt-4.1-mini
```

Validate OpenEnv metadata:

```bash
bash scripts/validate_env.sh
# If the openenv CLI is installed, this also runs: openenv validate openenv.yaml
```

## API Quick Start

The live environment is available at `https://suppops-supportopsenv.hf.space`.

Reset to a task:

```bash
curl -X POST https://suppops-supportopsenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_account_takeover"}'
```

Take a step:

```bash
curl -X POST https://suppops-supportopsenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect_ticket", "target": "T1", "value": ""}}'
```

Inspect the full environment state:

```bash
curl https://suppops-supportopsenv.hf.space/state
```

Get JSON schemas for all models:

```bash
curl https://suppops-supportopsenv.hf.space/schema
```

## Hugging Face Space Deployment

This repository includes a `Dockerfile`, `app.py`, and `openenv.yaml` and deploys as a Docker Space.

1. Create a new Hugging Face Space with SDK set to Docker.
2. Push this repository to the Space.
3. Add the `openenv` tag in the Space metadata (already present in this README's frontmatter).
4. Optionally set `OPENAI_API_KEY` as a Space secret for baseline experiments.

## Project Structure

```text
support_ops_env/
├── support_ops_env/
│   ├── env.py
│   ├── models.py
│   ├── reward.py
│   ├── state.py
│   ├── data/
│   ├── graders/
│   └── tasks/
├── scripts/
│   ├── run_baseline.py
│   ├── run_rule_baseline.py
│   └── validate_env.sh
├── tests/
├── app.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
