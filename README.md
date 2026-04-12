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
- Dockerized deployment path for Hugging Face Spaces

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

- `inspect_ticket`
- `request_context`
- `set_priority`
- `set_route`
- `set_resolution`
- `escalate`
- `rank_queue`
- `finalize`

## Reward Design

`RewardModel` is a Pydantic model with:

- `value`
- `components`
- `rationale`

Reward shaping is dense, not sparse:

- positive reward for discovering required context
- positive reward for correct intermediate decisions
- positive reward for correct queue ranking progress
- terminal reward from the deterministic grader score
- penalties for invalid actions, redundant actions, and wasted steps

This creates learning or evaluation signal over the full trajectory.

## Tasks

### Easy: Account Takeover Triage

Objective: correctly handle an urgent suspected account takeover with unauthorized ad spend.

Expected difficulty: easy.

Success criteria:

- request the right security and billing context
- assign `urgent`
- route to `account_security`
- choose `temporary_lock_and_manual_recovery`
- escalate to `security_specialist`

### Medium: Monetization Payout Hold

Objective: investigate a missing creator payout and avoid unsafe release of funds.

Expected difficulty: medium.

Success criteria:

- discover tax-expiry and compliance-hold context
- assign `high`
- route to `monetization_compliance`
- choose `request_tax_renewal`
- avoid unnecessary escalation

### Hard: Mixed Support Queue Triage

Objective: prioritize and resolve a heterogeneous queue under SLA pressure.

Expected difficulty: hard.

Success criteria:

- correctly rank the queue
- assign route and priority for each ticket
- choose correct resolutions
- escalate only the security-critical case

## Graders

Each task has a deterministic grader that returns a score in `0.0` to `1.0`.

- Easy grader weights context, priority, route, resolution, and escalation
- Medium grader weights context and policy-safe resolution more heavily
- Hard grader scores per-ticket handling and queue ranking

Programmatic graders live in [support_ops_env/graders](/home/batman/Downloads/presentation_template/support_ops_env/support_ops_env/graders).

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

Run the OpenAI baseline if you have an API key:

```bash
export OPENAI_API_KEY=your_key_here
python scripts/run_baseline.py --model gpt-4.1-mini
```

Validate metadata:

```bash
bash scripts/validate_env.sh
```

If the `openenv` CLI is installed, the script will also run `openenv validate openenv.yaml`.

## Baseline Scores

The repository now includes a deterministic baseline in [run_rule_baseline.py](/home/batman/Downloads/presentation_template/support_ops_env/scripts/run_rule_baseline.py), so you can produce reproducible scores without any external API.

In this workspace, use:

```bash
python scripts/run_rule_baseline.py
```

This writes `rule_baseline_results.json` with per-task transcripts and the average score.

The current deterministic baseline score from this workspace is:

- `easy_account_takeover`: `1.0`
- `medium_payout_hold`: `1.0`
- `hard_queue_triage`: `1.0`
- average: `1.0`

The OpenAI baseline in [run_baseline.py](/home/batman/Downloads/presentation_template/support_ops_env/scripts/run_baseline.py) is still available as an optional comparison path after installing dependencies and setting `OPENAI_API_KEY`.

## Hugging Face Space Deployment

This repository includes:

- `Dockerfile`
- `app.py`
- `openenv.yaml`

To deploy as a Docker Space:

1. Create a new Hugging Face Space with SDK set to Docker.
2. Upload this repository.
3. Add the `openenv` tag in the Space metadata.
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
├── tests/
├── app.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
