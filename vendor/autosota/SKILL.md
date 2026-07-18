---
name: autosota
description: Local project skill for tsinghua-fib-lab/AutoSOTA. Use whenever the user mentions AutoSOTA, autosota, 自动SOTA, 自动优化论文代码, SOTA 自动优化, or asks to configure, run, audit, or summarize AutoSOTA experiments. Treat AutoSOTA as the GitHub repository https://github.com/tsinghua-fib-lab/AutoSOTA and the local CLI installed on this machine.
---

# AutoSOTA

## Identity

When the user says `AutoSOTA`, `autosota`, or `自动SOTA`, treat it as:

- Source repo: `https://github.com/tsinghua-fib-lab/AutoSOTA`
- Local docs/cache: `/root/autodl-tmp/AutoSOTA`
- Installed CLI: `/usr/local/bin/autosota`
- Installed package: `/opt/node-v20/lib/node_modules/autosota`
- Resolved package path: `/opt/node-v20.20.2-linux-x64/lib/node_modules/autosota`
- Installed CLI version: `0.2.0`
- Node runtime installed for it: `/opt/node-v20`, with `node v20.20.2`
- Claude Code command: `/usr/local/bin/claude`

The GitHub repository README referenced CLI `0.3.0`, but the repository root exposed downloadable packages only up to `autosota-0.2.0.tgz` when installed here. Prefer the installed local version unless the user asks to upgrade or verify the latest package.

## Working Rules

- Use AutoSOTA only after the target codebase can run its baseline evaluation locally.
- Do not let AutoSOTA define the research problem or paper contribution by itself.
- Require a clear `paper/target.md` before optimization: main metric, baseline value, optimization direction, guardrail metrics, eval command, and stopping budget.
- Do not accept result inflation from unfair eval changes, test leakage, hand-merged task results, or protocol drift.
- Keep AutoSOTA outputs in a separate workspace from the target repo.
- For the user's second all-in-one image restoration work, use AutoSOTA as a training/code optimization assistant, not as the main novelty source.

## Common Commands

Check installation:

```bash
autosota --version
autosota doctor
```

Initialize a workspace:

```bash
mkdir -p /root/autodl-tmp/autosota_runs/<run-name>
cd /root/autodl-tmp/autosota_runs/<run-name>
autosota init
```

Run after editing `config.yaml` and `paper/target.md`:

```bash
autosota --repo /path/to/local/repo --devices 0 --max-iter 8
```

Useful inspection:

```bash
autosota sessions
autosota inspect latest
autosota inspect latest --logs
autosota ask "当前最好的一轮是什么，指标和改动是什么？"
autosota steer "下一轮保持评估协议不变，只尝试训练稳定性相关改动。"
```

## Required User Inputs

Before a real run, make sure the user or local files provide:

- API keys in `config.yaml`: `openrouter_api_key`, or separate `claude_api_key` and `research_api_key`.
- A runnable target repository path.
- A baseline eval command that already works without AutoSOTA.
- A concrete `paper/target.md`.
- GPU selection and budget: `--devices`, `--max-iter`, and optionally `--max-total-minutes`.

If these are missing, set up the workspace and tell the user exactly which fields remain blank.
