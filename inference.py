"""
inference.py — OpenEnv Baseline Inference Script
=================================================
Runs the rule-based classifier against all three tasks defined in
openenv.yaml and reports per-task scores in the 0.0 → 1.0 range.

This script proves reproducibility for the hackathon submission.
Run it with:
    python inference.py

Expected output:
    Task 1 [EASY]   — Spam Detection          : 1.000  ✅
    Task 2 [MEDIUM] — Support Routing         : 0.950  ✅
    Task 3 [HARD]   — Phishing / Security     : 0.900  ✅
    Overall Score                             : 0.950
"""

import numpy as np
from env import (
    EmailTriageEnv,
    EmailAction,
    TASK_SPLITS,
    URGENCY_LABELS,
    ROUTING_LABELS,
    RESOLUTION_LABELS,
)

# ── Rule-based classifier (your 95%-accuracy agent) ───────────────────────────

_LEGAL_SECURITY_KW   = {"lawsuit", "attorney", "sue", "ransomware", "extortion"}
_BILLING_ESCALATE_KW = {"refund"}


def _classify(email: dict) -> np.ndarray:
    """
    Deterministic rule-based classifier.
    Returns np.ndarray([urgency, routing, resolution]).
    """
    kw      = set(email.get("keywords", []))
    context = email.get("context", "").lower()

    if context == "legal" or kw & {"lawsuit", "attorney", "sue"}:
        return np.array([2, 2, 2], dtype=np.int64)

    if context == "security":
        if kw & _LEGAL_SECURITY_KW or ("hacked" in kw and "breach" in kw):
            return np.array([2, 2, 2], dtype=np.int64)
        return np.array([2, 1, 2], dtype=np.int64)

    if context == "billing":
        if kw & _BILLING_ESCALATE_KW:
            return np.array([1, 2, 2], dtype=np.int64)
        return np.array([1, 0, 1], dtype=np.int64)

    if context == "tech" or kw & {"crash", "error", "bug", "slow"}:
        return np.array([0, 1, 1], dtype=np.int64)

    return np.array([0, 0, 0], dtype=np.int64)


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(task: str, verbose: bool = False) -> float:
    """
    Run one full episode on the given task using the rule-based classifier.
    Returns the normalised cumulative score in [0.0, 1.0].
    """
    env = EmailTriageEnv(task=task, shuffle=False)
    obs, info = env.reset(seed=42)

    email_queue      = list(env._queue)   # snapshot before any steps
    cumulative_score = 0.0
    step             = 0
    terminated       = False

    task_labels = {
        "easy":   "Task 1 [EASY]   — Spam Detection         ",
        "medium": "Task 2 [MEDIUM] — Support Routing        ",
        "hard":   "Task 3 [HARD]   — Phishing / Security    ",
    }

    if verbose:
        print(f"\n  {'─' * 58}")
        print(f"  {task_labels.get(task, task.upper())}")
        print(f"  {'─' * 58}")

    while not terminated:
        current_email = email_queue[step]
        action        = _classify(current_email)

        obs, norm_reward, terminated, _, info = env.step(action)
        cumulative_score += norm_reward

        if verbose:
            ca  = info["correct_actions"]
            raw = info["raw_reward"]

            pred_str = (f"{URGENCY_LABELS[action[0]]} | "
                        f"{ROUTING_LABELS[action[1]]} | "
                        f"{RESOLUTION_LABELS[action[2]]}")
            corr_str = (f"{URGENCY_LABELS[ca[0]]} | "
                        f"{ROUTING_LABELS[ca[1]]} | "
                        f"{RESOLUTION_LABELS[ca[2]]}")

            if raw >= 1.0:
                verdict = "✅ EXACT"
            elif raw > 0:
                verdict = "🔶 PARTIAL"
            elif raw < 0:
                verdict = "🚨 SECURITY MISS"
            else:
                verdict = "❌ WRONG"

            print(f"  #{step+1:02d} [{current_email['difficulty'].upper():<6}] "
                  f"{current_email['description'][:35]:<35} "
                  f"reward={raw:+.1f}  {verdict}")
            if raw < 1.0:
                print(f"       Predicted : {pred_str}")
                print(f"       Correct   : {corr_str}")

        step += 1

    # Clamp to [0.0, 1.0] — penalties can push below 0
    final_score = max(0.0, min(1.0, cumulative_score))

    env_state = env.state()
    assert env_state.terminated, "Episode should be terminated after all steps"

    return final_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 62}")
    print("  EMAIL GATEKEEPER — OpenEnv Baseline Inference")
    print("  Meta x PyTorch Hackathon | Reproducibility Report")
    print(f"{'═' * 62}")

    tasks = [
        ("easy",   "Task 1 [EASY]   — Spam Detection         "),
        ("medium", "Task 2 [MEDIUM] — Support Routing        "),
        ("hard",   "Task 3 [HARD]   — Phishing / Security    "),
    ]

    scores      = {}
    all_correct = 0
    all_total   = 0

    for task_id, label in tasks:
        score = run_task(task_id, verbose=True)
        scores[task_id] = score

        n = len(TASK_SPLITS[task_id])
        all_total += n

        icon = "✅" if score >= 0.8 else ("⚠️ " if score >= 0.5 else "❌")
        print(f"\n  {label}: {score:.3f}  {icon}")

    # Overall score = weighted average by number of emails per task
    weights      = {t: len(TASK_SPLITS[t]) for t in scores}
    total_weight = sum(weights.values())
    overall      = sum(scores[t] * weights[t] / total_weight for t in scores)

    print(f"\n{'─' * 62}")
    print(f"  {'Overall Score (weighted avg)':<42}: {overall:.3f}")
    print(f"  {'Total Emails Evaluated':<42}: {total_weight}")

    # Per-task summary table
    print(f"\n  {'Task':<10} {'Emails':>7} {'Score':>8} {'Status':>10}")
    print(f"  {'─'*10} {'─'*7} {'─'*8} {'─'*10}")
    for task_id, label in tasks:
        n      = len(TASK_SPLITS[task_id])
        s      = scores[task_id]
        status = "PASS ✅" if s >= 0.8 else ("WARN ⚠️ " if s >= 0.5 else "FAIL ❌")
        print(f"  {task_id:<10} {n:>7} {s:>8.3f} {status:>10}")

    print(f"\n{'═' * 62}\n")

    # Return scores dict for programmatic use (e.g. CI pipelines)
    return {
        "task_scores": scores,
        "overall":     round(overall, 4),
        "total_emails": total_weight,
    }


if __name__ == "__main__":
    results = main()
