"""
MovieLens RL — Orchestrated Multi-Agent Training
=================================================
Trains the OrchestratorAgent (coordinating LinUCB + Q-Learning + Popularity)
alongside the GenreDiversityTool, and generates comparison figures.

Run: python train_orchestrated.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from data_loader import build_interactions, ARMS, N_ARMS, STATE_NAMES, N_STATES
from agents.agents import LinUCBAgent, QLearningAgent, RandomAgent, PopularityAgent
from agents.orchestrator import OrchestratorAgent, SPARSE_THRESHOLD, CONFIDENCE_THRESHOLD
from agents.diversity_tool import GenreDiversityTool

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
WINDOW = 500

COLORS = {
    "Orchestrator": "#1A3A5C",
    "LinUCB":       "#2D6A9F",
    "Q-Learning":   "#E67E22",
    "Popularity":   "#8E44AD",
    "Random":       "#C0392B",
}


# ---------------------------------------------------------------------------
# Baseline training (reuse from train_and_evaluate.py pattern)
# ---------------------------------------------------------------------------

def train_single_agent(agent, interactions, agent_type="linucb"):
    rewards, correct = [], []
    for ix in interactions:
        ctx, state, true_arm, reward = ix["context"], ix["state"], ix["arm"], ix["reward"]
        if agent_type == "linucb":
            chosen, _ = agent.select_arm(ctx)
        elif agent_type == "qlearning":
            chosen = agent.select_arm(state)
        else:
            chosen = agent.select_arm()

        if chosen == true_arm:
            if agent_type == "linucb":
                agent.update(chosen, ctx, reward)
            elif agent_type == "qlearning":
                agent.update(state, chosen, reward)
            else:
                agent.update(chosen, ctx, reward)
            rewards.append(reward)
        else:
            rewards.append(np.nan)
        correct.append(int(chosen == true_arm))
    return np.array(rewards), np.array(correct)


# ---------------------------------------------------------------------------
# Orchestrator training
# ---------------------------------------------------------------------------

def train_orchestrator(interactions):
    """Train the orchestrator with GenreDiversityTool integration."""
    orch = OrchestratorAgent(seed=42)
    diversity_tool = GenreDiversityTool(diversity_threshold=0.80, window_size=200)

    rewards, correct, agent_log = [], [], []
    routing_over_time = []

    for t, ix in enumerate(interactions):
        true_arm = ix["arm"]
        reward   = ix["reward"]

        # Step 1: Orchestrator selects agent and arm
        chosen_arm, agent_name, reason = orch.select_and_update(ix)

        # Step 2: GenreDiversityTool post-processes the selection
        final_arm, intervened, div_reason = diversity_tool.execute(chosen_arm, t=t)

        # Track results
        matched = int(final_arm == true_arm)
        rewards.append(reward if matched else np.nan)
        correct.append(matched)
        agent_log.append(agent_name)

        # Track routing distribution every 1000 steps
        if t % 1000 == 0 and t > 0:
            dist = orch.get_routing_distribution()
            routing_over_time.append({"t": t, **dist})

    return orch, diversity_tool, np.array(rewards), np.array(correct), agent_log, routing_over_time


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def smooth(arr, w):
    kernel = np.ones(w) / w
    return np.convolve(arr.astype(float), kernel, mode="valid")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_orchestrator_comparison(results_dict):
    """Compare orchestrator against individual agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Agent Orchestrator vs. Individual Agents — Real MovieLens Data",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for name, data in results_dict.items():
        s = smooth(data["correct"], WINDOW)
        ax.plot(s * 100, label=name, color=COLORS.get(name, "gray"),
                linewidth=2.2 if name == "Orchestrator" else 1.3,
                linestyle="-" if name == "Orchestrator" else "--")
    ax.axhline(y=100/N_ARMS, color="gray", linestyle=":", alpha=0.5, label="Random (16.7%)")
    ax.set_xlabel("Real rating interaction", fontsize=11)
    ax.set_ylabel("Genre match rate % (rolling)", fontsize=11)
    ax.set_title(f"Genre Match Rate (window={WINDOW})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 75])

    ax = axes[1]
    for name, data in results_dict.items():
        valid = data["rewards"][~np.isnan(data["rewards"])]
        cum = np.cumsum(valid)
        ax.plot(cum, label=name, color=COLORS.get(name, "gray"),
                linewidth=2.2 if name == "Orchestrator" else 1.3,
                linestyle="-" if name == "Orchestrator" else "--")
    ax.set_xlabel("Matched interactions", fontsize=11)
    ax.set_ylabel("Cumulative real reward", fontsize=11)
    ax.set_title("Cumulative Reward from Real Ratings", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = OUTPUT_DIR / "fig_orchestrator_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def fig_routing_decisions(agent_log):
    """Show how the orchestrator distributed work between agents over time."""
    chunk = 2000
    n_chunks = len(agent_log) // chunk
    linucb_pct = []
    ql_pct = []
    pop_pct = []

    for i in range(n_chunks):
        seg = agent_log[i*chunk:(i+1)*chunk]
        linucb_pct.append(seg.count("LinUCB") / len(seg) * 100)
        ql_pct.append(seg.count("Q-Learning") / len(seg) * 100)
        pop_pct.append(seg.count("Popularity") / len(seg) * 100)

    x = np.arange(n_chunks) * chunk
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(x, 0, linucb_pct, alpha=0.75, color=COLORS["LinUCB"], label="LinUCB")
    ax.fill_between(x, linucb_pct,
                    [l+q for l,q in zip(linucb_pct, ql_pct)],
                    alpha=0.75, color=COLORS["Q-Learning"], label="Q-Learning")
    ax.fill_between(x,
                    [l+q for l,q in zip(linucb_pct, ql_pct)],
                    [l+q+p for l,q,p in zip(linucb_pct, ql_pct, pop_pct)],
                    alpha=0.75, color=COLORS["Popularity"], label="Popularity (fallback)")

    ax.set_xlabel("Real rating interaction", fontsize=11)
    ax.set_ylabel("% of routing decisions", fontsize=11)
    ax.set_title("Orchestrator Routing Decisions Over Training\n"
                 "Shift from LinUCB (cold states) → Q-Learning (warm states) as data accumulates",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p = OUTPUT_DIR / "fig_routing_decisions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def fig_memory_state_visits(orch: OrchestratorAgent):
    """Visualize shared memory — state visit distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Shared Memory — Orchestrator State Visits & Arm Selections",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    visits = orch.memory.state_visits
    bars = ax.bar(STATE_NAMES, visits, color=COLORS["LinUCB"], alpha=0.8, edgecolor="white")
    ax.axhline(y=SPARSE_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
               label=f"Sparse threshold ({SPARSE_THRESHOLD})")
    for bar, v in zip(bars, visits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels([s.replace("_", "\n") for s in STATE_NAMES], fontsize=9)
    ax.set_ylabel("Interactions in shared memory", fontsize=11)
    ax.set_title("State Visit Counts in Shared Memory\n(above threshold → Q-Learning routed)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    arm_counts = orch.memory.arm_counts
    bars2 = ax.bar(ARMS, arm_counts, color="#27AE60", alpha=0.8, edgecolor="white")
    for bar, v in zip(bars2, arm_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Selections recorded in memory", fontsize=11)
    ax.set_title("Genre Arm Counts in Shared Memory\n(diversity tool acts on this distribution)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p = OUTPUT_DIR / "fig_memory_state_visits.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def fig_diversity_tool(diversity_tool: GenreDiversityTool):
    """Plot entropy history from the diversity tool."""
    diversity_tool.plot_entropy_history(
        str(OUTPUT_DIR / "fig_diversity_entropy.png"), window=100
    )
    print(f"  Saved: {OUTPUT_DIR}/fig_diversity_entropy.png")


def fig_state_routing_profile(orch: OrchestratorAgent):
    """Heatmap showing which agent was used per user state."""
    profile = orch.get_state_routing_profile()
    agent_to_num = {"LinUCB": 0, "Q-Learning": 1, "Popularity": 2}
    nums = [agent_to_num.get(profile.get(s, "LinUCB"), 0) for s in STATE_NAMES]

    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.cm.get_cmap("Set1", 3)
    im = ax.imshow([nums], cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(N_STATES))
    ax.set_xticklabels([s.replace("_", "\n") for s in STATE_NAMES], fontsize=10)
    ax.set_yticks([])

    for i, (s, n) in enumerate(zip(STATE_NAMES, nums)):
        agent = ["LinUCB", "Q-Learning", "Popularity"][n]
        ax.text(i, 0, agent, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=cmap(0), label="LinUCB"),
              Patch(facecolor=cmap(1), label="Q-Learning"),
              Patch(facecolor=cmap(2), label="Popularity fallback")]
    ax.legend(handles=legend, loc="lower right", fontsize=9, bbox_to_anchor=(1.0, -0.3))
    ax.set_title("Orchestrator Routing Profile — Primary Agent per User State",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    p = OUTPUT_DIR / "fig_state_routing_profile.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("MovieLens RL — Multi-Agent Orchestrated Training")
    print("Real Data: MovieLens 100K (GroupLens, Univ. of Minnesota)")
    print("=" * 65)

    print("\nLoading real MovieLens data...")
    interactions = build_interactions()
    print(f"  Loaded {len(interactions):,} real interactions")

    # --- Train orchestrator ---
    print("\n[1/5] Training Multi-Agent Orchestrator + GenreDiversityTool...")
    orch, diversity_tool, orch_rewards, orch_correct, agent_log, routing_over_time = \
        train_orchestrator(interactions)

    # --- Train individual baselines for comparison ---
    print("[2/5] Training individual agents (baselines)...")

    linucb = LinUCBAgent(alpha=1.0)
    l_rewards, l_correct = train_single_agent(linucb, interactions, "linucb")
    print("  LinUCB done")

    ql = QLearningAgent()
    q_rewards, q_correct = train_single_agent(ql, interactions, "qlearning")
    print("  Q-Learning done")

    pop = PopularityAgent()
    p_rewards, p_correct = train_single_agent(pop, interactions, "other")
    print("  Popularity done")

    rand = RandomAgent(seed=42)
    r_rewards, r_correct = train_single_agent(rand, interactions, "other")
    print("  Random done")

    results_dict = {
        "Orchestrator": {"rewards": orch_rewards, "correct": orch_correct},
        "LinUCB":       {"rewards": l_rewards,    "correct": l_correct},
        "Q-Learning":   {"rewards": q_rewards,    "correct": q_correct},
        "Popularity":   {"rewards": p_rewards,    "correct": p_correct},
        "Random":       {"rewards": r_rewards,    "correct": r_correct},
    }

    # --- Print results ---
    print("\n" + "=" * 65)
    print("ORCHESTRATOR RESULTS")
    print("=" * 65)
    for name, data in results_dict.items():
        valid = data["rewards"][~np.isnan(data["rewards"])]
        mean_r = float(np.nanmean(valid)) * 4 + 1 if len(valid) > 0 else 0.0
        final_acc = smooth(data["correct"], WINDOW)[-3000:].mean() * 100
        print(f"  {name:15s} | Mean rating: {mean_r:.3f}/5 | Match rate: {final_acc:.1f}%")

    print("\nORCHESTRATOR ROUTING DISTRIBUTION:")
    dist = orch.get_routing_distribution()
    for agent, pct in dist.items():
        print(f"  {agent:15s}: {pct:.1f}% of decisions")

    print("\nSHARED MEMORY SUMMARY:")
    mem = orch.memory.summary()
    print(f"  Total interactions logged : {mem['total_interactions']:,}")
    print(f"  LinUCB mean reward        : {mem['linucb_mean_reward']:.3f}/5")
    print(f"  Q-Learning mean reward    : {mem['ql_mean_reward']:.3f}/5")
    print(f"  Genre entropy             : {mem['genre_entropy']:.4f} (max={1.7918:.4f})")

    print("\nGENREDIVERSITYTOOL STATS:")
    stats = diversity_tool.get_stats()
    print(f"  Total calls        : {stats['total_calls']:,}")
    print(f"  Interventions      : {stats['total_interventions']:,}")
    print(f"  Intervention rate  : {stats['intervention_rate']:.2f}%")
    print(f"  Current entropy    : {stats['current_entropy']:.4f}")
    print(f"  Mean entropy       : {stats['mean_entropy']:.4f}")

    print("\nROUTING PROFILE BY USER STATE:")
    profile = orch.get_state_routing_profile()
    for state, agent in profile.items():
        visits = orch.memory.state_visits[list(STATE_NAMES).index(state) if state in STATE_NAMES else 0]
        print(f"  {state:15s} → {agent}")

    # --- Generate figures ---
    print("\n[3/5] Generating figures...")
    fig_orchestrator_comparison(results_dict)
    fig_routing_decisions(agent_log)
    fig_memory_state_visits(orch)
    fig_diversity_tool(diversity_tool)
    fig_state_routing_profile(orch)

    print("\n✓ All outputs written to:", OUTPUT_DIR.resolve())
    print("\nFILES ADDED:")
    for f in ["fig_orchestrator_comparison.png", "fig_routing_decisions.png",
              "fig_memory_state_visits.png", "fig_diversity_entropy.png",
              "fig_state_routing_profile.png"]:
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()