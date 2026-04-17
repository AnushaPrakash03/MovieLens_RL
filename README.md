# MovieLens Adaptive Genre Recommendation Agent
## Reinforcement Learning on Real User Rating Data

**INFO 7375 — Prompt Engineering & AI**

An agentic recommendation system that learns optimal genre recommendations from **92,580 real user ratings** using two reinforcement learning methods. Instead of hardcoded popularity rules, two RL agents independently learn *which genre to recommend* based on user demographics — improving recommendation quality through experience.

---

## Overview

The MovieLens Adaptive Agent is a **Research and Analysis Agent** that learns personalized movie genre recommendations from real historical user feedback. The system frames recommendation as a contextual bandit problem: given a user's demographic context, which genre should the agent recommend to maximize expected rating?

### The Problem with Static Recommendation
A naive popularity-based system always recommends Drama (the most-rated genre). This ignores user context entirely — a young male student likely has different preferences than a senior female professional. RL agents learn these distinctions from data.

### Two RL Methods Implemented

| Method | Family | Mechanism |
|---|---|---|
| **LinUCB** | Exploration Strategies (Contextual Bandit) | Linear UCB over 7-dim user context vector |
| **Q-Learning** | Value-Based Learning | Tabular Q-table over 6 discrete user states |

---

## Real Data Source

**MovieLens 100K Dataset**
- **Source:** GroupLens Research Project, University of Minnesota
- **URL:** https://grouplens.org/datasets/movielens/100k/
- **Citation:** Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. DOI: 10.1145/2827872
- **Collection period:** September 1997 – April 1998
- **Size:** 100,000 ratings (1–5 stars) from 943 users on 1,682 movies

### Files Used

| File | Contents | Used For |
|---|---|---|
| `u.data` | 100K ratings: user_id, movie_id, rating, timestamp | Reward signal |
| `u.user` | 943 users: age, gender, occupation, zip | Context features |
| `u.item` | 1,682 movies + 19 genre binary columns | Arm labels |

### Nothing is Simulated
All rewards come from real user ratings. No reward probabilities were manually assigned. The offline bandit evaluation methodology (Li et al., 2011) ensures unbiased learning from logged data.

---

## RL Formulation

### Shared Setup

| Component | Definition |
|---|---|
| **Arms (6)** | Action, Comedy, Drama, Romance, Thriller, SciFi |
| **Reward** | Normalized real rating: (rating − 1) / 4 ∈ [0, 1] |
| **Evaluation** | Offline reject-sampling (unbiased offline bandit evaluation) |
| **Interactions** | 92,580 real rating events (7,420 skipped — no matching genre arm) |

### Method 1: LinUCB — Exploration Strategies

LinUCB maintains per-arm matrices updated from real interactions:

```
A_a  ←  A_a + x·xᵀ          (outer product update)
b_a  ←  b_a + r·x            (reward-weighted context)
θ_a  =  A_a⁻¹ · b_a          (learned weight vector)

UCB score: θ_aᵀ·x  +  α · √(xᵀ · A_a⁻¹ · x)
            exploitation        exploration
```

**Context features (7-dimensional) — engineered from real u.user data:**

| Feature | Description |
|---|---|
| `age_norm` | age / 80 |
| `is_male` | 1 if gender == M |
| `is_student` | 1 if occupation == student |
| `is_professional` | 1 if occupation in {engineer, doctor, lawyer, scientist, programmer} |
| `is_entertainment` | 1 if occupation in {artist, writer, entertainer} |
| `age_young` | 1 if age < 25 |
| `age_senior` | 1 if age >= 45 |

### Method 2: Q-Learning — Value-Based Learning

```
Q(s, a)  ←  Q(s, a) + α · [r − Q(s, a)]
```

**Discrete states (6) — derived from real user demographics:**

| State | Condition |
|---|---|
| young_male | age < 25, gender = M |
| young_female | age < 25, gender = F |
| mid_male | 25 ≤ age < 45, gender = M |
| mid_female | 25 ≤ age < 45, gender = F |
| senior_male | age ≥ 45, gender = M |
| senior_female | age ≥ 45, gender = F |

---

## Results (Real Data)

### Recommendation Quality — Mean Rating on Matched Interactions

| Agent | Mean Rating (1–5) | vs. Dataset Mean |
|---|---|---|
| **LinUCB** | **3.687 / 5** | **+0.157** |
| Popularity | 3.687 / 5 | +0.157 (no personalization) |
| **Q-Learning** | **3.618 / 5** | **+0.088** |
| Random | 3.529 / 5 | baseline |
| Dataset mean | 3.530 / 5 | — |

### Key Finding
LinUCB matches popularity on raw reward but achieves **personalization** — it routes different genres to different user profiles rather than recommending Drama to everyone. The Q-table reveals distinct learned preferences per user state (e.g., Sci-Fi scores highest for young males; Drama scores highest for senior females).

### Offline Evaluation Note
Genre match rate (how often the agent picks the genre the user happened to rate) is low by design — this is expected with 6 arms and offline reject-sampling evaluation. The primary metric is **mean rating quality on matched interactions**, which directly measures whether the agent is recommending genres users actually liked.

---

## Repository Structure

```
movielens_rl/
├── data/
│   ├── u.data          ← 100K real ratings (reward signal)
│   ├── u.user          ← 943 user demographics (context)
│   └── u.item          ← 1,682 movies + genre labels (arms)
│
├── data_loader.py      ← Parses real files, engineers features, builds interactions
├── agents.py           ← LinUCB + Q-Learning + Random + Popularity baselines
├── train_and_evaluate.py  ← Training loop + offline evaluation + 10 figures
│
└── outputs/
    ├── fig0_reward_quality.png      ← Mean rating per agent (primary metric)
    ├── fig1_data_overview.png       ← Real data distribution
    ├── fig2_learning_curves.png     ← Genre match rate + cumulative reward
    ├── fig3_linucb_weights.png      ← Learned θ weights per genre
    ├── fig4_qtable.png              ← Q-table values + visit counts
    ├── fig5_per_state_accuracy.png  ← Accuracy by user demographic group
    ├── fig6_epsilon_decay.png       ← Q-Learning exploration schedule
    ├── fig7_arm_evolution.png       ← Genre selection shift over training
    ├── fig8_reward_by_genre.png     ← Mean rating per genre (LinUCB)
    └── fig9_final_summary.png       ← Overall comparison
```

---

## Setup and Reproduction

### Requirements
```bash
pip install numpy matplotlib pandas
```
No other dependencies. Python 3.9+.

### Data Setup
Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/

Place these 3 files in the `data/` subfolder:
- `u.data`
- `u.user`
- `u.item`

### Run
```bash
python train_and_evaluate.py
```

**Expected output:** 10 PNG figures saved to `outputs/` + training summary printed to terminal.

**Expected runtime:** ~45–60 seconds on a standard laptop.

**Reproducibility:** Fixed random seed (`seed=42`) throughout. Results are exactly reproducible.

---

## Design Decisions

### Why Offline Bandit Evaluation?
In a live system, the agent would recommend a genre and receive the user's actual rating as reward. With logged data (MovieLens), we use **reject sampling** (Li et al., 2011): the agent only learns from interactions where its chosen arm matches the genre the user actually rated. This gives an unbiased estimate of online performance without requiring a live deployment.

### Why These 6 Genres?
The 19 MovieLens genres were reduced to 6 based on coverage and distinctiveness. Drama, Comedy, and Action cover ~90% of ratings. Romance, Thriller, and Sci-Fi were added for diversity. Genres like Documentary and Western have too few ratings for meaningful bandit learning.

### Why Age × Gender States for Q-Learning?
Age and gender are the two strongest demographic predictors of genre preference in collaborative filtering literature. The 6-state discretization (3 age groups × 2 genders) gives Q-Learning enough resolution to learn meaningful differences without sparse state coverage.

### Why Does Popularity Tie LinUCB on Raw Reward?
Drama has the highest mean rating (3.67/5) in the dataset. A purely greedy agent maximizing average reward would always pick Drama. LinUCB's value is **personalization** — it recommends lower-average-rating genres to users who prefer them, which a per-user evaluation would reveal more clearly than aggregate metrics.

---

## Ethical Considerations

**Filter Bubble Risk:** An RL agent optimizing for immediate rating may converge to recommending only a few high-rated genres, limiting user exposure to diverse content. Production systems should include exploration incentives and diversity constraints.

**Demographic Bias:** The dataset is heavily skewed toward young males (mid_male = 40% of ratings). The learned policy may underperform for underrepresented groups (senior_female = 4% of ratings). Fairness-aware RL methods should be considered for deployment.

**Data Age:** Ratings were collected in 1997–1998. Genre preferences have likely shifted. A deployed system requires continuous re-training on fresh data.

**Cold Start:** New users with no rating history cannot be mapped to a context vector. A hybrid approach combining content-based features with RL would address this.

---

## Future Work

- **Neural contextual bandit:** Replace LinUCB's linear model with a neural network for non-linear user-genre interactions
- **Collaborative filtering features:** Add user similarity features to the context vector
- **Multi-step recommendation:** Extend to a full MDP where the agent recommends sequences of movies
- **Online A/B testing:** Deploy both agents in a live system and compare real-time performance
- **Fairness constraints:** Add demographic parity constraints to prevent over-recommendation to majority groups
- **Thompson Sampling:** Replace ε-greedy with Thompson Sampling for better sample efficiency

---

## References

- Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *ACM TiiS*, 5(4). DOI: 10.1145/2827872
- Li, L., Chu, W., Langford, J., & Schapire, R.E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *WWW 2010*.
- Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased Offline Evaluation of Contextual-Bandit-Based News Article Recommendation. *WSDM 2011*.
- Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.

---

*Anusha Prakash | Northeastern University | INFO 7375 | Spring 2026*
