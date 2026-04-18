"""
MovieLens 100K — Data Loader
=============================
Parses the three real MovieLens files and produces:
  - User context vectors (for LinUCB)
  - Discretized user states (for Q-Learning)
  - Rating interactions (reward signal)
  - Genre arm mappings

DATA SOURCE
-----------
GroupLens Research Project, University of Minnesota.
MovieLens 100K Dataset: 100,000 ratings (1-5) from 943 users on 1,682 movies.
Collected Sep 1997 – Apr 1998 via the MovieLens web site.
Citation: Harper & Konstan (2015), ACM TiiS 5(4), DOI:10.1145/2827872

FILES USED
----------
u.data  : user_id | movie_id | rating | timestamp  (tab-separated)
u.user  : user_id | age | gender | occupation | zip  (pipe-separated)
u.item  : movie_id | title | ... | 19 genre binary columns  (pipe-separated)

GENRE ARMS (6 selected — highest coverage in dataset)
------------------------------------------------------
0: Action      1: Comedy      2: Drama
3: Romance     4: Thriller    5: Sci-Fi

These 6 genres cover the majority of ratings and are distinct enough
for the bandit to learn meaningful preferences.

CONTEXT FEATURES (7-dimensional, all normalized to [0,1])
----------------------------------------------------------
0: age_norm          — age / 80
1: is_male           — 1 if gender == M
2: is_student        — 1 if occupation == student
3: is_professional   — 1 if occupation in {engineer, doctor, lawyer, scientist}
4: is_entertainment  — 1 if occupation in {artist, writer, musician, entertainer}
5: age_bucket_young  — 1 if age < 25
6: age_bucket_senior — 1 if age >= 45
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Genre column indices in u.item (after skipping first 5 non-genre columns)
# Full genre list from README:
# unknown|Action|Adventure|Animation|Children|Comedy|Crime|Documentary|
# Drama|Fantasy|FilmNoir|Horror|Musical|Mystery|Romance|SciFi|Thriller|War|Western
GENRE_NAMES_ALL = [
    "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
    "Documentary","Drama","Fantasy","FilmNoir","Horror","Musical","Mystery",
    "Romance","SciFi","Thriller","War","Western"
]

# Selected 6 arms — best coverage + diversity
ARMS = ["Action", "Comedy", "Drama", "Romance", "Thriller", "SciFi"]
N_ARMS = len(ARMS)
ARM_INDICES = {g: GENRE_NAMES_ALL.index(g) for g in ARMS}

OCCUPATIONS = [
    "administrator","artist","doctor","educator","engineer","entertainment",
    "executive","healthcare","homemaker","lawyer","librarian","marketing",
    "none","other","programmer","retired","salesman","scientist","student",
    "technician","writer"
]

# Discrete user states for Q-Learning (age group × gender = 6 states)
STATE_NAMES = [
    "young_male",    # age < 25, M
    "young_female",  # age < 25, F
    "mid_male",      # 25 <= age < 45, M
    "mid_female",    # 25 <= age < 45, F
    "senior_male",   # age >= 45, M
    "senior_female", # age >= 45, F
]
N_STATES = len(STATE_NAMES)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_users(path=None):
    """Load u.user → DataFrame with engineered features."""
    path = path or DATA_DIR / "u.user"
    users = pd.read_csv(path, sep="|", header=None,
                        names=["user_id","age","gender","occupation","zip"])
    users = users.set_index("user_id")
    return users


def load_items(path=None):
    """Load u.item → DataFrame with genre binary columns."""
    path = path or DATA_DIR / "u.item"
    cols = ["movie_id","title","release_date","video_release","imdb_url"] + GENRE_NAMES_ALL
    items = pd.read_csv(path, sep="|", header=None, names=cols,
                        encoding="latin-1")
    items = items.set_index("movie_id")
    return items


def load_ratings(path=None):
    """Load u.data → DataFrame of 100K real ratings."""
    path = path or DATA_DIR / "u.data"
    ratings = pd.read_csv(path, sep="\t", header=None,
                          names=["user_id","movie_id","rating","timestamp"])
    # Normalize rating to [0, 1]
    ratings["reward"] = (ratings["rating"] - 1) / 4.0
    return ratings


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def user_to_context(user_row):
    """
    Convert a user row to a 7-dim context vector for LinUCB.
    All values are in [0, 1].
    """
    age = user_row["age"]
    gender = user_row["gender"]
    occ = user_row["occupation"].lower() if isinstance(user_row["occupation"], str) else ""

    is_professional = int(occ in {"engineer","doctor","lawyer","scientist","programmer"})
    is_entertainment = int(occ in {"artist","writer","entertainer","musician"})
    is_student = int(occ == "student")

    ctx = np.array([
        min(age, 80) / 80.0,          # age_norm
        1.0 if gender == "M" else 0.0, # is_male
        float(is_student),             # is_student
        float(is_professional),        # is_professional
        float(is_entertainment),       # is_entertainment
        1.0 if age < 25 else 0.0,      # age_bucket_young
        1.0 if age >= 45 else 0.0,     # age_bucket_senior
    ], dtype=np.float64)
    return ctx


def user_to_state(user_row):
    """
    Map a user row to one of 6 discrete states for Q-Learning.
    State = age_group (3) × gender (2)
    """
    age = user_row["age"]
    gender = user_row["gender"]

    if age < 25:
        age_grp = 0
    elif age < 45:
        age_grp = 1
    else:
        age_grp = 2

    gender_idx = 0 if gender == "M" else 1
    return age_grp * 2 + gender_idx


def get_movie_genre_arm(item_row):
    """
    Return the arm index (0–5) of the primary genre of a movie,
    or None if the movie doesn't belong to any of the 6 selected genres.
    Priority order: Drama > Comedy > Action > Romance > Thriller > SciFi
    """
    priority = ["Drama","Comedy","Action","Romance","Thriller","SciFi"]
    for genre in priority:
        if item_row[genre] == 1:
            return ARMS.index(genre)
    return None


# ---------------------------------------------------------------------------
# Build interaction dataset
# ---------------------------------------------------------------------------

def build_interactions(ratings=None, users=None, items=None):
    """
    Join ratings + user features + movie genre to produce:
    a list of (context, state, arm, reward) tuples — the real RL dataset.

    Only includes ratings where the movie belongs to one of the 6 genre arms.
    Returns a list of dicts, shuffled randomly (seed=42).
    """
    if ratings is None: ratings = load_ratings()
    if users is None:   users   = load_users()
    if items is None:   items   = load_items()

    interactions = []
    skipped = 0

    for _, row in ratings.iterrows():
        uid = row["user_id"]
        mid = row["movie_id"]

        if uid not in users.index or mid not in items.index:
            skipped += 1
            continue

        user_row = users.loc[uid]
        item_row = items.loc[mid]

        arm = get_movie_genre_arm(item_row)
        if arm is None:
            skipped += 1
            continue

        context = user_to_context(user_row)
        state   = user_to_state(user_row)
        reward  = float(row["reward"])

        interactions.append({
            "user_id": uid,
            "movie_id": mid,
            "arm": arm,
            "reward": reward,
            "context": context,
            "state": state,
            "rating": int(row["rating"]),
        })

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    rng.shuffle(interactions)

    print(f"  Total interactions: {len(interactions):,} "
          f"(skipped {skipped:,} — no matching genre arm)")
    return interactions


# ---------------------------------------------------------------------------
# Dataset stats
# ---------------------------------------------------------------------------

def print_stats(interactions):
    """Print key statistics about the loaded dataset."""
    n = len(interactions)
    rewards = [x["reward"] for x in interactions]
    arms    = [x["arm"] for x in interactions]
    states  = [x["state"] for x in interactions]

    print(f"\n{'='*55}")
    print(f"MovieLens 100K — Dataset Statistics")
    print(f"{'='*55}")
    print(f"  Total interactions : {n:,}")
    print(f"  Mean reward (norm) : {np.mean(rewards):.3f}  (= rating {np.mean(rewards)*4+1:.2f}/5)")
    print(f"\n  Genre arm distribution:")
    for a, name in enumerate(ARMS):
        count = arms.count(a)
        mean_r = np.mean([x["reward"] for x in interactions if x["arm"] == a])
        print(f"    {name:12s}: {count:6,} ratings  |  mean reward: {mean_r:.3f}")
    print(f"\n  User state distribution:")
    for s, name in enumerate(STATE_NAMES):
        count = states.count(s)
        mean_r = np.mean([x["reward"] for x in interactions if x["state"] == s])
        print(f"    {name:15s}: {count:6,} ratings  |  mean reward: {mean_r:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    print("Loading MovieLens 100K...")
    interactions = build_interactions()
    print_stats(interactions)
