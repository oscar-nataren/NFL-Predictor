import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import xgboost
import sklearn
import warnings
warnings.filterwarnings('ignore')
import nflreadpy as nfl
#python 3.10 or above is required for nflreadpy to work. If you are using an older version of python, please upgrade to 3.10 or above.
print(f"Loading NFL data...\n")

#Team strength

# Elo rating — not in NFLverse, build it yourself (same calculate_elo() approach as World-Cup, different K-factor)
# Point differential (season + rolling) — compute from NFLverse game scores
# Recent form (last 3-5 games W/L) — compute from NFLverse schedule/results

# Efficiency

# EPA/play (off & def) — already in NFLverse play-by-play, just aggregate per team per game
# Turnover differential — already in NFLverse team stats, just diff it
# 3rd down % — already in NFLverse team stats
# Red zone % — already in NFLverse team stats, may need to compute from PBP if not pre-aggregated

# Situational

# Home field flag — trivial, from schedule data
# Rest days / short week / bye — computable from NFLverse schedule (date diffs)
# Divisional game flag — computable from team/division mapping
# Travel distance — not in NFLverse, you build it (stadium lat/long lookup + haversine distance)

# Injuries

# QB out/questionable flag + starter-vs-backup talent gap — NFLverse has injury reports, but you build the talent-gap calc yourself (diff in career EPA/passer rating between starter and backup)
# Aggregate injury severity (rest of roster) — NFLverse has injury data, you build the weighting/aggregation logic

# Head-to-head

# Recent matchup history — computable from NFLverse historical results
# Weather (wind especially) — NFLverse has some weather data, may need to supplement for missing games

pbp = nfl.load_pbp([2020, 2021, 2022, 2023, 2024, 2025]).to_pandas()
sched = nfl.load_schedules([2020, 2021, 2022, 2023, 2024, 2025]).to_pandas()

# -------------------- EPA --------------------

off_epa = pbp.groupby(["posteam", "game_id"]).agg(
    off_avg_epa=("epa", "mean"),
    off_total_yards=("yards_gained", "sum")
).reset_index()

def_epa = pbp.groupby(["defteam", "game_id"]).agg(
    def_avg_epa=("epa", "mean"),
    def_total_yards=("yards_gained", "sum")
).reset_index()

off_epa = off_epa.merge(sched[["game_id", "gameday"]], on="game_id", how="left")
def_epa = def_epa.merge(sched[["game_id", "gameday"]], on="game_id", how="left")

off_epa = off_epa.sort_values(["posteam", "gameday"]).reset_index(drop=True)
def_epa = def_epa.sort_values(["defteam", "gameday"]).reset_index(drop=True)

off_epa["off_epa_prev"] = off_epa.groupby("posteam")["off_avg_epa"].shift(1)
def_epa["def_epa_prev"] = def_epa.groupby("defteam")["def_avg_epa"].shift(1)


def add_rolling(df, group_col, value_col, window):
    col_name = f"{value_col}_roll{window}"
    df[col_name] = (df.groupby(group_col)[value_col].shift(1).rolling(window, min_periods=1).mean())
    return df

off_epa = add_rolling(off_epa, "posteam", "off_avg_epa", 3)
off_epa = add_rolling(off_epa, "posteam", "off_avg_epa", 8)

def_epa = add_rolling(def_epa, "defteam", "def_avg_epa", 3)
def_epa = add_rolling(def_epa, "defteam", "def_avg_epa", 8)

print(off_epa.head(15))
print(def_epa.head(15))

# -------------------- 3rd down % --------------------
