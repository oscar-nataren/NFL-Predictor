import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import sklearn
import os
import pathlib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import nflreadpy as nfl
#python 3.10 or above is required for nflreadpy to work. If you are using an older version of python, please upgrade to 3.10 or above.
print(f"Loading NFL data...\n")


first_season = 2020
last_season = 2025  
current_season = 2026
seasons = list(range(first_season, current_season + 1))

pbp = nfl.load_pbp(seasons).to_pandas()
team_stats = nfl.load_team_stats(seasons=seasons, summary_level="week").to_pandas()
player_stats = nfl.load_player_stats(seasons=seasons).to_pandas()
depth_charts = nfl.load_depth_charts(seasons=seasons).to_pandas()
latest_roster = nfl.load_rosters([current_season]).to_pandas()

sched_seasons = list(range(first_season, current_season + 1))   # schedule can safely include 2026
sched = nfl.load_schedules(sched_seasons).to_pandas()
# print(sched.columns.tolist())
# print(team_stats.columns.tolist())
# print(pbp.columns.tolist())

# -------------------- EPA --------------------

off_epa = pbp.groupby(["posteam", "game_id"]).agg(
    off_avg_epa=("epa", "mean"),
    off_total_yards=("yards_gained", "sum")
).reset_index()

def_epa = pbp.groupby(["defteam", "game_id"]).agg(
    def_avg_epa=("epa", "mean"),
    def_total_yards=("yards_gained", "sum")
).reset_index()

def add_gameday(df, group_col):
    df = df.merge(sched[["gameday", "gametime", "game_id"]], on="game_id", how="left")
    return df.sort_values([ "gameday", "gametime", group_col]).reset_index(drop=True)

off_epa = add_gameday(off_epa, "posteam")
def_epa = add_gameday(def_epa, "defteam")

off_epa["off_epa_prev"] = off_epa.groupby("posteam")["off_avg_epa"].shift(1)
def_epa["def_epa_prev"] = def_epa.groupby("defteam")["def_avg_epa"].shift(1)

rolling_short = 3
rolling_long = 8

def add_rolling(df, group_col, value_col, window):
    col_name = f"{value_col}_r{window}"
    df[col_name] = (df.groupby(group_col)[value_col].shift(1).rolling(window, min_periods=1).mean())
    return df


off_epa = add_rolling(off_epa, "posteam", "off_avg_epa", rolling_short)
off_epa = add_rolling(off_epa, "posteam", "off_avg_epa", rolling_long)

def_epa = add_rolling(def_epa, "defteam", "def_avg_epa", rolling_short)
def_epa = add_rolling(def_epa, "defteam", "def_avg_epa", rolling_long)

# print(off_epa.head(15))
# print(def_epa.head(15))

# -------------------- 3rd down % --------------------
third_down_plays = pbp[(pbp["down"] == 3) & (pbp["play_type"] != "no_play")]

third_down_off = third_down_plays.groupby(["posteam", "game_id"]).agg(
    att_3rd=("down", "count"),
    conv_3rd=("third_down_converted", "sum")
).reset_index()

third_down_off["off_3rd_down_pct"] = (
    third_down_off["conv_3rd"] / third_down_off["att_3rd"]
)

third_down_def = third_down_plays.groupby(["defteam", "game_id"]).agg(
    att_3rd_faced=("down", "count"),
    conv_3rd_allowed=("third_down_converted", "sum")
).reset_index()

third_down_def["def_3rd_down_pct_allowed"] = (
    third_down_def["conv_3rd_allowed"] / third_down_def["att_3rd_faced"]
)

third_down_off = add_gameday(third_down_off, "posteam")
third_down_off = add_rolling(third_down_off, "posteam", "off_3rd_down_pct", rolling_short)
third_down_off = add_rolling(third_down_off, "posteam", "off_3rd_down_pct", rolling_long)

third_down_def = add_gameday(third_down_def, "defteam")
third_down_def = add_rolling(third_down_def, "defteam", "def_3rd_down_pct_allowed", rolling_short)
third_down_def = add_rolling(third_down_def, "defteam", "def_3rd_down_pct_allowed", rolling_long)

# print(third_down_off.head(15))
# print(third_down_def.head(15))

# -------------------- Turnover Differential --------------------

turnovers = team_stats[["team", "game_id", "passing_interceptions", "fumbles_lost_total",
                         "def_interceptions", "fumble_recovery_opp"]].copy()

turnovers["giveaways"] = turnovers["passing_interceptions"] + turnovers["fumbles_lost_total"]
turnovers["takeaways"] = turnovers["def_interceptions"] + turnovers["fumble_recovery_opp"]
turnovers["turnover_diff"] = turnovers["takeaways"] - turnovers["giveaways"]


turnovers = add_gameday(turnovers, "team")
turnovers = add_rolling(turnovers, "team", "turnover_diff", rolling_short)
turnovers = add_rolling(turnovers, "team", "turnover_diff", rolling_long)

# print(turnovers.head(15))

# -------------------- Red Zone % --------------------
redzone_plays = pbp[pbp["yardline_100"] <= 20].copy()
redzone_plays["offense_td"] = (redzone_plays["touchdown"] == 1) & (redzone_plays["td_team"] == redzone_plays["posteam"])

redzone_off = redzone_plays.groupby(["posteam", "game_id"]).agg(
    redzone_trips=("drive", "nunique"),
    redzone_tds=("offense_td", "sum")
).reset_index()

redzone_off["off_redzone_pct"] = redzone_off["redzone_tds"] / redzone_off["redzone_trips"]

redzone_def = redzone_plays.groupby(["defteam", "game_id"]).agg(
    redzone_trips_faced=("drive", "nunique"),
    redzone_tds_allowed=("offense_td", "sum")
).reset_index()

redzone_def["def_redzone_pct_allowed"] = redzone_def["redzone_tds_allowed"] / redzone_def["redzone_trips_faced"]

redzone_off = add_gameday(redzone_off, "posteam")
redzone_off = add_rolling(redzone_off, "posteam", "off_redzone_pct", rolling_short)
redzone_off = add_rolling(redzone_off, "posteam", "off_redzone_pct", rolling_long)

redzone_def = add_gameday(redzone_def, "defteam")
redzone_def = add_rolling(redzone_def, "defteam", "def_redzone_pct_allowed", rolling_short)
redzone_def = add_rolling(redzone_def, "defteam", "def_redzone_pct_allowed", rolling_long)

# print(redzone_off.head(15))
# print(redzone_def.head(15))

# -------------------- Situational --------------------
situational = sched[["game_id", "season", "game_type", "home_team", "home_score", "away_team", "away_score", "home_rest", "away_rest", "div_game", "gameday", "overtime"]].copy()
weather = sched[["game_id", "home_team", "away_team", "gameday", "roof", "surface", "temp", "wind", "location", "stadium"]].copy()

# print("Missing temp:", weather["temp"].isna().sum(), "out of", len(weather))
# print("Missing wind:", weather["wind"].isna().sum(), "out of", len(weather))
# print(weather["roof"].value_counts())

weather["temp"] = weather["temp"].fillna(70)
weather["wind"] = weather["wind"].fillna(0)

# print(weather[weather["roof"] == "open"][["temp", "wind"]].isna().sum())

# -------------------- Starter (Injuries) --------------------
# print(injuries.columns.tolist())
# print(player_stats.columns.tolist())

injured = ["Out", "Doubtful"]

qb_stats = player_stats[player_stats["position"] == "QB"][
    ["player_id", "player_name", "player_display_name", "team", "game_id", "passing_epa"]
].copy()

qb_stats = add_gameday(qb_stats, "player_id")
qb_stats["career_qb_epa"] = (
    qb_stats.groupby("player_id")["passing_epa"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

# print(latest_roster.columns.tolist())
roster_qbs = latest_roster[latest_roster["position"] == "QB"]

known_ids = set(qb_stats["player_id"].unique())
new_qb_rows = []
for row in roster_qbs.itertuples(index=False):
    pid = row.gsis_id
    if pd.notna(pid) and pid not in known_ids:
        new_qb_rows.append({
            "player_id": pid,
            # "player_name": row.football_name,
            "player_display_name": row.full_name,
            "team": row.team,
            "game_id": None,
            "passing_epa": np.nan,
            "gameday": pd.NaT,
            "gametime": None,
            "career_qb_epa": 0.0,
        })
        known_ids.add(pid)  # prevent duplicate rows if a QB appears on multiple roster rows

if new_qb_rows:
    qb_stats = pd.concat([qb_stats, pd.DataFrame(new_qb_rows)], ignore_index=True)


# print(qb_stats.head(15))
# print(rosters_2026[rosters_2026["position"] == "QB"][["team", "full_name", "depth_chart_position", "status"]].head(100))

# -------------------- Betting Odds --------------------

odds = sched[[
    "game_id", "spread_line", "home_spread_odds", "away_spread_odds",
    "total_line", "over_odds", "under_odds",
    "home_moneyline", "away_moneyline"
]].copy()

# -------------------- Elo Ratings --------------------
nfl_history = list(range(1999, current_season +1))
elo_schedule = nfl.load_schedules(nfl_history).to_pandas()


starting_elo, expansion_elo = 1500.0, 1300.0
expansion_teams = {"CLE", "HOU"}
k, home_advantage, regression = 20.0, 48.0, 1 / 3

def mov_multiplier(margin, elo_diff_winner):
    #Bigger wins count more, with diminishing returns; damped for favourites
    return np.log(max(abs(margin), 1.0) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))

def calculate_elo(schedules):
    elo_ratings = {}
    elo_data = []
    played = schedules.dropna(subset=["result"])
    non_neutral = played[played.get("location", "Home") != "Neutral"]
    margin_by_season = non_neutral.groupby("season").result.mean()

    def hfa_for(season):
        # Only seasons strictly BEFORE this one — the leak-free way to adapt.
        prior = margin_by_season[(margin_by_season.index >= season - 3)
                                 & (margin_by_season.index < season)]
        if len(prior) < 3:
            return home_advantage
        return float(np.clip(25.0 * prior.mean(), 15.0, 70.0))

    games = schedules.sort_values(["gameday", "gametime", "game_id"])

    current_season = None
    for game in games.itertuples(index=False):
        if current_season == None:
            current_season = game.season
        elif game.season != current_season:
            # Regression to the mean at the start of a new season
            for team in elo_ratings:
                elo_ratings[team] += (starting_elo - elo_ratings[team]) * regression
            current_season = game.season

        current_advantage = hfa_for(game.season)
        for team in (game.home_team, game.away_team):
            if team not in elo_ratings:
                elo_ratings[team] = expansion_elo if team in expansion_teams else starting_elo

        home_elo = elo_ratings[game.home_team]
        away_elo = elo_ratings[game.away_team]
        elo_data.append({
            "game_id": game.game_id,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
        })
        if pd.isna(game.result):
            continue  

        if game.location == "Neutral":
            advantage = 0.0
        else:
            advantage = current_advantage

        expected = 1 / (1 + 10 ** ((away_elo - home_elo - advantage) / 400))

        if game.result > 0:
            actual, diff = 1.0, (home_elo + advantage) - away_elo
        elif game.result < 0:
            actual, diff = 0.0, (away_elo) - (home_elo + advantage)
        else:
            actual, diff = 0.5, 0.0

        shift = k * mov_multiplier(game.result, diff) * (actual - expected)
        elo_ratings[game.home_team] += shift
        elo_ratings[game.away_team] -= shift

    return pd.DataFrame(elo_data), elo_ratings

elo_df, elo_ratings = calculate_elo(elo_schedule)



# -------------------- Merge --------------------
data_set = sched[["game_id", "gameday", "gametime", "season", "home_team", "away_team", "home_score", "away_score", "result"]].copy()

def merge_features(data_set, feature_df, team_col, feature_cols):
    for col in feature_cols:
        home_col = f"{col}_home"
        away_col = f"{col}_away"

        data_set = data_set.merge(
            feature_df[["game_id", team_col, col]],
            left_on=["game_id", "home_team"],
            right_on=["game_id", team_col],
            how="left"
        ).rename(columns={col: home_col}).drop(columns=[team_col])

        data_set = data_set.merge(
            feature_df[["game_id", team_col, col]],
            left_on=["game_id", "away_team"],
            right_on=["game_id", team_col],
            how="left"
        ).rename(columns={col: away_col}).drop(columns=[team_col])

    return data_set

data_set = data_set.merge(
    situational[["game_id", "game_type", "div_game", "overtime", "home_rest", "away_rest"]],
    on="game_id", how="left"
)

data_set = data_set.merge(
    weather[["game_id", "roof", "surface", "temp", "wind", "location", "stadium"]],
    on="game_id", how="left"
)

data_set = data_set.merge(
    odds, on="game_id", how="left"
)

data_set = data_set.merge(sched[["game_id", "home_qb_name", "away_qb_name"]], on="game_id", how="left")

data_set = data_set.merge(
    qb_stats[["player_display_name", "game_id", "career_qb_epa"]],
    left_on=["game_id", "home_qb_name"], right_on=["game_id", "player_display_name"],
    how="left"
).rename(columns={"career_qb_epa": "home_qb_epa"}).drop(columns="player_display_name")

data_set = data_set.merge(
    qb_stats[["player_display_name", "game_id", "career_qb_epa"]],
    left_on=["game_id", "away_qb_name"], right_on=["game_id", "player_display_name"],
    how="left"
).rename(columns={"career_qb_epa": "away_qb_epa"}).drop(columns="player_display_name")

data_set = merge_features(data_set, off_epa, "posteam", ["off_epa_prev", "off_avg_epa_r3", "off_avg_epa_r8"])
data_set = merge_features(data_set, def_epa, "defteam", ["def_epa_prev", "def_avg_epa_r3", "def_avg_epa_r8"])
data_set = merge_features(data_set, third_down_off, "posteam", ["off_3rd_down_pct_r3", "off_3rd_down_pct_r8"])
data_set = merge_features(data_set, third_down_def, "defteam", ["def_3rd_down_pct_allowed_r3", "def_3rd_down_pct_allowed_r8"])
data_set = merge_features(data_set, turnovers, "team", ["turnover_diff_r3", "turnover_diff_r8"])
data_set = merge_features(data_set, redzone_off, "posteam", ["off_redzone_pct_r3", "off_redzone_pct_r8"])
data_set = merge_features(data_set, redzone_def, "defteam", ["def_redzone_pct_allowed_r3", "def_redzone_pct_allowed_r8"])
data_set = data_set.merge(elo_df[["game_id", "home_elo_pre", "away_elo_pre"]], on="game_id", how="left")


# print(data_set["home_qb_epa"].isna().sum(), "unmatched home QBs out of", len(data_set))
# print(data_set["away_qb_epa"].isna().sum(), "unmatched away QBs out of", len(data_set))

# print(sched["home_qb_name"].head(15))

data_set["winner"] = np.where(data_set["result"] > 0, 1, np.where(data_set["result"] < 0, 0, 0.5))

# -------------------- Fill NaN --------------------

data_set = data_set.sort_values("gameday").reset_index(drop=True)

pct_cols = [c for c in data_set.columns if "pct" in c.lower()]
for col in pct_cols:
    expanding_mean = data_set[col].expanding().mean()
    data_set[col] = data_set[col].fillna(expanding_mean)

epa_cols = [c for c in data_set.columns if "epa" in c.lower()]
data_set[epa_cols] = data_set[epa_cols].fillna(0.0)

turnover_cols = [c for c in data_set.columns if "turnover_diff" in c]
data_set[turnover_cols] = data_set[turnover_cols].fillna(0.0)

# print(data_set.isna().sum().sum())  # should be 0, or close to it    
# remaining = data_set.isna().sum()
# print(remaining[remaining > 0])

# print(data_set.columns.tolist())
# print(data_set.head(15))

# week1_2020_check = data_set[data_set["off_3rd_down_pct_r3_home"].isna()][["game_id", "gameday", "season"]]
# print(week1_2020_check)



# -------------------- Models --------------------

model_data = data_set[(data_set["season"] > first_season) & (data_set["result"].notna())].copy()
model_data = model_data[model_data["winner"] != 0.5].copy()
model_data["winner"] = model_data["winner"].astype(int)

categorical_cols = ["game_type", "roof", "surface"]
# model_data["surface"] = model_data["surface"].str.strip()
# model_data = pd.get_dummies(model_data, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in model_data.columns if c not in 
                ["game_id", "season", "week", "home_team", "away_team", "location", "stadium",
                 "home_qb_name", "away_qb_name",
                 "home_score", "away_score", "result", "winner", "gameday", "gametime"
                 , "overtime", "game_type", "roof", "surface"]]

X = model_data[feature_cols]
y = model_data["winner"]

good_rows = X.notna().all(axis=1)
X, y = X[good_rows], y[good_rows]


tscv = TimeSeriesSplit(n_splits=5)
modelXG = XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample = 0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="logloss", 
    random_state=42
    )

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    modelXG.fit(X_train, y_train)
    print(modelXG.score(X_test, y_test))

print(f"\n")
scaler = StandardScaler()

modelLR = LogisticRegression(max_iter=1000, random_state=42)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelLR.fit(X_train_scaled, y_train)
    print(modelLR.score(X_test_scaled, y_test))

print(f"\n")

modelRF = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    modelRF.fit(X_train, y_train)
    print(modelRF.score(X_test, y_test))

print(f"\n")


modelDT = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=50,
    criterion='log_loss',
    random_state=42
)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    modelDT.fit(X_train, y_train)
    print(modelDT.score(X_test, y_test))

print(f"\n")

# -------------------- Baseline: Elo Only --------------------
elo_features = ["home_elo_pre", "away_elo_pre"]
X_elo = model_data[elo_features]
y_elo = model_data["winner"]

model_elo = LogisticRegression(max_iter=1000, random_state=42)

print("ELO-ONLY BASELINE")
for train_idx, test_idx in tscv.split(X_elo):
    X_train_elo, X_test_elo = X_elo.iloc[train_idx], X_elo.iloc[test_idx]
    y_train_elo, y_test_elo = y_elo.iloc[train_idx], y_elo.iloc[test_idx]

    scaler_elo = StandardScaler()
    X_train_elo_scaled = scaler_elo.fit_transform(X_train_elo)
    X_test_elo_scaled = scaler_elo.transform(X_test_elo)

    model_elo.fit(X_train_elo_scaled, y_train_elo)
    print(model_elo.score(X_test_elo_scaled, y_test_elo))



# -------------------- Baseline: Vegas Only --------------------
vegas_features = ["spread_line", "home_spread_odds", "away_spread_odds",
                   "total_line", "over_odds", "under_odds",
                   "home_moneyline", "away_moneyline"]
X_vegas = model_data[vegas_features]
y_vegas = model_data["winner"]

model_vegas = LogisticRegression(max_iter=1000, random_state=42)

print("VEGAS-ONLY BASELINE")
for train_idx, test_idx in tscv.split(X_vegas):
    X_train_vegas, X_test_vegas = X_vegas.iloc[train_idx], X_vegas.iloc[test_idx]
    y_train_vegas, y_test_vegas = y_vegas.iloc[train_idx], y_vegas.iloc[test_idx]

    scaler_vegas = StandardScaler()
    X_train_vegas_scaled = scaler_vegas.fit_transform(X_train_vegas)
    X_test_vegas_scaled = scaler_vegas.transform(X_test_vegas)

    model_vegas.fit(X_train_vegas_scaled, y_train_vegas)
    print(model_vegas.score(X_test_vegas_scaled, y_test_vegas))



# ------------------------ Ensemble Model --------------------

modelLR_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

ensemble = VotingClassifier(
    estimators=[
        ('xgb', modelXG),
        ('lr', modelLR_pipeline),
        ('rf', modelRF),
        ('dt', modelDT)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)   # reuses the last fold's train split — same X_train/y_train left over from your XGBoost loop


# -------------------- Model Evaluation --------------------

def model_results(model, X_test, Y_test, Y_train, feature_cols):
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)

    # print("xgboost:", xgboost.__version__)
    # print("sklearn:", sklearn.__version__)
    # print("pandas:", pd.__version__)
    # print("numpy:", np.__version__)
    print(f"\nModel: {type(model).__name__} \n")

    print("="*60)
    print("ACCURACY & LOG LOSS")
    print("="*60)
    acc = accuracy_score(Y_test, Y_pred)
    ll = log_loss(Y_test, Y_proba)
    print(f"Total Games: {len(X_test)}")
    print(f"Accuracy: {acc:.1%}")
    print(f"Log Loss: {ll:.4f} (random guessing ~0.6931, lower is better)\n")

    print("="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    majority_class = Y_train.mode()[0]
    baseline_acc = (Y_test == majority_class).mean()
    print(f"Majority-class baseline: {baseline_acc:.1%}")
    print(f"Your model beats this baseline by {(acc - baseline_acc):.1%}.\n")

    print("="*60)
    print("MODEL CONFIDENCE / CALIBRATION")
    print("="*60)
    print("Checks whether a stated probability matches reality — e.g. among games")
    print("the model called '70% home win', did the home team actually win ~70% of the time?\n")

    probs_home = Y_proba[:, 1]
    y_true = np.array(Y_test)
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges[1:-1], probs_home), 0, n_bins - 1)

    print(f"{'Predicted':>10} {'Actual':>10} {'Games':>8}")
    gaps = []
    weights = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 5:
            pred_avg = probs_home[mask].mean()
            actual_avg = y_true[mask].mean()
            count = mask.sum()
            print(f"{pred_avg:>9.1%} {actual_avg:>9.1%} {count:>8d}")
            gaps.append(abs(actual_avg - pred_avg))
            weights.append(count)

    if gaps:
        avg_gap = np.average(gaps, weights=weights)
        print(f"\nAverage calibration gap: {avg_gap:.3f} (0 = perfectly confident and correct)")
    else:
        print("\nNot enough games per bucket to check calibration.")

    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE")
    print("="*60)
    print(classification_report(Y_test, Y_pred, target_names=['Away Win', 'Home Win']))

    print("="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    labels = [0, 1]
    label_names = ['Away Win', 'Home Win']
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual: {n}" for n in label_names],
                            columns=[f"Pred: {n}" for n in label_names])
    print(cm_df)

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    if isinstance(model, XGBClassifier) or isinstance(model, RandomForestClassifier):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        for name, value in importances.items():
            print(f"{name:<40} {value:.4f}")
    elif isinstance(model, LogisticRegression):
        importances = pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
        for name, value in importances.items():
            print(f"{name:<40} {value:.4f}")
    else:
        print(f"No feature importance method available for {type(model).__name__}")
    print("="*60)


# model_results(modelXG, X_test, y_test, y_train, feature_cols)
# model_results(modelLR, X_test_scaled, y_test, y_train, feature_cols)
# model_results(modelRF, X_test, y_test, y_train, feature_cols)
# model_results(modelDT, X_test, y_test, y_train, feature_cols)


results_summary = []

def record_results(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    probs_home = y_proba[:, 1]
    y_true = np.array(y_test)
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges[1:-1], probs_home), 0, n_bins - 1)

    gaps, weights = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 5:
            gaps.append(abs(y_true[mask].mean() - probs_home[mask].mean()))
            weights.append(mask.sum())
    cal_gap = np.average(gaps, weights=weights) if gaps else float("nan")

    results_summary.append({
        "model": model_name,
        "accuracy": acc,
        "log_loss": ll,
        "cal_gap": cal_gap
    })


def print_summary():
    print("\n" + "="*60)
    print("SUMMARY — ALL MODELS")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Log Loss':>10} {'Cal Gap':>10}")
    for r in results_summary:
        print(f"{r['model']:<25} {r['accuracy']:>9.1%} {r['log_loss']:>10.4f} {r['cal_gap']:>10.3f}")
    print("="*60)



record_results("Elo-only", model_elo, X_test_elo_scaled, y_test_elo)
record_results("Vegas-only", model_vegas, X_test_vegas_scaled, y_test_vegas)
record_results("XGBoost", modelXG, X_test, y_test)
record_results("Logistic Regression", modelLR, X_test_scaled, y_test)
record_results("Random Forest", modelRF, X_test, y_test)
record_results("Decision Tree", modelDT, X_test, y_test)
record_results("Ensemble", ensemble, X_test, y_test)

print_summary()

scaler_vegas_full = StandardScaler()
X_vegas_scaled_full = scaler_vegas_full.fit_transform(X_vegas)
model_vegas.fit(X_vegas_scaled_full, y_vegas)

modelXG.fit(X, y)

X_scaled_full = scaler.fit_transform(X)
modelLR.fit(X_scaled_full, y)

modelRF.fit(X, y)
modelDT.fit(X, y)
ensemble.fit(X, y)
# -------------------- Prediction ---------------------

def get_current_team_stat(raw_df, team_col, value_col, team, window):
    team_games = raw_df[raw_df[team_col] == team].sort_values("gameday")
    recent = team_games[value_col].tail(window)
    return recent.mean() if len(recent) > 0 else 0.0


def get_current_elo(team):
    return elo_ratings.get(team, starting_elo)

qb_lookup = {"BAL": "Lamar Jackson", "BUF": "Josh Allen", "CIN": "Joe Burrow", "CLE": "Deshaun Watson", "DEN": "Bo Nix", "HOU": "C.J. Stroud", "IND": "Daniel Jones", "JAX": "Trevor Lawrence", 
             "KC": "Patrick Mahomes", "LV": "Kirk Cousins", "LAC": "Justin Herbert", "MIA": "Malik Willis", "NE": "Drake Maye", "NYJ": "Geno Smith", "PIT": "Aaron Rodgers", "TEN": "Cam Ward",
             "ARI": "Jacoby Brissett", "ATL": "Michael Penix Jr.", "CAR": "Bryce Young", "CHI": "Tyson Bagent", "DAL": "Dak Prescott", "DET": "Jared Goff", "GB": "Jordan Love", "LA": "Matthew Stafford", 
             "MIN": "Kyler Murray", "NO": "Tyler Shough", "NYG": "Jameis Winston", "PHI": "Jalen Hurts", "SF": "Brock Purdy", "SEA": "Sam Darnold", "TB": "Jalon Daniels", "WAS": "Jayden Daniels"}

def get_qb_epa(qb_name):
    rows = qb_stats[qb_stats["player_display_name"] == qb_name].sort_values("gameday")
    if len(rows) == 0:
        return 0.0
    val = rows["career_qb_epa"].iloc[-1]
    return val if not pd.isna(val) else rows["passing_epa"].mean()

# print("Seasons requested:", seasons)
# print("Seasons actually in depth_charts:", sorted(depth_charts["season"].unique()))
# print("last_season variable:", last_season)
# print("pbp seasons available:", sorted(pbp["season"].dropna().unique()))
# print("player_stats seasons available:", sorted(player_stats["season"].dropna().unique()))

def encode_categorical(value, prefix, feature_cols):
    result = {}
    for col in feature_cols:
        if col.startswith(prefix + "_"):
            result[col] = 1 if col == f"{prefix}_{value}" else 0
    return result


def predict_week(model, model_name, week, season, feature_cols, is_scaled_model=False, scaler=None, show_moneyline = False):
    stat_specs = [
        (off_epa, "posteam", "off_avg_epa", "off_epa_prev", [1]),
        (off_epa, "posteam", "off_avg_epa", "off_avg_epa", [3, 8]),
        (def_epa, "defteam", "def_avg_epa", "def_epa_prev", [1]),
        (def_epa, "defteam", "def_avg_epa", "def_avg_epa", [3, 8]),
        (third_down_off, "posteam", "off_3rd_down_pct", "off_3rd_down_pct", [3, 8]),
        (third_down_def, "defteam", "def_3rd_down_pct_allowed", "def_3rd_down_pct_allowed", [3, 8]),
        (turnovers, "team", "turnover_diff", "turnover_diff", [3, 8]),
        (redzone_off, "posteam", "off_redzone_pct", "off_redzone_pct", [3, 8]),
        (redzone_def, "defteam", "def_redzone_pct_allowed", "def_redzone_pct_allowed", [3, 8]),
    ]

    week_games = sched[(sched["season"] == season) & (sched["week"] == week)]

    odds_cols = ["spread_line", "home_spread_odds", "away_spread_odds", "total_line",
                 "over_odds", "under_odds", "home_moneyline", "away_moneyline",
                 "home_rest", "away_rest", "div_game"]

    results = []

    for game in week_games.itertuples(index=False):
        home, away = game.home_team, game.away_team
        row = {col: getattr(game, col) for col in odds_cols}
        row["temp"], row["wind"] = 70.0, 0.0

        # row.update(encode_categorical(game.game_type, "game_type", feature_cols))
        # row.update(encode_categorical(game.roof, "roof", feature_cols))
        # row.update(encode_categorical(game.surface.strip(), "surface", feature_cols))

        row["home_elo_pre"] = get_current_elo(home)
        row["away_elo_pre"] = get_current_elo(away)

        for raw_df, team_col, value_col, out_name, windows in stat_specs:
            for w in windows:
                suffix = "" if w == 1 else f"_r{w}"
                row[f"{out_name}{suffix}_home"] = get_current_team_stat(raw_df, team_col, value_col, home, w)
                row[f"{out_name}{suffix}_away"] = get_current_team_stat(raw_df, team_col, value_col, away, w)

        row["home_qb_epa"] = get_qb_epa(qb_lookup.get(home))
        row["away_qb_epa"] = get_qb_epa(qb_lookup.get(away))

        for col in feature_cols:
            if col not in row:
                row[col] = 0

        row_df = pd.DataFrame([row])[feature_cols]
        row_df = row_df.fillna(0)

        # test_team = "CHI"
        # if home == test_team or away == test_team:  
        #     print(f"\n=== ALL FEATURES: {away} @ {home} ===")
        #     for col in feature_cols:
        #         print(f"{col:<40} {row.get(col)}")
        
        
        if is_scaled_model and scaler is not None:
            row_df = scaler.transform(row_df)

        proba = model.predict_proba(row_df)[0]
        away_probability = proba[0]
        home_probability = proba[1]

        print(
            f"{away + ' Wins:':<12}{away_probability:>5.1%}"
            f"{' @ ':^15}"
            f"{home + ' Wins:':<12}{home_probability:>5.1%}"
        )
        if show_moneyline:
            print(f"    Moneyline: {away} {row['away_moneyline']:>6} | {home} {row['home_moneyline']:>6}")

        results.append({
                "game_id": game.game_id,
                "season": season,
                "week": week,
                "model": model_name,
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(float(home_probability), 4),
                "away_win_prob": round(float(away_probability), 4),
                "predicted_winner": home if home_probability >= 0.5 else away,
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    return pd.DataFrame(results)

# ------------------------ Save (Claude) -------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PREDICTIONS_LOG = SCRIPT_DIR / "predictions_log.csv"

def update_predictions_log(new_predictions, sched, log_path=PREDICTIONS_LOG):
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
    else:
        log = pd.DataFrame(columns=list(new_predictions.columns) + ["actual_winner", "correct", "home_score", "away_score"])

    results_lookup = sched.set_index("game_id")["result"]
    scores_lookup = sched.set_index("game_id")[["home_score", "away_score"]]
    played_game_ids = set(results_lookup[results_lookup.notna()].index)

    # A week is "locked" once its first game (earliest gameday) has been
    # played. Once locked, nothing in that week gets overwritten, even
    # games in it that haven't kicked off yet.
    week_of_game = sched.set_index("game_id")[["season", "week"]]
    first_game_per_week = (
        sched.sort_values("gameday")
        .groupby(["season", "week"])["game_id"]
        .first()
    )
    locked_weeks = set(first_game_per_week[first_game_per_week.isin(played_game_ids)].index)

    def is_locked(gid):
        if gid not in week_of_game.index:
            return False
        season, week = week_of_game.loc[gid]
        return (season, week) in locked_weeks

    def fill_actual(row):
        gid = row["game_id"]

        # Backfill scores for any played game, even one that was already
        # graded before this column existed — so older rows pick them up
        # on the next run instead of staying blank forever.
        if gid in played_game_ids:
            hs, aws = scores_lookup.loc[gid, "home_score"], scores_lookup.loc[gid, "away_score"]
        else:
            hs, aws = row.get("home_score", np.nan), row.get("away_score", np.nan)

        if pd.notna(row.get("actual_winner")):
            return pd.Series({"actual_winner": row["actual_winner"], "correct": row.get("correct"),
                               "home_score": hs, "away_score": aws})

        if gid in played_game_ids:
            res = results_lookup.loc[gid]
            actual = row["home_team"] if res > 0 else row["away_team"] if res < 0 else "TIE"
            return pd.Series({
                "actual_winner": actual,
                "correct": (actual == row["predicted_winner"]) if actual != "TIE" else np.nan,
                "home_score": hs, "away_score": aws,
            })
        return pd.Series({"actual_winner": row.get("actual_winner", np.nan),
                           "correct": row.get("correct", np.nan),
                           "home_score": hs, "away_score": aws})

    if not log.empty:
        log[["actual_winner", "correct", "home_score", "away_score"]] = log.apply(fill_actual, axis=1)

    # skip anything for a game whose week is already locked in
    new_predictions = new_predictions[~new_predictions["game_id"].apply(is_locked)]

    key_cols = ["game_id", "model"]
    if not log.empty:
        log_locked = log["game_id"].apply(is_locked)
        in_new_batch = log.set_index(key_cols).index.isin(new_predictions.set_index(key_cols).index)
        log = log[~(~log_locked & in_new_batch)]

    new_predictions["actual_winner"] = np.nan
    new_predictions["correct"] = np.nan
    new_predictions["home_score"] = np.nan
    new_predictions["away_score"] = np.nan

    log = pd.concat([log, new_predictions], ignore_index=True)
    odds_lookup = sched.set_index("game_id")[["home_moneyline", "away_moneyline", "spread_line"]]

    for col in ["home_moneyline", "away_moneyline", "spread_line"]:
        if col not in log.columns:
            log[col] = np.nan
        latest = log["game_id"].map(odds_lookup[col])
        log[col] = latest.fillna(log[col])
    log.to_csv(log_path, index=False)
    print(f"\nSaved {len(new_predictions)} predictions to {log_path} (log now has {len(log)} rows total)")
    return log

# ------------------------ Debugging --------------------
def search_qb(name_query, season=None):
    stats_matches = qb_stats[qb_stats["player_display_name"].str.contains(name_query, case=False, na=False)]

    print(f"=== qb_stats matches for '{name_query}' ===")
    if len(stats_matches) == 0:
        print("No matches — this player has no recorded NFL pass attempts in the loaded seasons.")
    else:
        print(stats_matches["player_display_name"].unique())

    if season is not None:
        rosters = nfl.load_rosters([season]).to_pandas()
        roster_matches = rosters[
            (rosters["position"] == "QB") &
            (rosters["full_name"].str.contains(name_query, case=False, na=False))
        ]
        print(f"\n=== {season} roster matches for '{name_query}' ===")
        if len(roster_matches) == 0:
            print(f"No matches — not found on any {season} QB roster.")
        else:
            print(roster_matches[["team", "full_name", "depth_chart_position", "status"]])

# print("\n=== QB LOOKUP — EPA CHECK ===")
# for team, qb_name in qb_lookup.items():
#     epa = get_qb_epa(qb_name)
#     print(f"{team:<5} {qb_name:<25} qb_epa={epa}")



# ---------------------- Predictions for Week --------------------
current_week = 5
last_week = current_week - 1
now = datetime.now()

print(f"\nXGBOOST PREDICTIONS — WEEK {current_week}, {current_season}")
pred_xgb = predict_week(modelXG, "XGBOOST", current_week, current_season, feature_cols)

print(f"\nLOGISTIC REGRESSION PREDICTIONS — WEEK {current_week}, {current_season}")
pred_lr = predict_week(modelLR, "LOGISTIC REGRESSION", current_week, current_season, feature_cols, is_scaled_model=True, scaler=scaler)

print(f"\nRANDOM FOREST PREDICTIONS — WEEK {current_week}, {current_season}")
pred_rf = predict_week(modelRF, "RANDOM FOREST", current_week, current_season, feature_cols)

print(f"\nDECISION TREE PREDICTIONS — WEEK {current_week}, {current_season}")
pred_dt = predict_week(modelDT, "DECISION TREE", current_week, current_season, feature_cols)

print(f"\nVEGAS-ONLY PREDICTIONS — WEEK {current_week}, {current_season}")
print(now.strftime("%m/%d %H:%M"))
pred_vegas = predict_week(model_vegas, "VEGAS-ONLY", current_week, current_season, vegas_features, is_scaled_model=True, scaler=scaler_vegas_full, show_moneyline=True)

print(f"\n")
print(f"\nENSEMBLE FINAL NFL PREDICTIONS — WEEK {current_week}, {current_season}")
print(now.strftime("%m/%d %H:%M"))
pred_ensemble = predict_week(ensemble, "ENSEMBLE", current_week, current_season, feature_cols)
print(f"\n")

all_predictions = pd.concat(
    [pred_xgb, pred_lr, pred_rf, pred_dt, pred_vegas, pred_ensemble],
    ignore_index=True
)
update_predictions_log(all_predictions, sched)

