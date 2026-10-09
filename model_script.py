"""
Prescriptive defensive-coverage recommender.

Run through the terminal from this folder with:   python model_script.py

Predicts which pass coverage is most EFFECTIVE (a defensive win/lose outcome),
not exact yardage. Two models:
  1. ML classifier  -> P(defensive success | situation, coverage)
  2. Empirical model -> actual historical success rate per coverage

Requires: pandas, scikit-learn  (install once: python -m pip install pandas scikit-learn)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

RNG = 42
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Step 1: build plays_filtered.csv from plays.csv (keeps the columns we need)
# ---------------------------------------------------------------------------
def build_filtered():
    plays = pd.read_csv(HERE / "plays.csv")
    columns = [
        "playId", "possessionTeam", "defensiveTeam", "quarter", "down",
        "yardsToGo", "yardlineSide", "yardlineNumber", "gameClock",
        "preSnapHomeScore", "preSnapVisitorScore", "passResult",
        "penaltyYards", "prePenaltyPlayResult", "playResult",
        "absoluteYardlineNumber", "offenseFormation", "personnelO",
        "defendersInBox", "personnelD", "pff_passCoverageType", "pff_passCoverage",
    ]
    filtered = plays[columns]
    filtered.to_csv(HERE / "plays_filtered.csv", index=False)
    print(f"plays_filtered.csv: {filtered.shape[0]} rows x {filtered.shape[1]} cols")
    return filtered


def clock_to_seconds(s):
    """'MM:SS' -> seconds remaining in the quarter."""
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Step 2: load data and build the target (defensive success)
# ---------------------------------------------------------------------------
def load_data():
    fp = HERE / "plays_filtered.csv"
    if not fp.exists():
        build_filtered()
    df = pd.read_csv(fp).set_index("playId")
    df["penaltyYards"] = df["penaltyYards"].fillna(0)   # NaN means no penalty
    df["clockSeconds"] = df["gameClock"].map(clock_to_seconds)

    def offense_succeeded(row):
        ytg, gained = row["yardsToGo"], row["playResult"]
        if row["down"] == 1:   need = 0.4 * ytg
        elif row["down"] == 2: need = 0.6 * ytg
        else:                  need = ytg          # 3rd/4th down: must convert
        return gained >= need

    df["def_success"] = (~df.apply(offense_succeeded, axis=1)).astype(int)
    df.loc[df["penaltyYards"] > 0, "def_success"] = 0   # defensive penalty = failure
    print(f"{len(df)} plays loaded; defensive success rate: {df['def_success'].mean():.3f}")
    return df


SITUATION_NUM = ["quarter", "down", "yardsToGo", "yardlineNumber", "clockSeconds",
                 "preSnapHomeScore", "preSnapVisitorScore"]
SITUATION_CAT = ["possessionTeam", "defensiveTeam", "offenseFormation", "personnelO"]
COVERAGE_CAT  = ["pff_passCoverage", "pff_passCoverageType"]
FEATURE_COLS  = SITUATION_NUM + SITUATION_CAT + COVERAGE_CAT


# ---------------------------------------------------------------------------
# Step 3: train the ML classifier
# ---------------------------------------------------------------------------
def train_model(df):
    model_df = df.dropna(subset=["clockSeconds"]).copy()
    for c in SITUATION_CAT + COVERAGE_CAT:
        model_df[c] = model_df[c].astype(str).astype("category")

    X = model_df[FEATURE_COLS]
    y = model_df["def_success"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG, stratify=y)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=RNG, stratify=y_train)

    base_clf = HistGradientBoostingClassifier(
        categorical_features=SITUATION_CAT + COVERAGE_CAT, random_state=RNG,
        max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=50, l2_regularization=1.0,
    )
    base_clf.fit(X_fit, y_fit)
    cal_model = CalibratedClassifierCV(FrozenEstimator(base_clf), method="isotonic")
    cal_model.fit(X_cal, y_cal)

    proba = cal_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n--- ML classifier (held-out test) ---")
    print(f"AUC: {roc_auc_score(y_test, proba):.3f} (0.5 = no skill)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f} | baseline: {max(y_test.mean(), 1 - y_test.mean()):.3f}")
    print(f"LogLoss:  {log_loss(y_test, proba):.4f} | baseline: {log_loss(y_test, np.full_like(proba, y_train.mean(), dtype=float)):.4f}")

    combos = model_df[COVERAGE_CAT].drop_duplicates().sort_values(COVERAGE_CAT).reset_index(drop=True)
    return base_clf, X, combos


def recommend_ml(base_clf, X, combos, possessionTeam, defensiveTeam, quarter, down,
                 yardsToGo, yardlineNumber, gameClock, preSnapHomeScore, preSnapVisitorScore,
                 offenseFormation, personnelO, top_n=5):
    """Rank coverages by predicted defensive-success probability (most effective first)."""
    base = {
        "quarter": quarter, "down": down, "yardsToGo": yardsToGo,
        "yardlineNumber": yardlineNumber, "clockSeconds": clock_to_seconds(gameClock),
        "preSnapHomeScore": preSnapHomeScore, "preSnapVisitorScore": preSnapVisitorScore,
        "possessionTeam": possessionTeam, "defensiveTeam": defensiveTeam,
        "offenseFormation": str(offenseFormation), "personnelO": str(personnelO),
    }
    trial = combos.copy()
    for k, v in base.items():
        trial[k] = v
    for c in SITUATION_CAT + COVERAGE_CAT:
        trial[c] = pd.Categorical(trial[c], categories=X[c].cat.categories)
    trial["def_success_prob"] = base_clf.predict_proba(trial[FEATURE_COLS])[:, 1]
    return (trial.sort_values("def_success_prob", ascending=False)
                 [["pff_passCoverage", "pff_passCoverageType", "def_success_prob"]]
                 .head(top_n).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Step 4: empirical model (actual historical success rates)
# ---------------------------------------------------------------------------
def field_zone(yl):
    if yl <= 10: return "own_deep(1-10)"
    if yl <= 50: return "own_mid(11-50)"
    if yl <= 80: return "opp_mid(51-80)"
    return "opp_redzone(81-100)"


def dist_bucket(d):
    if d <= 3: return "short(1-3)"
    if d <= 7: return "mid(4-7)"
    return "long(8+)"


def recommend_empirical(df, down, yardsToGo, yardlineNumber, min_plays=10, top_n=10):
    emp = df.copy()
    emp["field_zone"]  = emp["yardlineNumber"].map(field_zone)
    emp["dist_bucket"] = emp["yardsToGo"].map(dist_bucket)
    fz, db = field_zone(yardlineNumber), dist_bucket(yardsToGo)
    sub = emp[(emp["down"] == down) & (emp["dist_bucket"] == db) & (emp["field_zone"] == fz)]
    g = (sub.groupby(["pff_passCoverage", "pff_passCoverageType"])
            .agg(def_success_rate=("def_success", "mean"),
                 avg_yards_allowed=("playResult", "mean"),
                 plays=("def_success", "size"))
            .reset_index())
    g = g[g["plays"] >= min_plays].sort_values("def_success_rate", ascending=False).reset_index(drop=True)
    print(f"\nSituation: down {down}, {db} to go, {fz}  |  {len(sub)} comparable past plays")
    return g.head(top_n)


# ---------------------------------------------------------------------------
# GUI: pop-up window with an input box per field
# ---------------------------------------------------------------------------
def _relaunch_in_own_console():
    """If started from an editor 'Run' button (which can hide GUI windows), re-spawn
    this script in a fresh, detached console so the Tk window shows normally.
    (No longer used - kept for reference.) Always returns False."""
    return False


def launch_gui():
    import threading
    import tkinter as tk
    from tkinter import ttk, messagebox

    # --- Build and SHOW the window first, before any slow work, so it always appears. ---
    root = tk.Tk()
    root.title("Coverage recommender")
    root.geometry("760x600")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="x")

    # Static option lists so the form can be built before the model finishes loading.
    teams = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
             "IND","JAX","KC","LA","LAC","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT",
             "SEA","SF","TB","TEN","WAS"]
    formations = ["EMPTY","I_FORM","JUMBO","PISTOL","SHOTGUN","SINGLEBACK","WILDCAT"]
    personnel = ["1 RB, 1 TE, 3 WR","1 RB, 2 TE, 2 WR","2 RB, 1 TE, 2 WR","1 RB, 0 TE, 4 WR",
                 "1 RB, 3 TE, 1 WR","2 RB, 2 TE, 1 WR","0 RB, 1 TE, 4 WR","6 OL, 1 RB, 1 TE, 2 WR",
                 "2 RB, 0 TE, 3 WR","0 RB, 2 TE, 3 WR"]

    # field spec: (label, kind, default, options)
    fields = [
        ("Possession team", "combo", "DAL", teams),
        ("Defensive team",  "combo", "TB",  teams),
        ("Quarter",         "entry", "4",   None),
        ("Down",            "entry", "3",   None),
        ("Yards to go",     "entry", "8",   None),
        ("Yard line number","entry", "25",  None),
        ("Game clock (MM:SS)", "entry", "02:00", None),
        ("Pre-snap home score", "entry", "14", None),
        ("Defensive team score (visitor)", "entry", "10", None),
        ("Offense formation", "combo", "SHOTGUN", formations),
        ("Offense personnel", "combo", "1 RB, 1 TE, 3 WR", personnel),
    ]

    widgets = {}
    for i, (label, kind, default, options) in enumerate(fields):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=3, padx=(0, 8))
        if kind == "combo":
            var = tk.StringVar(value=default)
            w = ttk.Combobox(frm, textvariable=var, values=options, width=32)
            w.set(default)
        else:
            w = ttk.Entry(frm, width=34)
            w.insert(0, default)
        w.grid(row=i, column=1, sticky="w", pady=3)
        widgets[label] = w

    # Button row goes directly under the inputs so it is always visible,
    # then the results box fills whatever space is left below it.
    btns = ttk.Frame(root, padding=12)
    btns.pack(fill="x")

    out = tk.Text(root, height=16, width=90, wrap="none")
    out.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def show(text):
        out.delete("1.0", "end")
        out.insert("1.0", text)

    # Holds the trained model once the background thread finishes.
    state = {"ready": False, "df": None, "base_clf": None, "X": None, "combos": None}

    def on_recommend():
        if not state["ready"]:
            messagebox.showinfo("Please wait", "The model is still loading. Try again in a moment.")
            return
        try:
            possessionTeam = widgets["Possession team"].get().strip()
            defensiveTeam = widgets["Defensive team"].get().strip()
            quarter = int(widgets["Quarter"].get())
            down = int(widgets["Down"].get())
            yardsToGo = int(widgets["Yards to go"].get())
            yardlineNumber = int(widgets["Yard line number"].get())
            gameClock = widgets["Game clock (MM:SS)"].get().strip()
            preSnapHomeScore = int(widgets["Pre-snap home score"].get())
            preSnapVisitorScore = int(widgets["Defensive team score (visitor)"].get())
            offenseFormation = widgets["Offense formation"].get().strip()
            personnelO = widgets["Offense personnel"].get().strip()
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Quarter, Down, Yards to go, Yard line and Scores must be whole numbers.")
            return

        ml = recommend_ml(state["base_clf"], state["X"], state["combos"],
                          possessionTeam, defensiveTeam, quarter, down,
                          yardsToGo, yardlineNumber, gameClock, preSnapHomeScore, preSnapVisitorScore,
                          offenseFormation, personnelO)
        emp = recommend_empirical(state["df"], down, yardsToGo, yardlineNumber)
        text = (
            "=== ML recommendation (ranked by predicted defensive-success probability) ===\n"
            + ml.to_string(index=False)
            + "\n\n=== Empirical: actual historical success rate in comparable plays ===\n"
            + "(prefer rows with a larger 'plays' count)\n"
            + emp.to_string(index=False)
        )
        show(text)

    rec_btn = ttk.Button(btns, text="Recommend coverage", command=on_recommend, state="disabled")
    rec_btn.pack(side="left")
    ttk.Button(btns, text="Quit", command=root.destroy).pack(side="left", padx=8)

    show("Loading model, please wait... (this takes a few seconds the first time)")

    # Train the model in a background thread so the window stays responsive and visible.
    def _load():
        df = load_data()
        base_clf, X, combos = train_model(df)
        state.update(df=df, base_clf=base_clf, X=X, combos=combos, ready=True)

    def _poll():
        if state["ready"]:
            rec_btn.config(state="normal")
            show("Model ready. Fill in the situation above and click 'Recommend coverage'.")
        else:
            root.after(200, _poll)

    threading.Thread(target=_load, daemon=True).start()
    root.after(200, _poll)

    # Force the window to the front so it isn't hidden behind the editor/terminal.
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))  # stay on top only briefly
    root.focus_force()

    root.mainloop()


def _acquire_single_instance_lock(stale_after_seconds=10):
    """Stop the rapid re-launch loop (an editor re-running the file every save) from
    stacking windows, WITHOUT ever blocking a normal relaunch.

    The loop fires many times per second, so a lock that is only a few seconds old
    means 'another copy just started' -> skip. Any older lock is treated as stale and
    reclaimed, so a crashed/closed previous run never locks you out. The lock is also
    refreshed and removed on exit.
    """
    import os, time, atexit
    lock = HERE / ".recommender.lock"

    if lock.exists():
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = stale_after_seconds + 1
        if age < stale_after_seconds:
            print("A recommender window was just opened; not opening another.")
            return None
        # old lock -> stale, reclaim it

    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.exists() and lock.unlink())
    return lock


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        # Old behaviour: print a worked example to the terminal, no GUI.
        df = load_data()
        base_clf, X, combos = train_model(df)
        print("\n=== ML recommendation: 3rd & 8 from own 25, SHOTGUN / 1 RB,1 TE,3 WR ===")
        print(recommend_ml(base_clf, X, combos,
                           "DAL", "TB", quarter=4, down=3, yardsToGo=8,
                           yardlineNumber=25, gameClock="02:00", preSnapHomeScore=14,
                           preSnapVisitorScore=10,
                           offenseFormation="SHOTGUN", personnelO="1 RB, 1 TE, 3 WR").to_string(index=False))
        print("\n=== Empirical recommendation: 3rd & 8 from own 25 ===")
        print(recommend_empirical(df, down=3, yardsToGo=8, yardlineNumber=25).to_string(index=False))
    else:
        if _acquire_single_instance_lock() is not None:
            launch_gui()
