#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
IPL Win Predictor — Streamlit App
===================================
Run with: streamlit run app.py

Requires in same directory:
  - prematch_model.pkl
  - live_model.pkl
  - prematch_features.csv   (for team/venue lists)
  - live_features.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.05em;
}

.stApp {
    background: #0d0d0d;
    color: #f0f0f0;
}

.metric-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.metric-box .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    margin-bottom: 6px;
}

.metric-box .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    line-height: 1;
}

.win-bar-wrap {
    background: #1a1a1a;
    border-radius: 999px;
    height: 14px;
    margin: 8px 0;
    overflow: hidden;
}

.win-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.tag {
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.tag-live { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b44; }
.tag-pre  { background: #ffd70022; color: #ffd700; border: 1px solid #ffd70044; }

section[data-testid="stSidebar"] {
    background: #111;
}
</style>
""", unsafe_allow_html=True)


# ── Load models & data ───────────────────────

@st.cache_resource
def load_models():
    with open("prematch_model.pkl", "rb") as f:
        pre = pickle.load(f)
    with open("live_model.pkl", "rb") as f:
        live = pickle.load(f)
    return pre, live

@st.cache_data
def load_data():
    pm = pd.read_csv("prematch_features.csv")
    lv = pd.read_csv("live_features.csv")
    return pm, lv

try:
    pre_bundle, live_bundle = load_models()
    pm_df, lv_df = load_data()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    missing = str(e)


# ── Helpers ──────────────────────────────────

def win_bar(prob, team1, team2):
    p1 = int(prob * 100)
    p2 = 100 - p1
    color1 = "#ffd700"
    color2 = "#ff4b4b"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
        <span style="color:{color1};font-weight:600">{team1}</span>
        <span style="color:{color2};font-weight:600">{team2}</span>
    </div>
    <div class="win-bar-wrap">
        <div class="win-bar-fill" style="width:{p1}%;background:linear-gradient(90deg,{color1},{color2});"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-family:'Bebas Neue',sans-serif;font-size:1.4rem">
        <span style="color:{color1}">{p1}%</span>
        <span style="color:{color2}">{p2}%</span>
    </div>
    """, unsafe_allow_html=True)


def shap_bar_chart(shap_vals, feature_names, title="Feature contributions"):
    sv  = np.array(shap_vals)
    idx = np.argsort(np.abs(sv))[-10:]   # top 10 by magnitude
    vals  = sv[idx]
    names = [feature_names[i] for i in idx]
    colors = ["#ffd700" if v > 0 else "#ff4b4b" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    bars = ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#444", linewidth=1)
    ax.set_xlabel("SHAP value", color="#888", fontsize=9)
    ax.set_title(title, color="#f0f0f0", fontsize=10, pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    gold  = mpatches.Patch(color="#ffd700", label="↑ Increases win prob")
    red   = mpatches.Patch(color="#ff4b4b", label="↓ Decreases win prob")
    ax.legend(handles=[gold, red], fontsize=7, facecolor="#111",
              labelcolor="#ccc", loc="lower right")
    plt.tight_layout()
    return fig


# ── Header ───────────────────────────────────

st.markdown("""
<div style="padding: 2rem 0 1rem;">
    <div style="font-size:13px;color:#888;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px">
        Machine Learning · IPL 2008–2024
    </div>
    <h1 style="font-size:3.5rem;margin:0;color:#f0f0f0;line-height:1">
        🏏 IPL WIN PREDICTOR
    </h1>
    <p style="color:#666;margin-top:8px;font-size:14px">
        Two models: pre-match (XGBoost) and live ball-by-ball (LightGBM) · 77% live accuracy
    </p>
</div>
<hr style="border-color:#222;margin-bottom:2rem">
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"Could not load model files: {missing}")
    st.info("Make sure `prematch_model.pkl` and `live_model.pkl` are in the same directory as `app.py`.")
    st.stop()


# ── Tabs ─────────────────────────────────────

tab1, tab2 = st.tabs(["🔮 Pre-Match Predictor", "📡 Live Win Probability"])


# ════════════════════════════════════════════
# TAB 1 — PRE-MATCH
# ════════════════════════════════════════════

with tab1:
    st.markdown('<span class="tag tag-pre">Pre-match · XGBoost</span>', unsafe_allow_html=True)
    st.markdown("### Predict the winner before the first ball")
    st.caption("Model uses Elo ratings, H2H history, venue stats, toss, and recent form.")

    teams  = sorted(pd.concat([pm_df["team1"], pm_df["team2"]]).unique())
    venues = sorted(pm_df["venue"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        team1  = st.selectbox("Team 1 (batting order)", teams, index=0)
    with col2:
        team2  = st.selectbox("Team 2", teams, index=3)
    with col3:
        venue  = st.selectbox("Venue", venues, index=0)

    col4, col5 = st.columns(2)
    with col4:
        toss_winner = st.radio("Toss won by", [team1, team2], horizontal=True)
    with col5:
        toss_decision = st.radio("Toss decision", ["Bat", "Field"], horizontal=True)

    st.markdown("---")
    st.markdown("**Historical context** (auto-filled from data; adjust if needed)")
    c1, c2, c3, c4 = st.columns(4)

    # Pull defaults from data
    def safe_mean(series, default=0.5):
        return float(series.mean()) if len(series) > 0 else default

    t1_rows = pm_df[(pm_df["team1"] == team1) | (pm_df["team2"] == team1)]
    t2_rows = pm_df[(pm_df["team1"] == team2) | (pm_df["team2"] == team2)]
    h2h_rows = pm_df[
        ((pm_df["team1"] == team1) & (pm_df["team2"] == team2)) |
        ((pm_df["team1"] == team2) & (pm_df["team2"] == team1))
    ]

    with c1:
        t1_elo = st.number_input("Team 1 Elo", value=1500, step=10)
    with c2:
        t2_elo = st.number_input("Team 2 Elo", value=1500, step=10)
    with c3:
        t1_form = st.slider("Team 1 recent form", 0.0, 1.0,
                            value=safe_mean(pm_df.loc[t1_rows.index, "t1_weighted_form"
                                ] if "t1_weighted_form" in pm_df.columns else pd.Series([0.5])), step=0.05)
    with c4:
        t2_form = st.slider("Team 2 recent form", 0.0, 1.0,
                            value=safe_mean(pm_df.loc[t2_rows.index, "t2_weighted_form"
                                ] if "t2_weighted_form" in pm_df.columns else pd.Series([0.5])), step=0.05)

    if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):
        if team1 == team2:
            st.warning("Please select two different teams.")
        else:
            model     = pre_bundle["model"]
            explainer = pre_bundle["explainer"]
            features  = pre_bundle["features"]

            # Build feature vector using data medians as defaults
            elo_diff   = t1_elo - t2_elo
            elo_prob   = 1 / (1 + 10 ** (-elo_diff / 400))

            # Encode teams/venue
            sample = pm_df[
                ((pm_df["team1"] == team1) & (pm_df["team2"] == team2)) |
                ((pm_df["team1"] == team2) & (pm_df["team2"] == team1))
            ]
            if len(sample) == 0:
                sample = pm_df[pm_df["team1"] == team1]
            if len(sample) == 0:
                sample = pm_df.iloc[:1]

            # Get encodings from existing rows
            t1_enc = int(pm_df[pm_df["team1"] == team1]["team1_enc"].mode()[0]) \
                     if team1 in pm_df["team1"].values else 0
            t2_enc = int(pm_df[pm_df["team2"] == team2]["team2_enc"].mode()[0]) \
                     if team2 in pm_df["team2"].values else 1
            v_enc  = int(pm_df[pm_df["venue"] == venue]["venue_enc"].mode()[0]) \
                     if venue in pm_df["venue"].values else 0
            season_enc = int(pm_df["season_enc"].max())

            # Venue stats
            v_rows = pm_df[pm_df["venue"] == venue]
            vt1 = v_rows[(v_rows["team1"] == team1) | (v_rows["team2"] == team1)]
            vt2 = v_rows[(v_rows["team1"] == team2) | (v_rows["team2"] == team2)]

            t1_venue_wr   = safe_mean(pm_df.loc[vt1.index, "t1_venue_wr"] if len(vt1) > 0 else pd.Series([0.5]))
            t2_venue_wr   = safe_mean(pm_df.loc[vt2.index, "t2_venue_wr"] if len(vt2) > 0 else pd.Series([0.5]))
            toss_venue    = safe_mean(v_rows["toss_venue_adv"]) if "toss_venue_adv" in pm_df.columns else 0.5
            field_adv     = safe_mean(v_rows["field_first_adv"]) if "field_first_adv" in pm_df.columns else 0.5

            t1_won_toss   = 1 if toss_winner == team1 else 0
            toss_bat      = 1 if toss_decision == "Bat" else 0
            toss_interact = t1_won_toss * (1 - toss_bat) * field_adv

            # H2H
            h2h_wr = safe_mean(pm_df.loc[h2h_rows.index, "h2h_t1_win_rate"]) if len(h2h_rows) > 0 else 0.5
            h2h_n  = len(h2h_rows)

            # Season form
            t1_s_wr = safe_mean(pm_df.loc[t1_rows.index, "t1_season_wr"]) if "t1_season_wr" in pm_df.columns else 0.5
            t2_s_wr = safe_mean(pm_df.loc[t2_rows.index, "t2_season_wr"]) if "t2_season_wr" in pm_df.columns else 0.5

            payload = {
                "t1_elo"            : t1_elo,
                "t2_elo"            : t2_elo,
                "elo_diff"          : elo_diff,
                "t1_elo_prob"       : elo_prob,
                "t1_weighted_form"  : t1_form,
                "t2_weighted_form"  : t2_form,
                "form_diff"         : t1_form - t2_form,
                "t1_season_wr"      : t1_s_wr,
                "t2_season_wr"      : t2_s_wr,
                "season_wr_diff"    : t1_s_wr - t2_s_wr,
                "t1_venue_wr"       : t1_venue_wr,
                "t2_venue_wr"       : t2_venue_wr,
                "venue_wr_diff"     : t1_venue_wr - t2_venue_wr,
                "toss_venue_adv"    : toss_venue,
                "field_first_adv"   : field_adv,
                "toss_field_interact": toss_interact,
                "h2h_t1_win_rate"   : h2h_wr,
                "h2h_matches"       : h2h_n,
                "team1_won_toss"    : t1_won_toss,
                "toss_winner_batted": toss_bat,
                "team1_enc"         : t1_enc,
                "team2_enc"         : t2_enc,
                "venue_enc"         : v_enc,
                "season_enc"        : season_enc,
            }

            X = pd.DataFrame([payload])[features]
            prob  = model.predict_proba(X)[0]
            t1_wp = prob[1]

            st.markdown("---")
            st.markdown("### 🏆 Prediction")
            win_bar(t1_wp, team1, team2)

            winner = team1 if t1_wp >= 0.5 else team2
            conf   = max(t1_wp, 1 - t1_wp)
            st.success(f"**{winner}** is favoured to win · Confidence: {conf:.0%}")

            # SHAP
            with st.expander("🔍 Why this prediction? (SHAP)"):
                sv = explainer.shap_values(X)
                shap_vals = sv[0] if not isinstance(sv, list) else sv[0]
                fig = shap_bar_chart(shap_vals, features, "Feature contributions to prediction")
                st.pyplot(fig)
                plt.close()


# ════════════════════════════════════════════
# TAB 2 — LIVE
# ════════════════════════════════════════════

with tab2:
    st.markdown('<span class="tag tag-live">Live · LightGBM · 77% accuracy</span>', unsafe_allow_html=True)
    st.markdown("### Live win probability — enter current match state")
    st.caption("Updates probability as wickets fall and run rate changes over-by-over.")

    live_teams  = sorted(pd.concat([lv_df["batting_team"], lv_df["bowling_team"]]).unique())
    live_venues = sorted(lv_df["venue"].unique())

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        bat_team = st.selectbox("Batting team", live_teams, key="bat")
    with lc2:
        bowl_team = st.selectbox("Bowling team", live_teams, index=2, key="bowl")
    with lc3:
        live_venue = st.selectbox("Venue", live_venues, key="lv")

    st.markdown("---")
    st.markdown("**Current match state**")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        over_num = st.slider("Over completed", 1, 19, value=10)
    with mc2:
        cum_runs = st.number_input("Runs scored", min_value=0, max_value=400, value=80)
    with mc3:
        cum_wkts = st.number_input("Wickets fallen", min_value=0, max_value=9, value=3)
    with mc4:
        target = st.number_input("Target (runs to win)", min_value=1, max_value=500, value=165)

    # Derived values
    runs_req       = max(0, target - cum_runs)
    balls_rem      = 120 - over_num * 6
    wkts_in_hand   = 10 - cum_wkts
    crr            = cum_runs / over_num if over_num > 0 else 0
    overs_rem      = balls_rem / 6
    rrr            = runs_req / overs_rem if overs_rem > 0 else 99.0
    rrr_crr_delta  = rrr - crr

    st.markdown("**Derived stats:**")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.metric("Runs required", runs_req)
    ds2.metric("Balls remaining", balls_rem)
    ds3.metric("Current RR", f"{crr:.2f}")
    ds4.metric("Required RR", f"{rrr:.2f}", delta=f"{rrr_crr_delta:+.2f}")

    if st.button("📡 Get Live Probability", use_container_width=True, type="primary"):
        if bat_team == bowl_team:
            st.warning("Batting and bowling teams must be different.")
        elif runs_req <= 0:
            st.info("Target already achieved — batting team wins!")
        elif wkts_in_hand <= 0:
            st.info("All wickets fallen — bowling team wins!")
        else:
            model     = live_bundle["model"]
            explainer = live_bundle["explainer"]
            features  = live_bundle["features"]

            # Get encodings
            bt_enc = int(lv_df[lv_df["batting_team"] == bat_team]["batting_team_enc"].mode()[0]) \
                     if bat_team in lv_df["batting_team"].values else 0
            bl_enc = int(lv_df[lv_df["bowling_team"] == bowl_team]["bowling_team_enc"].mode()[0]) \
                     if bowl_team in lv_df["bowling_team"].values else 1
            vn_enc = int(lv_df[lv_df["venue"] == live_venue]["venue_enc"].mode()[0]) \
                     if live_venue in lv_df["venue"].values else 0

            payload = {
                "batting_team_enc" : bt_enc,
                "bowling_team_enc" : bl_enc,
                "venue_enc"        : vn_enc,
                "over_num"         : over_num,
                "cum_runs"         : cum_runs,
                "cum_wickets"      : cum_wkts,
                "target"           : target,
                "runs_required"    : runs_req,
                "balls_remaining"  : balls_rem,
                "wickets_in_hand"  : wkts_in_hand,
                "crr"              : crr,
                "rrr"              : rrr,
                "rrr_crr_delta"    : rrr_crr_delta,
            }

            X = pd.DataFrame([payload])[features]
            prob   = model.predict_proba(X)[0]
            bat_wp = prob[1]

            st.markdown("---")
            st.markdown("### 📡 Live Probability")
            win_bar(bat_wp, bat_team, bowl_team)

            if bat_wp >= 0.5:
                verdict = f"**{bat_team}** (batting) in control"
            else:
                verdict = f"**{bowl_team}** (bowling) in control"
            st.info(f"After over {over_num}: {verdict}")

            # Over-by-over simulation
            with st.expander("📈 Simulated win probability curve (this match state)"):
                st.caption("Simulates how probability would evolve if run rate stays constant.")
                overs  = list(range(1, over_num + 1))
                probs  = []

                for ov in overs:
                    frac      = ov / 20
                    sim_runs  = int(cum_runs * (ov / over_num))
                    sim_wkts  = int(cum_wkts * (ov / over_num))
                    sim_req   = max(0, target - sim_runs)
                    sim_balls = 120 - ov * 6
                    sim_wih   = max(0, 10 - sim_wkts)
                    sim_crr   = sim_runs / ov if ov > 0 else 0
                    sim_ov_r  = sim_balls / 6
                    sim_rrr   = sim_req / sim_ov_r if sim_ov_r > 0 else 99.0

                    sim_payload = {
                        "batting_team_enc": bt_enc,
                        "bowling_team_enc": bl_enc,
                        "venue_enc"       : vn_enc,
                        "over_num"        : ov,
                        "cum_runs"        : sim_runs,
                        "cum_wickets"     : sim_wkts,
                        "target"          : target,
                        "runs_required"   : sim_req,
                        "balls_remaining" : sim_balls,
                        "wickets_in_hand" : sim_wih,
                        "crr"             : sim_crr,
                        "rrr"             : sim_rrr,
                        "rrr_crr_delta"   : sim_rrr - sim_crr,
                    }
                    X_sim = pd.DataFrame([sim_payload])[features]
                    p = model.predict_proba(X_sim)[0][1]
                    probs.append(p)

                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_facecolor("#1a1a1a")
                ax.set_facecolor("#1a1a1a")
                ax.plot(overs, probs, color="#ffd700", linewidth=2.5, marker="o",
                        markersize=4)
                ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
                ax.fill_between(overs, probs, 0.5,
                                where=[p >= 0.5 for p in probs],
                                alpha=0.2, color="#ffd700")
                ax.fill_between(overs, probs, 0.5,
                                where=[p < 0.5 for p in probs],
                                alpha=0.2, color="#ff4b4b")
                ax.set_xlabel("Over", color="#888")
                ax.set_ylabel(f"P({bat_team} wins)", color="#888")
                ax.set_ylim(0, 1)
                ax.tick_params(colors="#666")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                ax.set_title(f"{bat_team} chasing {target} · Over-by-over",
                             color="#ccc", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # SHAP
            with st.expander("🔍 Why this probability? (SHAP)"):
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    shap_vals = sv[1][0]
                else:
                    shap_vals = sv[0]
                fig = shap_bar_chart(shap_vals, features,
                                     f"What's driving P({bat_team} wins)")
                st.pyplot(fig)
                plt.close()

# ── Footer ───────────────────────────────────
st.markdown("""
<hr style="border-color:#222;margin-top:3rem">
<div style="text-align:center;color:#444;font-size:12px;padding:1rem 0">
    Built with XGBoost · LightGBM · SHAP · Streamlit &nbsp;·&nbsp; IPL 2008–2024
</div>
""", unsafe_allow_html=True)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Streamlit.ipynb --output app')


# In[ ]:





# In[ ]:




