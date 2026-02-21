"""
Move & Chill — Sensor Analysis Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Move & Chill",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #f5f4f0; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* ── top banner ── */
.banner {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.2rem;
}
.banner h1 { color: #fff; margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
.banner p  { color: #b7e4c7; margin: 0.2rem 0 0; font-size: 0.85rem; }

/* ── metric cards ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.kpi {
    flex: 1; background: #fff; border-radius: 12px;
    padding: 1rem 1.3rem; border-top: 3px solid #2d6a4f;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.kpi .label { font-size: 0.7rem; color: #999; text-transform: uppercase;
              letter-spacing: .08em; margin-bottom: .25rem; }
.kpi .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem;
              font-weight: 600; color: #1a1a1a; }
.kpi .sub   { font-size: 0.72rem; color: #aaa; margin-top: .1rem; }

/* ── section header ── */
.sec { font-size: 1rem; font-weight: 700; color: #1a1a1a;
       border-left: 4px solid #2d6a4f; padding-left: .6rem;
       margin: 1.4rem 0 .9rem; }

/* ── sensor pills ── */
.pill { display:inline-block; border-radius:20px; padding:2px 11px;
        font-family:'IBM Plex Mono',monospace; font-size:.75rem; margin:2px; }
.pill-g  { background:#e9f5ee; color:#2d6a4f; border:1px solid #b7e4c7; }
.pill-r  { background:#fdecea; color:#c0392b; border:1px solid #f5b7b1; }
.pill-y  { background:#fef9e7; color:#b7770d; border:1px solid #fad7a0; }

/* ── tab strip ── */
div[data-baseweb="tab-list"] { gap: 4px; }
div[data-baseweb="tab"]      { font-size:.85rem; padding:.4rem .9rem; border-radius:8px 8px 0 0; }

/* ── sidebar ── */
[data-testid="stSidebar"] { background: #1b4332; }
[data-testid="stSidebar"] * { color: #d8f3dc !important; }
[data-testid="stSidebar"] hr { border-color: #2d6a4f; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label { color: #95d5b2 !important; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def kpi_row(items):
    """items = list of (label, value, sub)"""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

def monday_vlines(ax, full_index):
    for ts in full_index:
        if ts.weekday() == 0 and ts.hour == 0 and ts.minute == 0:
            ax.axvline(ts, color='#1b4332', lw=1.3, ls='--', alpha=.7)

def ellipse_confidence(df, x_col='TSQR', y_col='DModX',
                       confidence=0.95, facecolor='none', edgecolor='darkred'):
    xy  = df[[x_col, y_col]].to_numpy()
    mu  = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order   = eigvals.argsort()[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    chi2_q  = chi2.ppf(confidence, df=2)
    width   = 2. * np.sqrt(eigvals[0] * chi2_q)
    height  = 2. * np.sqrt(eigvals[1] * chi2_q)
    angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ell     = mpatches.Ellipse(xy=mu, width=width, height=height, angle=angle,
                               facecolor=facecolor, edgecolor=edgecolor,
                               alpha=0.12, zorder=1, linewidth=1.)
    inv_cov = np.linalg.pinv(cov)
    diffs   = xy - mu
    d2      = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    outside = list(df.index.astype(str)[d2 > chi2_q])
    return ell, outside


# ══════════════════════════════════════════════════════════════════════════════
# ── DATA PIPELINE  (mirrors your notebook exactly) ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = ['temperature', 'humidity', 'noise', '% occupancy']

@st.cache_data(show_spinner="Loading & processing data …")
def run_pipeline(csv_path: str):
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, index_col='id')
    COLUMN_MAP = {
        'objectid'   : 'ID',
        'sensor_eui' : 'sensor_ID',
        'zeitpunkt'  : 'timestamp',
        'temperature': 'temperature',
        'humidity'   : 'humidity',
        'latitude'   : 'latitude',
        'longitude'  : 'longitude',
        'noise'      : 'noise',
        'sit'        : '% occupancy',
    }
    df.rename(columns=COLUMN_MAP, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df = df.sort_values('timestamp')
    df['date']    = df['timestamp'].dt.date
    df['hour']    = df['timestamp'].dt.hour
    df['slot']    = df['timestamp'].dt.hour * 2 + df['timestamp'].dt.minute // 30

    # ── 2. Coverage ───────────────────────────────────────────────────────────
    n_expected = 48
    coverage   = (
        df.groupby(['date', 'sensor_ID'])['slot']
        .count()
        .unstack(level='sensor_ID')
        .fillna(0)
    )
    dates_cov = pd.to_datetime(list(coverage.index))

    # ── 3. Best Mon→Sun week ──────────────────────────────────────────────────
    coverage_arr = coverage.values
    week_scores  = []
    for i, d in enumerate(dates_cov):
        if d.weekday() != 0 or i + 7 > len(dates_cov):
            continue
        block        = coverage_arr[i:i+7]
        completeness = (block / n_expected).clip(0, 1)
        week_scores.append({
            'start_date'  : dates_cov[i].date(),
            'end_date'    : dates_cov[i+6].date(),
            'mean_coverage': round(completeness.mean(), 3),
            'min_coverage' : round(completeness.min(),  3),
            'score'        : round(completeness.mean()*.5 + completeness.min()*.5, 3),
        })
    weeks_df   = pd.DataFrame(week_scores).sort_values('score', ascending=False)
    best        = weeks_df.iloc[0]
    week_start  = pd.Timestamp(best['start_date'])
    week_end    = week_start + pd.Timedelta(days=7)

    # ── 4. Week subset ────────────────────────────────────────────────────────
    week_df = df[(df['timestamp'] >= week_start) & (df['timestamp'] < week_end)].copy()
    week_df['day_idx']    = week_df['timestamp'].dt.date.apply(
                                lambda d: (pd.Timestamp(d) - week_start).days)
    week_df['slot_in_day']= week_df['timestamp'].dt.hour*2 + week_df['timestamp'].dt.minute//30
    week_df['global_slot']= week_df['day_idx']*48 + week_df['slot_in_day']

    n_slots    = 7 * 48
    full_slots = pd.RangeIndex(n_slots)
    full_index = pd.date_range(start=week_start, periods=n_slots, freq='30min')

    all_sensors = sorted(week_df['sensor_ID'].unique())
    sensor_dfs  = {}
    for sid in all_sensors:
        s = (week_df[week_df['sensor_ID'] == sid]
             .groupby('global_slot')[FEATURES].mean()
             .reindex(full_slots))
        s.index = full_index
        sensor_dfs[sid] = s

    # ── 5. Missingness filter (90 %) ─────────────────────────────────────────
    MISSING_THRESHOLD = 0.90
    dropped_sensors  = []
    reliable_sensors = []
    miss_pct         = {}
    for sid, s in sensor_dfs.items():
        pct = s.isnull().any(axis=1).mean()
        miss_pct[sid] = round(pct*100, 1)
        if pct > MISSING_THRESHOLD:
            dropped_sensors.append(sid)
        else:
            reliable_sensors.append(sid)

    sensor_dfs = {sid: sensor_dfs[sid] for sid in reliable_sensors}

    # ── 6. Fill with -1 ───────────────────────────────────────────────────────
    fill_values = {'temperature': -1, 'humidity': -1, 'noise': -1, '% occupancy': -1}
    sensor_dfs_raw    = {sid: s.copy() for sid, s in sensor_dfs.items()}
    sensor_dfs_filled = {}
    for sid, s in sensor_dfs.items():
        sf = s.copy()
        for feat in FEATURES:
            sf[feat] = sf[feat].fillna(fill_values.get(feat, -1))
        sensor_dfs_filled[sid] = sf

    # ── 7. Build tensor ───────────────────────────────────────────────────────
    n_sensors  = len(reliable_sensors)
    n_features = len(FEATURES)
    X = np.zeros((n_sensors, n_features, n_slots))
    for i, sid in enumerate(reliable_sensors):
        for j, feat in enumerate(FEATURES):
            X[i, j, :] = sensor_dfs_filled[sid][feat].values

    # ── 8. Scale ──────────────────────────────────────────────────────────────
    X_2d     = X.reshape(-1, n_features)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d).reshape(n_sensors, n_features, n_slots)

    # ── 9. KMeans on lat/lon ──────────────────────────────────────────────────
    sensor_coords = (
        week_df[week_df['sensor_ID'].isin(reliable_sensors)]
        .groupby('sensor_ID')[['latitude', 'longitude']].mean()
        .reset_index()
    )
    coords_scaled = StandardScaler().fit_transform(
        sensor_coords[['latitude', 'longitude']].values)
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++', n_init='auto')
    sensor_coords['cluster_kmeans'] = kmeans.fit_predict(coords_scaled)
    sensor_to_cluster = dict(zip(sensor_coords['sensor_ID'], sensor_coords['cluster_kmeans']))

    return dict(
        df=df, week_df=week_df, coverage=coverage, dates_cov=dates_cov,
        weeks_df=weeks_df, week_start=week_start, week_end=week_end,
        full_index=full_index, n_slots=n_slots,
        reliable_sensors=reliable_sensors, dropped_sensors=dropped_sensors,
        miss_pct=miss_pct, sensor_dfs_raw=sensor_dfs_raw,
        sensor_dfs_filled=sensor_dfs_filled,
        X=X, X_scaled=X_scaled,
        sensor_coords=sensor_coords, sensor_to_cluster=sensor_to_cluster,
        n_sensors=n_sensors,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🪑 Move & Chill")
    st.markdown("Smart seat sensor pilot  \nZürich · 2022")
    st.markdown("---")

    csv_path = st.text_input(
        "CSV path",
        value="resources/data/taz.view_moveandchill.csv",
        help="Path to your raw CSV file"
    )
    load_btn = st.button("▶ Load / Reload data", use_container_width=True)

    st.markdown("---")
    st.markdown("**Tucker3 thresholds**")
    TSQR_THRESH  = st.slider("T²R threshold",   50,  2000, 600,  step=50)
    DMODX_THRESH = st.slider("DModX threshold",  1,    30,   6,  step=1)
    CONF_LEVEL   = st.slider("Ellipse confidence", 0.80, 0.99, 0.95, step=0.01)

    st.markdown("---")
    st.markdown("**Inspect**")
    # these will be filled after data loads
    _feat_placeholder   = st.empty()
    _sensor_placeholder = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# ── BANNER ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <div style="font-size:2.5rem">🪑</div>
  <div>
    <h1>Move & Chill — Sensor Dashboard</h1>
    <p>Tucker3 dimensionality reduction · Reliability &amp; robustness analysis · Vulkanplatz &amp; Münsterhof</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if "pipeline" not in st.session_state or load_btn:
    try:
        st.session_state["pipeline"] = run_pipeline(csv_path)
    except FileNotFoundError:
        st.error(f"CSV not found at `{csv_path}`. Update the path in the sidebar.")
        st.stop()

p = st.session_state["pipeline"]

# unpack
df              = p["df"]
week_df         = p["week_df"]
coverage        = p["coverage"]
dates_cov       = p["dates_cov"]
weeks_df        = p["weeks_df"]
week_start      = p["week_start"]
week_end        = p["week_end"]
full_index      = p["full_index"]
n_slots         = p["n_slots"]
reliable_sensors= p["reliable_sensors"]
dropped_sensors = p["dropped_sensors"]
miss_pct        = p["miss_pct"]
sensor_dfs_raw  = p["sensor_dfs_raw"]
sensor_dfs_filled = p["sensor_dfs_filled"]
X               = p["X"]
X_scaled        = p["X_scaled"]
sensor_coords   = p["sensor_coords"]
sensor_to_cluster = p["sensor_to_cluster"]
n_sensors       = p["n_sensors"]

# sidebar dropdowns now that we have sensor list
sel_feature = _feat_placeholder.selectbox("Feature", FEATURES)
sel_sensor  = _sensor_placeholder.selectbox("Sensor", reliable_sensors,
                                             format_func=lambda x: x[-6:])  # shorten long EUI


# ══════════════════════════════════════════════════════════════════════════════
# ── TOP KPIs ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
kpi_row([
    ("Total sensors",    len(reliable_sensors)+len(dropped_sensors), f"{len(dropped_sensors)} dropped"),
    ("Reliable sensors", len(reliable_sensors), "passed 90% threshold"),
    ("Analysis week",    str(week_start.date()), f"→ {str((week_end-pd.Timedelta(days=1)).date())}"),
    ("Time slots",       n_slots, "7 days × 48 slots"),
    ("Locations",        "2", "Vulkanplatz · Münsterhof"),
])

# sensor pills in sidebar
with st.sidebar:
    st.markdown("**Reliable sensors**")
    for sid in reliable_sensors:
        cl  = sensor_to_cluster.get(sid, '?')
        loc = "Vulkpl." if cl == 0 else "Münst."
        pct = miss_pct.get(sid, 0)
        cls = "pill-y" if pct > 10 else "pill-g"
        st.markdown(f'<span class="pill {cls}">🟢 …{sid[-6:]} {loc} {pct}%</span>',
                    unsafe_allow_html=True)
    if dropped_sensors:
        st.markdown("**Dropped sensors**")
        for sid in dropped_sensors:
            st.markdown(f'<span class="pill pill-r">⚠️ …{sid[-6:]}</span>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── TABS ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
tab_data, tab_map, tab_cov, tab_sig, tab_tucker, tab_diag = st.tabs([
    "🗃️ Dataset",
    "🗺️ Sensor map",
    "📡 Coverage",
    "📈 Signals",
    "🔲 Tucker3 factors",
    "🎯 Diagnostic",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    sec("Raw dataset explorer")

    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        sel_sids = st.multiselect("Filter sensors", df['sensor_ID'].unique().tolist(),
                                   default=df['sensor_ID'].unique().tolist(),
                                   format_func=lambda x: x[-8:])
    with cf2:
        date_range = st.date_input("Date range",
                                    value=(df['timestamp'].min().date(),
                                           df['timestamp'].max().date()))
    with cf3:
        show_cols = st.multiselect("Columns",
                                    ['timestamp','sensor_ID'] + FEATURES + ['latitude','longitude'],
                                    default=['timestamp','sensor_ID'] + FEATURES)

    mask = df['sensor_ID'].isin(sel_sids)
    if len(date_range) == 2:
        mask &= df['timestamp'].dt.date.between(date_range[0], date_range[1])
    df_view = df.loc[mask, show_cols].reset_index(drop=True)

    kpi_row([
        ("Rows",          f"{len(df_view):,}",             "filtered"),
        ("Sensors",       df_view['sensor_ID'].nunique() if 'sensor_ID' in df_view else "—", "unique EUIs"),
        ("Missing cells", f"{df_view.isnull().sum().sum():,}", "NaN count"),
        ("Days",          df_view['timestamp'].dt.date.nunique() if 'timestamp' in df_view else "—", "in selection"),
    ])

    st.dataframe(df_view.style.highlight_null(color='#fdecea'),
                 use_container_width=True, height=400)

    st.download_button("⬇️ Download CSV", df_view.to_csv(index=False).encode(),
                       "move_chill_filtered.csv", "text/csv")

    with st.expander("📊 Descriptive statistics"):
        num_cols = [c for c in show_cols if c in FEATURES]
        if num_cols:
            st.dataframe(df_view[num_cols].describe().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    sec("Sensor locations — Vulkanplatz & Münsterhof")

    sc2 = sensor_coords.copy()
    sc2['label']    = sc2['sensor_ID'].str[-8:]
    sc2['location'] = sc2['cluster_kmeans'].map({0: 'Vulkanplatz', 1: 'Münsterhof'})
    sc2['missing']  = sc2['sensor_ID'].map(lambda s: miss_pct.get(s, 100))
    sc2['status']   = sc2['sensor_ID'].apply(
                          lambda s: 'Dropped' if s in dropped_sensors else 'Active')

    fig_map = px.scatter_map(
        sc2, lat="latitude", lon="longitude",
        color="location",
        color_discrete_map={"Vulkanplatz": "#2d6a4f", "Münsterhof": "#457b9d"},
        hover_name="label",
        hover_data={"missing": ":.1f", "status": True,
                    "latitude": ":.5f", "longitude": ":.5f"},
        size=[14]*len(sc2),
        zoom=14, height=480,
        title="Sensor cluster map"
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    col_leg, col_tbl = st.columns([1, 2])
    with col_leg:
        sec("Legend")
        for cl, name, color in [(0,"Vulkanplatz","🟢"),(1,"Münsterhof","🔵")]:
            sids = [s for s,c in sensor_to_cluster.items() if c == cl]
            st.markdown(f"{color} **{name}** — {len(sids)} sensor(s)")
        if dropped_sensors:
            st.markdown(f"🔴 **Dropped** — {len(dropped_sensors)} sensor(s)")
    with col_tbl:
        sec("Sensor coordinate table")
        st.dataframe(
            sc2[['label','location','latitude','longitude','missing','status']]
              .rename(columns={'label':'sensor (short)','missing':'missing %'}),
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_cov:
    sec("Full data coverage — all sensors over entire period")

    fig, ax = plt.subplots(figsize=(15, max(3, len(coverage.columns)*0.45)))
    im = ax.imshow(coverage.T / 48 * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    for i, d in enumerate(dates_cov):
        if d.weekday() == 0:
            ax.axvline(i-.5, color='#1b4332', lw=1.4, ls='--', alpha=.7)
            ax.text(i, -0.9, 'Mon', fontsize=6, ha='center', color='#1b4332')
    ax.set_yticks(range(len(coverage.columns)))
    ax.set_yticklabels([s[-8:] for s in coverage.columns], fontsize=7)
    ax.set_xticks(range(0, len(dates_cov), 7))
    ax.set_xticklabels([str(d.date()) for d in dates_cov[::7]], rotation=45, fontsize=7)
    plt.colorbar(im, ax=ax, label='% slots received', shrink=.7)
    ax.set_title('Daily slot coverage per sensor (Monday markers)', fontweight='bold', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    sec(f"Best week selected: {week_start.date()} → {(week_end-pd.Timedelta(days=1)).date()}")
    kpi_row([
        ("Mean coverage", f"{weeks_df.iloc[0]['mean_coverage']*100:.1f}%", "best week"),
        ("Min coverage",  f"{weeks_df.iloc[0]['min_coverage']*100:.1f}%",  "worst sensor/day"),
        ("Sensors kept",  len(reliable_sensors), f">{int((1-0.9)*100)}% present"),
        ("Sensors dropped", len(dropped_sensors), ">90% missing"),
    ])

    with st.expander("Week score table"):
        st.dataframe(weeks_df, use_container_width=True, hide_index=True)

    # Best week zoom
    best_mask  = (coverage.index >= weeks_df.iloc[0]['start_date']) & \
                 (coverage.index <= weeks_df.iloc[0]['end_date'])
    best_block = coverage.loc[best_mask]
    best_dates = pd.to_datetime(list(best_block.index))
    rel_cols   = [c for c in best_block.columns if c in reliable_sensors]

    fig2, ax2 = plt.subplots(figsize=(10, max(2.5, len(rel_cols)*0.4)))
    im2 = ax2.imshow(best_block[rel_cols].T / 48 * 100,
                     aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    for i, d in enumerate(best_dates):
        if d.weekday() == 0:
            ax2.axvline(i-.5, color='#1b4332', lw=2, ls='--')
            ax2.text(i, -0.7, 'Mon', fontsize=8, ha='center', fontweight='bold', color='#1b4332')
    ax2.set_yticks(range(len(rel_cols)))
    ax2.set_yticklabels([s[-8:] for s in rel_cols], fontsize=8)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels([d.strftime('%a %d %b') for d in best_dates], rotation=40, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='% slots', shrink=.8)
    ax2.set_title('Zoom — best week (reliable sensors only)', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_sig:
    sec(f"Signal: {sel_feature} — sensor …{sel_sensor[-8:]}")

    raw    = sensor_dfs_raw[sel_sensor][sel_feature]
    filled = sensor_dfs_filled[sel_sensor][sel_feature]
    gap    = raw.isnull()

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(full_index, filled, color='#2d6a4f', lw=0.9, label=sel_feature)
    ax.fill_between(full_index, filled.min(), filled.max(),
                    where=gap, color='#e74c3c', alpha=0.25, label='missing → filled –1')
    monday_vlines(ax, full_index)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.set_ylabel(sel_feature); ax.legend(fontsize=8)
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.set_title(f'{sel_feature} — …{sel_sensor[-8:]}', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec(f"All sensors — {sel_feature}")
    n = len(reliable_sensors)
    fig2, axes = plt.subplots(n, 1, figsize=(14, 2.6*n), sharex=True)
    if n == 1: axes = [axes]
    colors_cycle = plt.cm.tab10.colors
    for ax, sid in zip(axes, reliable_sensors):
        col  = colors_cycle[reliable_sensors.index(sid) % 10]
        fil  = sensor_dfs_filled[sid][sel_feature]
        raw2 = sensor_dfs_raw[sid][sel_feature]
        ax.plot(full_index, fil, color=col, lw=0.85)
        ax.fill_between(full_index, fil.min(), fil.max(),
                        where=raw2.isnull(), color='#e74c3c', alpha=0.2)
        monday_vlines(ax, full_index)
        ax.set_ylabel(f'…{sid[-6:]}', fontsize=8, rotation=0, labelpad=55, va='center')
        ax.grid(axis='y', ls=':', alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    fig2.suptitle(f'{sel_feature} — all reliable sensors', fontweight='bold', fontsize=11)
    plt.tight_layout(); st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TUCKER3 FACTORS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tucker:

    # ── Check if Tucker3 results have been injected ───────────────────────────
    if "tucker_results" not in st.session_state:
        st.info("""
        **Tucker3 results not loaded yet.**

        After running your Tucker3 model in Python, call:
        ```python
        import streamlit as st
        st.session_state['tucker_results'] = {
            'A'      : results['scores'],        # (n_sensors, r1)
            'B'      : results['varLoadings'],    # (n_features, r2)
            'C'      : results['timeLoadings'],   # (n_slots, r3)
            'F'      : results['coreTensor'],
            'df_out' : df_out,                   # DataFrame with HT2, DModX, TSQR …
            'df_iSPE': df_iSPE,
        }
        ```
        Or paste your results below ↓
        """)

        with st.expander("📥 Paste numpy arrays as JSON (optional quick-load)"):
            st.code("""
# In your notebook, export like this:
import json, numpy as np
print(json.dumps({
  'A': results['scores'].tolist(),
  'B': results['varLoadings'].tolist(),
  'C': results['timeLoadings'].tolist(),
}))
            """)
        st.stop()

    tr = st.session_state["tucker_results"]
    A, B, C = tr['A'], tr['B'], tr['C']
    df_out  = tr['df_out']
    df_iSPE = tr.get('df_iSPE', None)

    short_names = [s[-8:] for s in reliable_sensors]

    # ── Factor plots (A, B, C) ────────────────────────────────────────────────
    sec("Tucker3 factor matrices")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(A, cmap='RdBu_r', aspect='auto')
        ax.set_yticks(range(len(reliable_sensors)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xticks(range(A.shape[1]))
        ax.set_xticklabels([f'SC{i+1}' for i in range(A.shape[1])])
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title('A — Sensor loadings', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(B, cmap='RdBu_r', aspect='auto')
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(FEATURES, fontsize=8)
        ax.set_xticks(range(B.shape[1]))
        ax.set_xticklabels([f'FC{i+1}' for i in range(B.shape[1])])
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title('B — Feature loadings', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("C — Temporal factor matrix")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    pal = ['#2d6a4f','#e76f51','#457b9d','#e9c46a','#9b5de5','#f15bb5']
    for k in range(C.shape[1]):
        ax.plot(full_index, C[:, k], lw=0.9, color=pal[k % len(pal)], label=f'TC{k+1}')
    monday_vlines(ax, full_index)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.legend(fontsize=8); ax.grid(ls=':', alpha=0.3)
    ax.set_title('Temporal components over the selected week', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Score line plot (mirrors notebook cell 22) ────────────────────────────
    sec("Factor matrix line plots")
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    axes[0].plot(A, 'o-', lw=1.4, ms=5)
    axes[0].set_xticks(range(len(reliable_sensors)))
    axes[0].set_xticklabels(short_names, rotation=40, ha='right', fontsize=7)
    axes[0].set_ylabel('Scores'); axes[0].grid(True, alpha=.3)

    axes[1].plot(B, 'o-', lw=1.4, ms=5)
    axes[1].set_xticks(range(len(FEATURES)))
    axes[1].set_xticklabels(FEATURES, rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel('Var Loadings'); axes[1].grid(True, alpha=.3)

    axes[2].plot(C[:48], 'o-', lw=1, ms=3)   # first day for readability
    axes[2].set_ylabel('Time Loading (day 1)'); axes[2].grid(True, alpha=.3)
    axes[2].set_xlabel('Slot (30 min)')

    fig.suptitle('Tucker3 — Factor matrix line plots', fontweight='bold', fontsize=13)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── 3-D scatter LV1/LV2/LV3 coloured by DModX ────────────────────────────
    if all(c in df_out.columns for c in ['LV1','LV2','LV3','DModX']):
        sec("3-D score space")
        fig3d = px.scatter_3d(
            df_out.reset_index().rename(columns={'index':'sensor'}),
            x='LV1', y='LV2', z='LV3',
            color='DModX', size_max=12,
            color_continuous_scale='RdBu_r',
            text=df_out.index.str[-8:],
            hover_name=df_out.index.str[-8:],
            title='Tucker3 score space  (LV1 × LV2 × LV3, coloured by DModX)',
            height=550,
        )
        fig3d.update_traces(marker=dict(line=dict(width=0.5, color='black')))
        fig3d.update_layout(margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig3d, use_container_width=True)

    # ── iSPE bar charts (mirrors notebook cell 26) ────────────────────────────
    if df_iSPE is not None:
        sec("Individual SPE contributions")
        s1 = st.selectbox("Sensor A", reliable_sensors, key='spe1',
                          format_func=lambda x: x[-8:])
        s2 = st.selectbox("Sensor B", reliable_sensors,
                          index=min(1, len(reliable_sensors)-1), key='spe2',
                          format_func=lambda x: x[-8:])
        c1, c2 = st.columns(2)
        for col, sid in zip([c1, c2], [s1, s2]):
            with col:
                fig, ax = plt.subplots(figsize=(6, 4))
                vals = df_iSPE.loc[sid]
                sns.barplot(x=vals.index, y=vals.values, palette='viridis', ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
                ax.set_title(f'iSPE — …{sid[-8:]}', fontweight='bold')
                ax.set_ylabel('Contribution')
                plt.tight_layout(); col.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTIC MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    if "tucker_results" not in st.session_state:
        st.info("Load Tucker3 results first (see Tucker3 factors tab).")
    else:
        tr     = st.session_state["tucker_results"]
        df_out = tr['df_out']

        sec("Sensor diagnostic map — T²R vs DModX")

        x = df_out['TSQR']
        y = df_out['DModX']
        v = df_out.index.astype(str).to_numpy()

        x_max = max(x.max()*1.18, TSQR_THRESH*1.35)
        y_max = max(y.max()*1.18, DMODX_THRESH*1.35)

        fig, ax = plt.subplots(figsize=(11, 8))

        # quadrant fills
        ax.fill_betweenx([0, DMODX_THRESH], 0, TSQR_THRESH,
                         color='#2ecc71', alpha=0.10)
        ax.fill_betweenx([0, DMODX_THRESH], TSQR_THRESH, x_max,
                         color='#f39c12', alpha=0.10)
        ax.fill_betweenx([DMODX_THRESH, y_max], 0, TSQR_THRESH,
                         color='#3498db', alpha=0.10)
        ax.fill_betweenx([DMODX_THRESH, y_max], TSQR_THRESH, x_max,
                         color='#e74c3c', alpha=0.10)

        # threshold lines
        ax.axvline(TSQR_THRESH, color='#c0392b', lw=1.5, ls='--',
                   label=f'T²R = {TSQR_THRESH}')
        ax.axhline(DMODX_THRESH, color='#2980b9', lw=1.5, ls='--',
                   label=f'DModX = {DMODX_THRESH}')

        # quadrant labels
        lbl = dict(fontsize=9, fontweight='bold', alpha=0.55, ha='center', va='center')
        ax.text(TSQR_THRESH*.5, DMODX_THRESH*.5,
                '✅ In Control\nLow T² | Low DModX', color='#27ae60', **lbl)
        ax.text(TSQR_THRESH+(x_max-TSQR_THRESH)*.5, DMODX_THRESH*.5,
                '⚠️ Score Outlier\nHigh T² | Low DModX', color='#e67e22', **lbl)
        ax.text(TSQR_THRESH*.5, DMODX_THRESH+(y_max-DMODX_THRESH)*.5,
                '🔵 Residual Outlier\nLow T² | High DModX', color='#2980b9', **lbl)
        ax.text(TSQR_THRESH+(x_max-TSQR_THRESH)*.5, DMODX_THRESH+(y_max-DMODX_THRESH)*.5,
                '🔴 Critical Outlier\nHigh T² | High DModX', color='#c0392b', **lbl)

        # confidence ellipse
        ell, outside_ids = ellipse_confidence(df_out, confidence=CONF_LEVEL,
                                              facecolor='grey', edgecolor='#555')
        ax.add_patch(ell)

        # scatter
        sc = ax.scatter(x, y, c=x, cmap='RdBu_r', alpha=0.85,
                        edgecolors='k', linewidths=0.5, zorder=5, s=90)
        for xi, yi, vi in zip(x, y, v):
            ax.text(xi + x_max*.01, yi + y_max*.01, vi[-8:],
                    fontsize=7, alpha=0.9, zorder=6)

        plt.colorbar(sc, ax=ax, label='T²R value', shrink=0.5)
        ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
        ax.set_xlabel('T²R  (Hotelling) — distance within model space', labelpad=10)
        ax.set_ylabel('DModX — residual distance outside model space', labelpad=10)
        ax.set_title('Sensor Diagnostic Map — T²R vs DModX', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, framealpha=.8)
        ax.grid(True, ls=':', alpha=0.35)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # outside summary
        if outside_ids:
            st.warning(f"**{len(outside_ids)} sensor(s) outside the {CONF_LEVEL:.0%} "
                       f"confidence ellipse:** {', '.join([s[-8:] for s in outside_ids])}")
        else:
            st.success(f"All sensors inside the {CONF_LEVEL:.0%} confidence ellipse ✅")

        sec("Diagnostic table")
        diag_cols = [c for c in ['TSQR','DModX','HT2','T_SPE'] if c in df_out.columns]
        st.dataframe(
            df_out[diag_cols].style
                .background_gradient(subset=['TSQR'], cmap='RdYlGn_r')
                .background_gradient(subset=['DModX'], cmap='RdYlGn_r')
                .format("{:.3f}"),
            use_container_width=True
        )