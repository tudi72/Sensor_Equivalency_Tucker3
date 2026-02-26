"""
Move & Chill — Sensor Analysis Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

import matplotlib.cm as cm 
import matplotlib.colors as mcolors
# ══════════════════════════════════════════════════════════════════════════════
# TUCKER3 — embedded from tucker3.py
# ══════════════════════════════════════════════════════════════════════════════
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

class Tucker3:
    """
    Tucker3 decomposition with Hotelling T², DModX and SPE diagnostics.
    Mirrors the notebook interface: fit_transform / transform / calcIMD / calcOOMD.
    """
    def __init__(self, LowRankApprox=np.array([3, 3, 3]),
                 maxIter=100, tolCriteria=0.02, ErrorScaled=True):
        self.rank        = list(LowRankApprox)
        self.maxIter     = maxIter
        self.tolCriteria = tolCriteria
        self.ErrorScaled = ErrorScaled
        self._X = self._A = self._B = self._C = self._core = None

    def fit_transform(self, X: np.ndarray, verbose=True):
        T             = tl.tensor(X.astype(float))
        core, factors = tl_tucker(T, rank=self.rank)
        self._core    = np.array(core)
        self._A       = np.array(factors[0])   # sensors
        self._B       = np.array(factors[1])   # features
        self._C       = np.array(factors[2])   # time
        self._X       = X

        if verbose:
            recon = tl.tucker_to_tensor((core, factors))
            err   = np.linalg.norm(X - np.array(recon)) / (np.linalg.norm(X) + 1e-12)
            st.write(f"Tucker3 relative reconstruction error: **{err:.4f}**")

        return {
            'scores'      : self._A,
            'varLoadings' : self._B,
            'timeLoadings': self._C,
            'coreTensor'  : self._core,
        }

    def transform(self, X: np.ndarray, verbose=False):
        recon = tl.tucker_to_tensor(
            (tl.tensor(self._core),
             [tl.tensor(f) for f in [self._A, self._B, self._C]])
        )
        return None, np.array(recon), None

    def calcIMD(self, input: np.ndarray, metric="HotellingT2"):
        """Hotelling T² per sensor (axis 0), normalised by component std."""
        scores = self._A / (self._A.std(axis=0) + 1e-12)
        return np.sum(scores ** 2, axis=1)

    def calcOOMD(self, input: np.ndarray, metric="DModX"):
        """DModX — RMS residual per sensor across features × time."""
        recon = tl.tucker_to_tensor(
            (tl.tensor(self._core),
             [tl.tensor(f) for f in [self._A, self._B, self._C]])
        )
        resid = input - np.array(recon)
        return np.sqrt((resid ** 2).mean(axis=(1, 2)))

    def calcContribution(self, XT_scaled):
        """TSQR, T_SPE, per-feature con_TSQR, and iSPE_batch."""
        scores   = self._A / (self._A.std(axis=0) + 1e-12)
        TSQR     = np.sum(scores ** 2, axis=1)

        recon    = tl.tucker_to_tensor(
            (tl.tensor(self._core),
             [tl.tensor(f) for f in [self._A, self._B, self._C]])
        )
        resid    = self._X - np.array(recon)
        T_SPE    = (resid ** 2).sum(axis=(1, 2))

        n_feat   = self._B.shape[0]
        con_TSQR = np.zeros((len(self._A), n_feat))
        for j in range(n_feat):
            con_TSQR[:, j] = np.abs(scores @ self._B[j, :])

        iSPE = (resid ** 2).mean(axis=2)   # (sensors, features)

        return {'TSQR': TSQR, 'T_SPE': T_SPE,
                'con_TSQR': con_TSQR, 'iSPE_batch': iSPE}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Move & Chill", page_icon="🪑",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
for _key, _val in {
    "tucker_results": None,
    "pipeline":       None,
    "week_data":      None,
    "chosen_start":   None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }
.main{ background:#f5f4f0; }
.block-container{ padding-top:1.8rem; padding-bottom:2rem; }
.banner{ background:linear-gradient(135deg,#1b4332 0%,#2d6a4f 60%,#40916c 100%);
         border-radius:14px; padding:1.6rem 2rem; margin-bottom:1.5rem;
         display:flex; align-items:center; gap:1.2rem; }
.banner h1{ color:#fff; margin:0; font-size:1.8rem; font-weight:700; }
.banner p { color:#b7e4c7; margin:.2rem 0 0; font-size:.85rem; }
.kpi{ background:#fff; border-radius:12px; padding:1rem 1.3rem;
      border-top:3px solid #2d6a4f; box-shadow:0 2px 8px rgba(0,0,0,.06); }
.kpi .label{ font-size:.7rem; color:#999; text-transform:uppercase;
             letter-spacing:.08em; margin-bottom:.25rem; }
.kpi .value{ font-family:'IBM Plex Mono',monospace; font-size:1.55rem;
             font-weight:600; color:#1a1a1a; }
.kpi .sub  { font-size:.72rem; color:#aaa; margin-top:.1rem; }
.sec{ font-size:1rem; font-weight:700; color:#1a1a1a;
      border-left:4px solid #2d6a4f; padding-left:.6rem; margin:1.4rem 0 .9rem; }
.pill{ display:inline-block; border-radius:20px; padding:2px 11px;
       font-family:'IBM Plex Mono',monospace; font-size:.75rem; margin:2px; }
.pill-g{ background:#e9f5ee; color:#2d6a4f; border:1px solid #b7e4c7; }
.pill-r{ background:#fdecea; color:#c0392b; border:1px solid #f5b7b1; }
.pill-y{ background:#fef9e7; color:#b7770d; border:1px solid #fad7a0; }
[data-testid="stSidebar"]  { background:#1b4332; }
[data-testid="stSidebar"] *{ color:#d8f3dc !important; }
[data-testid="stSidebar"] hr{ border-color:#2d6a4f; }
</style>
""", unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                     f'<div class="value">{value}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def monday_vlines(ax, full_index):
    for ts in full_index:
        if ts.weekday() == 0 and ts.hour == 0 and ts.minute == 0:
            ax.axvline(ts, color='#1b4332', lw=1.3, ls='--', alpha=.7)

FEATURES = ['temperature', 'humidity', 'noise', '% occupancy']

@st.cache_data(show_spinner="Loading & processing data …")
def run_pipeline(csv_path: str):
    df  = pd.read_csv(csv_path, index_col='id')
    df.rename(columns={
                    'objectid'      : 'ID',
                    'sensor_eui'    : 'sensor_ID',
                    'zeitpunkt'     : 'timestamp',
                    'temperature'   : 'temperature',
                    'humidity'      : 'humidity',
                    'latitude'      : 'latitude',
                    'longitude'     : 'longitude',
                    'noise'         : 'noise',
                    'sit'           : '% occupancy',
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df              = df.sort_values('timestamp')
    df['date']      = df['timestamp'].dt.date
    df['slot']      = df['timestamp'].dt.hour*2 + df['timestamp'].dt.minute//30

    coverage        = (df.groupby(['date','sensor_ID'])['slot'].count().unstack('sensor_ID').fillna(0))
    dates_cov       = pd.to_datetime(list(coverage.index))
    cov_arr,\
    week_scores     = coverage.values, []

    for i, d in enumerate(dates_cov):
        if d.weekday() != 0 or i+7 > len(dates_cov): 
            continue
        block        = cov_arr[i:i+7]
        completeness = (block/48).clip(0, 1)
        week_scores.append({
            'start_date'   : dates_cov[i].date(),
            'end_date'     : dates_cov[i+6].date(),
            'start_idx'    : i,
            'mean_coverage': round(completeness.mean(), 3),
            'min_coverage' : round(completeness.min(),  3),
            'score'        : round(completeness.mean()*.5 + completeness.min()*.5, 3),
        })
    weeks_df        = (pd.DataFrame(week_scores).sort_values('score', ascending=False).reset_index(drop=True))
    best_start      = pd.Timestamp(weeks_df.iloc[0]['start_date'])
    sensor_coords   = (df.groupby('sensor_ID')[['latitude','longitude']].mean().reset_index())
    cs              = StandardScaler().fit_transform(sensor_coords[['latitude','longitude']].values)
    sensor_coords['cluster_kmeans'] = (KMeans(n_clusters=2, random_state=42,init='k-means++', n_init='auto').fit_predict(cs))
    sensor_to_cluster = dict(zip(sensor_coords['sensor_ID'],sensor_coords['cluster_kmeans']))

    return dict(df              = df, 
                coverage        = coverage, 
                dates_cov       = dates_cov,
                weeks_df        = weeks_df, 
                best_start      = best_start,
                sensor_coords   = sensor_coords, 
                sensor_to_cluster=sensor_to_cluster)

def build_week_data(df, week_start, missing_threshold=0.90):

    all_sensors         = df['sensor_ID'].unique().tolist()
    week_end            = week_start + pd.Timedelta(days=7)
    wdf                 = df[(df['timestamp'] >= week_start) & (df['timestamp'] < week_end)].copy()
    wdf['day_idx']      = wdf['timestamp'].dt.date.apply(lambda d: (pd.Timestamp(d)-week_start).days)
    wdf['slot_in_day']  = wdf['timestamp'].dt.hour*2 + wdf['timestamp'].dt.minute//30
    wdf['global_slot']  = wdf['day_idx']*48 + wdf['slot_in_day']

    n_slots             = 336
    full_index          = pd.date_range(start=week_start, periods=n_slots, freq='30min')
    sensor_dfs          = {}

    for sid in sorted(wdf['sensor_ID'].unique()):
        s               = (wdf[wdf['sensor_ID']==sid].groupby('global_slot')[FEATURES].mean().reindex(pd.RangeIndex(n_slots)))
        s.index         = full_index
        sensor_dfs[sid] = s

    dropped, reliable, miss_pct = [], [], {}
    for sid, s in sensor_dfs.items():
        pct = s.isnull().any(axis=1).mean()
        miss_pct[sid] = round(pct*100, 1)
        (dropped if pct > missing_threshold else reliable).append(sid)

    sensor_dfs     = {sid: sensor_dfs[sid] for sid in reliable}
    raw            = {sid: s.copy() for sid, s in sensor_dfs.items()}
    filled         = {sid: s.fillna(-1) for sid, s in sensor_dfs.items()}

    n = len(reliable)
    X = np.zeros((n, len(FEATURES), n_slots))
    for i, sid in enumerate(reliable):
        for j, feat in enumerate(FEATURES):
            X[i, j, :] = filled[sid][feat].values
    X_scaled = StandardScaler().fit_transform(
        X.reshape(-1, len(FEATURES))).reshape(n, len(FEATURES), n_slots)

    return dict(full_index=full_index, n_slots=n_slots,
                reliable_sensors=reliable, dropped_sensors=dropped,
                miss_pct=miss_pct, sensor_dfs_raw=raw, sensor_dfs_filled=filled,
                X=X, X_scaled=X_scaled, n_sensors=n,all_sensors=all_sensors)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Move & Chill")
    st.markdown("Smart seat sensors · Zürich · 2022")
    st.markdown("---")
    csv_path = st.text_input("CSV path",value="resources/data/taz.view_moveandchill.csv")
    load_btn = st.button("▶ Load / Reload data", use_container_width=True)
    st.markdown("""
            <style>
            input[type="password"] {
                color: black !important;
                -webkit-text-security: disc;
            }
            label[data-testid="stWidgetLabel"] p {
                color: black !important;
            }
            </style>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Tucker3 settings**")
    rank_sensors  = st.slider("Rank — sensors",  1, 6, 3)
    rank_features = st.slider("Rank — features", 1, 4, 3)
    rank_time     = st.slider("Rank — time",     1, 8, 3)
    max_iter      = st.slider("Max iterations",  10, 500, 100, step=10)
    tol           = st.number_input("Tolerance", value=0.02, format="%.4f")
    run_tucker_btn= st.button("⚡ Run Tucker3", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Diagnostic thresholds**")
    TSQR_THRESH  = st.slider("T²R threshold",     10, 600, 300,  step=10)
    DMODX_THRESH = st.slider("DModX threshold",    0.1, 10.0, 1.0, step=0.1)
    st.markdown("---")
    _feat_ph   = st.empty()
    _sensor_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════════════════════════════════════
import base64 
from pathlib import Path
def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()
    
img_b64 = img_to_b64(Path(__file__).parent / 'resources' / 'image_bench.avif')

st.markdown(f"""
<div class="banner">
  <div style="font-size:2.5rem">
    <img src="data:image/avif;base64,{img_b64}" style="height:60px; width:auto; border-radius:8px;">
  </div>
  <div>
    <h1>Move & Chill — Sensor Dashboard</h1>
    <p>Tucker3 dimensionality reduction · Reliability &amp; robustness · Vulkanplatz &amp; Münsterhof</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if "pipeline" not in st.session_state or load_btn:
    try:
        st.session_state["pipeline"] = run_pipeline(csv_path)
        st.session_state.pop("week_data",      None)
        st.session_state.pop("tucker_results", None)
        st.session_state.pop("chosen_start",   None)
    except FileNotFoundError:
        st.error(f"CSV not found at `{csv_path}`."); st.stop()

p = st.session_state["pipeline"]

if p is None:
    st.info("👆 Enter the CSV path in the sidebar and click **▶ Load / Reload data** to begin.")
    st.stop()

df            = p["df"]
coverage      = p["coverage"]
dates_cov     = p["dates_cov"]
weeks_df      = p["weeks_df"]
sensor_coords = p["sensor_coords"]
sensor_to_cluster = p["sensor_to_cluster"]


# ══════════════════════════════════════════════════════════════════════════════
# WEEK SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
sec("Select analysis week")

week_labels = []
for _, r in weeks_df.iterrows():
    star = " ★" if r['start_date'] == weeks_df.iloc[0]['start_date'] else ""
    week_labels.append(
        f"{r['start_date']} → {r['end_date']}  "
        f"| coverage {r['mean_coverage']*100:.0f}%  "
    )

wc1, wc2 = st.columns([4, 1])
with wc1:
    sel_week_label = st.selectbox(
        "Mon → Sun week  (★ = auto best)",
        options=week_labels, index=0)
sel_idx      = week_labels.index(sel_week_label)
chosen_row   = weeks_df.iloc[sel_idx]
chosen_start = pd.Timestamp(chosen_row['start_date'])
chosen_end   = pd.Timestamp(chosen_row['end_date'])
is_best_week = (sel_idx == 0)

with wc2:
    st.markdown("<br>", unsafe_allow_html=True)
    if is_best_week:
        st.success("★ Best")
    else:
        st.info("Custom")

# Rebuild tensor when week changes
if ("week_data" not in st.session_state or
        st.session_state.get("chosen_start") != chosen_start):
    with st.spinner("Building week tensor …"):
        st.session_state["week_data"]    = build_week_data(df, chosen_start)
        st.session_state["chosen_start"] = chosen_start
        st.session_state.pop("tucker_results", None)

wd = st.session_state["week_data"]
reliable_sensors  = wd["reliable_sensors"]
all_sensors       = wd["all_sensors"]
dropped_sensors   = wd["dropped_sensors"]
miss_pct          = wd["miss_pct"]
sensor_dfs_raw    = wd["sensor_dfs_raw"]
sensor_dfs_filled = wd["sensor_dfs_filled"]
full_index        = wd["full_index"]
n_slots           = wd["n_slots"]
X, X_scaled       = wd["X"], wd["X_scaled"]
n_sensors         = wd["n_sensors"]

sel_feature = _feat_ph.selectbox("Feature", FEATURES)
sel_sensor  = _sensor_ph.selectbox("Sensor",  reliable_sensors,
                                    format_func=lambda x: f"…{x[-8:]}")

with st.sidebar:
    st.markdown(f"**Week** `{chosen_start.date()}` → `{chosen_end.date()}`")
    for sid in reliable_sensors:
        cl  = sensor_to_cluster.get(sid, 0)
        loc = "Vulkpl." if cl==0 else "Münst."
        pct = miss_pct.get(sid, 0)
        cls = "pill-y" if pct > 10 else "pill-g"
        st.markdown(f'<span class="pill {cls}">🟢 …{sid[-6:]} {loc} {pct}%</span>',
                    unsafe_allow_html=True)
    if dropped_sensors:
        st.markdown("**Dropped**")
        for sid in dropped_sensors:
            st.markdown(f'<span class="pill pill-r">⚠️ …{sid[-6:]}</span>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN TUCKER3
# ══════════════════════════════════════════════════════════════════════════════
if run_tucker_btn:
    with st.spinner("Running Tucker3 …"):
        model   = Tucker3(
            LowRankApprox=np.array([rank_sensors, rank_features, rank_time]),
            maxIter=max_iter, tolCriteria=tol, ErrorScaled=True)
        results         = model.fit_transform(X_scaled, verbose=False)
        _, XT_scaled, _ = model.transform(X_scaled, verbose=False)
        metrics         = model.calcContribution(XT_scaled)

        A = results['scores']
        B = results['varLoadings']
        C = results['timeLoadings']

        HT2   = model.calcIMD(X_scaled).reshape(n_sensors, 1)
        DModX = model.calcOOMD(X_scaled).reshape(n_sensors, 1)

        df_out              = pd.DataFrame(np.hstack((HT2, DModX)),columns=["HT2","DModX"], index=reliable_sensors)
        df_out['T_SPE']     = metrics['T_SPE']
        df_out['TSQR']      = metrics['TSQR']
        df_out['cluster']   = df_out.index.map(sensor_to_cluster).fillna(100)
        new_cols            = [f"con_TSQR_{f}" for f in FEATURES]
        df_out[new_cols]    = metrics['con_TSQR']
        df_iSPE             = pd.DataFrame(metrics['iSPE_batch'],columns=FEATURES, index=reliable_sensors)
        for i in range(A.shape[1]):
            df_out[f'LV{i+1}'] = A[:, i]

        st.session_state["tucker_results"] = dict(
            A=A, B=B, C=C, df_out=df_out, df_iSPE=df_iSPE)
    st.success(f"✅ Tucker3 done — rank [{rank_sensors},{rank_features},{rank_time}]")


# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════
kpi_row([
    ("Total sensors",    len(all_sensors),
                         f"{len(all_sensors) - len(dropped_sensors)} dropped"),
    ("Reliable sensors", len(reliable_sensors), "passed 90% threshold"),
    ("Analysis week",    str(chosen_start.date()), f"→ {str(chosen_end.date())}"),
    ("Time slots",       n_slots, "7 × 48"),
    ("Tucker3",
     "✅ Ready" if "tucker_results" in st.session_state else "⏳ Not run",
     f"rank [{rank_sensors},{rank_features},{rank_time}]"),
])


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_data, tab_map, tab_cov, tab_sig, tab_tucker, tab_diag = st.tabs([
    "Dataset", "Sensor map", "Coverage",
    "Signals",  "Tucker3",    "Diagnostic",
])


# ── TAB 1 — DATASET ──────────────────────────────────────────────────────────
with tab_data:
    sec("Raw dataset explorer")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        sel_sids = st.multiselect("Sensors", df['sensor_ID'].unique().tolist(),
                                   default=df['sensor_ID'].unique().tolist(),
                                   format_func=lambda x: x[-8:])
    with cf2:
        dr = st.date_input("Date range",
                            value=(df['timestamp'].min().date(),
                                   df['timestamp'].max().date()))
    with cf3:
        show_cols = st.multiselect("Columns",
                                    ['timestamp','sensor_ID']+FEATURES+['latitude','longitude'],
                                    default=['timestamp','sensor_ID']+FEATURES)

    mask = df['sensor_ID'].isin(sel_sids)
    if len(dr) == 2:
        mask &= df['timestamp'].dt.date.between(dr[0], dr[1])
    dv = df.loc[mask, show_cols].reset_index(drop=True)

    kpi_row([
        ("Rows",    f"{len(dv):,}", "filtered"),
        ("Sensors", dv['sensor_ID'].nunique() if 'sensor_ID' in dv else "—", "unique"),
        ("Missing", f"{dv.isnull().sum().sum():,}", "NaN cells"),
        ("Days",    dv['timestamp'].dt.date.nunique() if 'timestamp' in dv else "—", "in view"),
    ])
    st.dataframe(dv.style.highlight_null(color='#fdecea'),
                 use_container_width=True, height=400)
    st.download_button("⬇️ Download CSV", dv.to_csv(index=False).encode(),
                       "move_chill.csv", "text/csv")
    with st.expander("📊 Statistics"):
        nc = [c for c in show_cols if c in FEATURES]
        if nc: st.dataframe(dv[nc].describe().round(3), use_container_width=True)


# ── TAB 2 — MAP ──────────────────────────────────────────────────────────────
with tab_map:
    sec("Sensor locations")
    sc2 = sensor_coords.copy()
    sc2['label']    = sc2['sensor_ID'].str[-8:]
    sc2['location'] = sc2['cluster_kmeans'].map({0:'Vulkanplatz',1:'Münsterhof'})
    sc2['missing']  = sc2['sensor_ID'].map(lambda s: miss_pct.get(s, 100))
    sc2['status']   = sc2['sensor_ID'].apply(
                          lambda s: 'Dropped' if s in dropped_sensors else 'Active')

    fig_map = px.scatter_map(
        sc2, lat            = "latitude", 
        lon                 = "longitude", 
        color               = "location",
        color_discrete_map  = {"Vulkanplatz":"#2d6a4f","Münsterhof":"#457b9d"},
        hover_name          = "label",
        hover_data          = {"missing":":.1f","status":True,"latitude":":.5f","longitude":":.5f"},
        size                = [14]*len(sc2), zoom=14, height=480)
    fig_map.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    cl, ct = st.columns([1,2])
    with cl:
        sec("Legend")
        st.markdown("🟢 **Vulkanplatz**\n\n🔵 **Münsterhof**")
        if dropped_sensors:
            st.markdown(f"🔴 **Dropped** — {len(dropped_sensors)}")
    with ct:
        sec("Coordinates")
        st.dataframe(sc2[['label','location','latitude','longitude','missing','status']]
                       .rename(columns={'label':'sensor','missing':'missing %'}),
                     use_container_width=True, hide_index=True)


# ── TAB 3 — COVERAGE  ────────────────────────────────────────────────────────
with tab_cov:
    sec("Full data coverage — all sensors")

    n_days  = len(dates_cov)
    n_sens  = len(coverage.columns)
    cov_pct = coverage.values / 48 * 100          # (days, sensors)

    best_start_idx   = int(weeks_df.iloc[0]['start_idx'])
    chosen_start_idx = int(chosen_row['start_idx'])

    fig, ax = plt.subplots(figsize=(16, max(3.5, n_sens * 0.44)))
    im_cov = ax.imshow(cov_pct.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    # Monday dashed lines
    for i, d in enumerate(dates_cov):
        if d.weekday() == 0:
            ax.axvline(i-.5, color='#1b4332', lw=1.2, ls='--', alpha=.5)
            # ax.text(i, -0.9, 'Mon', fontsize=5.5, ha='center', color='#1b4332')

    # ── Best week — red hue (only when different from chosen) ────────────────
    if not is_best_week:
        ax.add_patch(mpatches.FancyBboxPatch(
            (best_start_idx-.5, -.5), 7, n_sens,
            boxstyle="round,pad=0.15", lw=2,
            edgecolor='#c0392b', facecolor='#e74c3c', alpha=0.15, zorder=3))
        ax.text(best_start_idx+3, n_sens-.05, '★ best week',
                fontsize=7, ha='center', color='#c0392b',
                fontweight='bold', va='bottom', zorder=4)

    # ── Chosen week — blue hue (red if it IS the best week) ──────────────────
    ch_fc = '#e74c3c' if is_best_week else '#3498db'
    ch_ec = '#c0392b' if is_best_week else '#2471a3'
    ax.add_patch(mpatches.FancyBboxPatch(
        (chosen_start_idx-.5, -.5), 7, n_sens,
        boxstyle="round,pad=0.15", lw=2.5,
        edgecolor=ch_ec, facecolor=ch_fc, alpha=0.22, zorder=3))
    # ch_lbl = '★ best week (selected)' if is_best_week else '● selected week'
    # ax.text(chosen_start_idx+3, -.95, ch_lbl,
    #         fontsize=7, ha='center', color=ch_ec,
    #         fontweight='bold', va='top', zorder=4)

    ax.set_yticks(range(n_sens))
    ax.set_yticklabels([s[-8:] for s in coverage.columns], fontsize=7)
    ax.set_xticks(range(0, n_days, 7))
    ax.set_xticklabels([str(d.date()) for d in dates_cov[::7]], rotation=45, fontsize=7)

    # colorbar via separate axes
    plt.colorbar(im_cov, ax=ax, label='% slots received', shrink=.7)

    ax.legend(handles=[
        mpatches.Patch(fc='#e74c3c', alpha=.4, label='★ best week (auto)'),
        mpatches.Patch(fc='#3498db', alpha=.4, label='● selected week'),
    ], loc='upper right', fontsize=7, framealpha=.85)
    ax.set_title('Daily slot coverage — Monday markers | week highlights',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Zoom on chosen week
    sec(f"Zoom — {chosen_start.date()} → {chosen_end.date()}")
    kpi_row([
        ("Avg coverage", f"{chosen_row['mean_coverage']*100:.1f}%", "this week"),
        ("Min coverage", f"{chosen_row['min_coverage']*100:.1f}%",  "worst sensor/day"),
        ("Kept",         len(reliable_sensors), "reliable sensors"),
        ("Dropped",      len(dropped_sensors),  ">90% missing"),
    ])

    bm   = ((coverage.index >= chosen_row['start_date']) &
            (coverage.index <= chosen_row['end_date']))
    bb   = coverage.loc[bm]
    bd   = pd.to_datetime(list(bb.index))
    rc   = [c for c in bb.columns if c in reliable_sensors]

    fig2, ax2 = plt.subplots(figsize=(10, max(2.5, len(rc)*.42)))
    im2 = ax2.imshow(bb[rc].T/48*100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    for i, d in enumerate(bd):
        if d.weekday() == 0:
            ax2.axvline(i-.5, color='#1b4332', lw=2, ls='--')
            ax2.text(i, -0.7, 'Mon', fontsize=8, ha='center',
                     fontweight='bold', color='#1b4332')
    ax2.set_yticks(range(len(rc)))
    ax2.set_yticklabels([s[-8:] for s in rc], fontsize=8)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels([d.strftime('%a %d %b') for d in bd], rotation=40, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='% slots', shrink=.8)
    ax2.set_title('Zoom — selected week (reliable sensors)', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("All Mon→Sun week scores"):
        st.dataframe(weeks_df.drop(columns='start_idx', errors='ignore'),
                     use_container_width=True, hide_index=True)


# ── TAB 4 — SIGNALS ──────────────────────────────────────────────────────────
with tab_sig:
    sec(f"{sel_feature} — …{sel_sensor[-8:]}")
    raw    = sensor_dfs_raw[sel_sensor][sel_feature]
    filled = sensor_dfs_filled[sel_sensor][sel_feature]

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(full_index, filled, color='#2d6a4f', lw=0.9, label=sel_feature)
    ax.fill_between(full_index, filled.min(), filled.max(),
                    where=raw.isnull(), color='#e74c3c', alpha=0.25, label='missing → –1')
    monday_vlines(ax, full_index)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.set_ylabel(sel_feature); ax.legend(fontsize=8)
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.set_title(f'{sel_feature} — …{sel_sensor[-8:]}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    sec(f"All sensors — {sel_feature}")
    n = len(reliable_sensors)
    fig2, axes = plt.subplots(n, 1, figsize=(14, 2.6*n), sharex=True)
    if n == 1: axes = [axes]
    for ax, sid in zip(axes, reliable_sensors):
        col = plt.cm.tab10.colors[reliable_sensors.index(sid) % 10]
        fil = sensor_dfs_filled[sid][sel_feature]
        r2  = sensor_dfs_raw[sid][sel_feature]
        ax.plot(full_index, fil, color=col, lw=0.85)
        ax.fill_between(full_index, fil.min(), fil.max(),
                        where=r2.isnull(), color='#e74c3c', alpha=0.2)
        monday_vlines(ax, full_index)
        ax.set_ylabel(f'…{sid[-6:]}', fontsize=8, rotation=0, labelpad=55, va='center')
        ax.grid(axis='y', ls=':', alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    fig2.suptitle(f'{sel_feature} — all sensors', fontweight='bold', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ── TAB 5 — TUCKER3 FACTORS ──────────────────────────────────────────────────
with tab_tucker:
    if "tucker_results" not in st.session_state:
        st.info("**Tucker3 has not been run yet.**  "
                "Click **⚡ Run Tucker3** in the sidebar, "
                "or upload a pre-computed `.npz` below.")

        with st.expander("📥 Upload tucker_results.npz"):
            st.code("""# Export from notebook:
import numpy as np
np.savez("tucker_results.npz",
    A=results['scores'], B=results['varLoadings'], C=results['timeLoadings'],
    df_out_values=df_out.values,
    df_out_index=np.array(df_out.index.tolist()),
    df_out_columns=np.array(df_out.columns.tolist()),
    df_iSPE_values=df_iSPE.values,
    df_iSPE_columns=np.array(df_iSPE.columns.tolist()),
)""")
            up = st.file_uploader("Upload .npz", type="npz",
                                   label_visibility="collapsed")
            if up is not None:
                npz     = np.load(up, allow_pickle=True)
                _df_out = pd.DataFrame(
                    npz['df_out_values'],
                    index=npz['df_out_index'].tolist(),
                    columns=npz['df_out_columns'].tolist()
                ).apply(pd.to_numeric, errors='coerce')
                _df_iSPE = None
                if 'df_iSPE_values' in npz:
                    _df_iSPE = pd.DataFrame(
                        npz['df_iSPE_values'],
                        index=npz['df_out_index'].tolist(),
                        columns=npz['df_iSPE_columns'].tolist()
                    ).apply(pd.to_numeric, errors='coerce')
                st.session_state["tucker_results"] = dict(
                    A=npz['A'], B=npz['B'], C=npz['C'],
                    df_out=_df_out, df_iSPE=_df_iSPE)
                st.success("✅ Loaded — reopen this tab.")
        st.stop()

    tr      = st.session_state["tucker_results"]
    A, B, C = tr['A'], tr['B'], tr['C']
    df_out  = tr['df_out']
    df_iSPE = tr.get('df_iSPE')
    sn      = [s[-8:] for s in reliable_sensors]

    sec("Tucker3 factor matrices")

    ca, cb = st.columns(2)
    cmap = cm.get_cmap('RdBu_r')
    colors = [mcolors.to_hex(cmap(v)) for v in [0.1, 0.5, 0.9]]  # blue, white, red
    labels = ['LV1', 'LV2', 'LV3']
    with ca:
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(A, cmap='RdBu_r', aspect='auto')
        ax.set_yticks(range(len(sn))); ax.set_yticklabels(sn, fontsize=7)
        ax.set_xticks(range(A.shape[1]))
        ax.set_xticklabels([f'SC{i+1}' for i in range(A.shape[1])])
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title('A — Sensor loadings', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cb:
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(B, cmap='RdBu_r', aspect='auto')
        ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
        ax.set_xticks(range(B.shape[1]))
        ax.set_xticklabels([f'FC{i+1}' for i in range(B.shape[1])])
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title('B — Feature loadings', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    sec("C — Temporal factor matrix")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    pal = ['#2d6a4f','#e76f51','#457b9d','#e9c46a','#9b5de5','#f15bb5']
    for k in range(C.shape[1]):
        ax.plot(full_index, C[:,k],'o-', lw=0.9, color=pal[k%len(pal)], label=f'TC{k+1}')
    monday_vlines(ax, full_index)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.legend(fontsize=8); ax.grid(ls=':', alpha=0.3)
    ax.set_title('Temporal components', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    # ========SVD ====================================================================
    sec("Factor matrix line plots")
    cmap = cm.get_cmap('tab10')
    colors = [mcolors.to_hex(cmap(i)) for i in [1, 0, 2]]  # orange, blue, green
    labels = ['LV1', 'LV2', 'LV3']
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    for i, (col, label) in enumerate(zip(colors, labels)):
        axes[0].plot(A[:, i], 'o-', color=col, label=label, lw=1.4, ms=5)
        axes[1].plot(B[:, i], 'o-', color=col, label=label, lw=1.4, ms=5)
        axes[2].plot(C[:, i], 'o-', color=col, label=label, lw=1.0, ms=3)
    axes[0].set_xticks(range(len(sn))); axes[0].set_xticklabels(sn, rotation=40, ha='right', fontsize=7)
    axes[0].set_ylabel('Scores'); axes[0].grid(True, alpha=.3); axes[0].legend()
    axes[1].set_xticks(range(len(FEATURES))); axes[1].set_xticklabels(FEATURES, rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel('Var Loadings'); axes[1].grid(True, alpha=.3); axes[1].legend()
    axes[2].set_ylabel('Time Loading (day 1)'); axes[2].set_xlabel('Slot')
    axes[2].grid(True, alpha=.3); axes[2].legend()
    fig.suptitle('Tucker3 — Factor line plots', fontweight='bold', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    # ===================================================================================
    lv_cols = [c for c in df_out.columns if c.startswith('LV')]
    if len(lv_cols) >= 3:
        sec("3-D score space")
        fig3d = px.scatter_3d(
            df_out.reset_index().rename(columns={'index':'sensor'}),
            x='LV1',y='LV2',z='LV3',color='cluster',
            color_continuous_scale='RdBu_r',
            hover_name=df_out.index.str[-8:],
            title='Score space LV1×LV2×LV3 — coloured by DModX', height=520)
        fig3d.update_traces(marker=dict(line=dict(width=0.5,color='black')))
        fig3d.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(
                xaxis=dict(range=[-4, 4], title='LV1'),
                yaxis=dict(range=[-4, 4], title='LV2'),
                zaxis=dict(range=[-4, 4], title='LV3'),
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)

    if df_out is not None:
        sec("Individual SPE contributions")
        s1 = st.selectbox("Sensor A", reliable_sensors, key='spe1',
                           format_func=lambda x: f"…{x[-8:]}")
        s2 = st.selectbox("Sensor B", reliable_sensors,
                           index=min(1,len(reliable_sensors)-1), key='spe2',
                           format_func=lambda x: f"…{x[-8:]}")
        c1, c2 = st.columns(2)

        new_cols            = [f"con_TSQR_{f}" for f in FEATURES]
       
        t_vals = df_out.loc[[s1, s2], new_cols].values.flatten()
        i_vals = df_iSPE.loc[[s1, s2]].values.flatten()

        def lim(vals, pad=0.05):
            lo, hi = vals.min(), vals.max()
            p = (hi - lo) * pad
            return (lo - p, hi + p)
        t_lim = lim(t_vals)
        i_lim = lim(i_vals)

        for col, sid in zip([c1,c2],[s1,s2]):
            with col:
                fig, ax = plt.subplots(2,1,figsize=(6,8))
                vals = df_out.loc[sid,new_cols]
                sns.barplot(x=vals.index, y=vals.values, palette='viridis', ax=ax[0])
                ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=40,ha='right',fontsize=8)
                ax[0].set_title(f'T_SPE — …{sid[-8:]}', fontweight='bold')
                ax[0].set_ylabel('Contribution')
                ax[0].set_ylim(t_lim)
                vals = df_iSPE.loc[sid]
                sns.barplot(x=vals.index, y=vals.values, palette='viridis', ax=ax[1])
                ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=40,ha='right',fontsize=8)
                ax[1].set_title(f'i_SPE — …{sid[-8:]}', fontweight='bold')
                ax[1].set_ylabel('Error')
                ax[1].set_ylim(i_lim)
                plt.tight_layout()
                col.pyplot(fig)
                plt.close(fig)


# ── TAB 6 — DIAGNOSTIC ───────────────────────────────────────────────────────
with tab_diag:
    if "tucker_results" not in st.session_state:
        st.info("Run Tucker3 first (⚡ sidebar) to unlock this tab.")
    else:
        tr     = st.session_state["tucker_results"]
        df_out = tr['df_out']
        sec("Sensor diagnostic map — T²R vs DModX")

        x, y  = df_out['TSQR'], df_out['DModX']
        v     = df_out.index.astype(str).to_numpy()
        x_max = max(x.max()*1.18, TSQR_THRESH*1.35)
        y_max = max(y.max()*1.18, DMODX_THRESH*1.35)

        fig, ax = plt.subplots(figsize=(11, 8))
        for (y0,y1),(x0,x1),fc in [
            ([0,DMODX_THRESH],    [0,TSQR_THRESH],       '#2ecc71'),
            ([0,DMODX_THRESH],    [TSQR_THRESH, x_max],  '#f39c12'),
            ([DMODX_THRESH,y_max],[0,TSQR_THRESH],       '#3498db'),
            ([DMODX_THRESH,y_max],[TSQR_THRESH, x_max],  '#e74c3c'),
        ]:
            ax.fill_betweenx([y0, y1], x0, x1, color=fc, alpha=0.10)

        ax.axvline(TSQR_THRESH,  color='#c0392b',lw=1.5,ls='--',label=f'T²R={TSQR_THRESH}')
        ax.axhline(DMODX_THRESH, color='#2980b9',lw=1.5,ls='--',label=f'DModX={DMODX_THRESH}')

        lbl = dict(fontsize=9,fontweight='bold',alpha=0.55,ha='center',va='center')
        ax.text(TSQR_THRESH*.5, DMODX_THRESH*.5,
                '✅ In Control\nLow T² | Low DModX', color='#27ae60',**lbl)
        ax.text(TSQR_THRESH+(x_max-TSQR_THRESH)*.5, DMODX_THRESH*.5,
                '⚠️ Score Outlier\nHigh T² | Low DModX', color='#e67e22',**lbl)
        ax.text(TSQR_THRESH*.5, DMODX_THRESH+(y_max-DMODX_THRESH)*.5,
                '🔵 Residual Outlier\nLow T² | High DModX', color='#2980b9',**lbl)
        ax.text(TSQR_THRESH+(x_max-TSQR_THRESH)*.5, DMODX_THRESH+(y_max-DMODX_THRESH)*.5,
                '🔴 Critical Outlier\nHigh T² | High DModX', color='#c0392b',**lbl)

        sc = ax.scatter(x, y, c=x, cmap='RdBu_r', alpha=0.85,
                        edgecolors='k', linewidths=0.5, zorder=5, s=90)
        for xi, yi, vi in zip(x, y, v):
            ax.text(xi+x_max*.01, yi+y_max*.01, vi[-8:],
                    fontsize=7, alpha=0.9, zorder=6)

        plt.colorbar(sc, ax=ax, label='T²R value', shrink=0.5)
        ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
        ax.set_xlabel('T²R (Hotelling) — within model space', labelpad=10)
        ax.set_ylabel('DModX — outside model space', labelpad=10)
        ax.set_title('Sensor Diagnostic Map — T²R vs DModX',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, framealpha=.8)
        ax.grid(True, ls=':', alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Sensors above both thresholds = critical outliers
        outside_ids = df_out[
            (df_out['TSQR'] > TSQR_THRESH) & (df_out['DModX'] > DMODX_THRESH)
        ].index.tolist()
        if outside_ids:
            st.warning(f"**{len(outside_ids)} critical outlier sensor(s) "
                       f"(High T² AND High DModX):** "
                       f"{', '.join(s[-8:] for s in outside_ids)}")
        else:
            st.success("No critical outliers — all sensors within both thresholds ✅")

        sec("Diagnostic table")
        dc = [c for c in ['TSQR','DModX','HT2','T_SPE'] if c in df_out.columns]
        st.dataframe(
            df_out[dc].style
              .background_gradient(subset=['TSQR'], cmap='RdYlGn_r')
              .background_gradient(subset=['DModX'],cmap='RdYlGn_r')
              .format("{:.3f}"),
            use_container_width=True)