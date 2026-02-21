# ═══════════════════════════════════════════════════════════
# TAB : Dataset explorer
# ═══════════════════════════════════════════════════════════
import pandas as pd 
import streamlit as st 

def read_csv(): 
    df = pd.read_csv('resources/data/taz.view_moveandchill.csv',index_col='id')
    COLUMN_MAP =  {
        'objectid'      : 'ID',
        'sensor_eui'    : 'sensor_ID',
        'zeitpunkt'     : 'timestamp',
        'temperature'   : 'temperature',
        'humidity'      : 'humidity',
        'latitude'      : 'latitude', 
        'longitude'     : 'longitude',
        'noise'         : 'noise',
        'sit'           : "% occupancy"   # every 15-30 mins the sensor resets and checks occupancy 
        
    }
    df.rename(columns=COLUMN_MAP,inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df = df.sort_values(by='timestamp',ascending=True)

    return df 

tab1, tab2, tab_data, tab_map = st.tabs([
    "Tab One",      # → tab1
    "Tab Two",      # → tab2
    "🗃️ Dataset",   # → tab_data
    "🗺️ Sensor map" # → tab_map
])

with tab_data:
    st.markdown('<div class="section-header">Raw dataset</div>', unsafe_allow_html=True)

    df = read_csv()
    features   = ['temperature', 'humidity', 'noise', '% occupancy']

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_sensors = st.multiselect("Filter by sensor", 
                                      options=df['sensor_ID'].unique().tolist(),
                                      default=df['sensor_ID'].unique().tolist())
    with col_f2:
        sel_dates = st.date_input("Date range",
                                   value=(df['timestamp'].min().date(),
                                          df['timestamp'].max().date()))
    with col_f3:
        sel_features = st.multiselect("Columns to show",
                                       options=features + ['timestamp', 'sensor_ID'],
                                       default=['timestamp','sensor_ID'] + features)

    # ── Filter dataframe ──────────────────────────────────────────────────────
    mask = df['sensor_ID'].isin(sel_sensors)
    if len(sel_dates) == 2:
        mask &= df['timestamp'].dt.date.between(sel_dates[0], sel_dates[1])

    df_view = df.loc[mask, sel_features].reset_index(drop=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows shown",     f"{len(df_view):,}")
    m2.metric("Sensors",        df_view['sensor_ID'].nunique() if 'sensor_ID' in df_view else "—")
    m3.metric("Missing values", f"{df_view.isnull().sum().sum():,}")
    m4.metric("Date range",     f"{df_view['timestamp'].dt.date.nunique()} days"
                                 if 'timestamp' in df_view else "—")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(
        df_view.style.highlight_null(color='#fdecea'),   # red cells for NaN
        use_container_width=True,
        height=420
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download filtered CSV",
        data=df_view.to_csv(index=False).encode('utf-8'),
        file_name=f"move_chill_filtered.csv",
        mime="text/csv"
    )

    # ── Quick stats ───────────────────────────────────────────────────────────
    with st.expander("📊 Descriptive statistics"):
        num_cols = [c for c in sel_features if c in features]
        if num_cols:
            st.dataframe(df_view[num_cols].describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB : Sensor map
# ═══════════════════════════════════════════════════════════
# with tab_map:
#     st.markdown('<div class="section-header">Sensor locations — Vulkanplatz & Münsterhof</div>',
#                 unsafe_allow_html=True)

#     # ── Build location dataframe ──────────────────────────────────────────────
#     # Replace with your actual lat/lon per sensor
#     sensor_locations = (
#         df.groupby('sensor_ID')[['latitude', 'longitude']]
#         .mean()
#         .reset_index()
#     )

#     # Enrich with cluster + reliability info
#     sensor_locations['cluster']  = sensor_locations['sensor_ID'].map(sensor_to_cluster)
#     sensor_locations['location'] = sensor_locations['cluster'].map(
#                                        {0: 'Vulkanplatz', 1: 'Münsterhof'})
#     sensor_locations['status']   = sensor_locations['sensor_ID'].apply(
#                                        lambda s: 'dropped' if s in dropped_sensors else 'active')

#     # Missing rate per sensor (for tooltip)
#     missing_rate = {}
#     for sid in reliable_sensors:
#         raw  = sensor_dfs_raw[sid]
#         rate = raw.isnull().any(axis=1).mean() * 100
#         missing_rate[sid] = round(rate, 1)

#     sensor_locations['missing_%'] = sensor_locations['sensor_ID'].map(
#                                         lambda s: missing_rate.get(s, 100.0))

#     # ── Color column for st.map (must be RGBA list) ───────────────────────────
#     def cluster_color(row):
#         if row['status'] == 'dropped':
#             return [231, 76, 60, 180]      # red   — dropped
#         elif row['cluster'] == 0:
#             return [45, 106, 79, 220]      # green — Vulkanplatz
#         else:
#             return [52, 120, 180, 220]     # blue  — Münsterhof

#     sensor_locations['color'] = sensor_locations.apply(cluster_color, axis=1)

#     # ── Map ───────────────────────────────────────────────────────────────────
#     col_map, col_legend = st.columns([3, 1])

#     with col_map:
#         st.map(
#             sensor_locations.rename(columns={'latitude': 'lat', 'longitude': 'lon'}),
#             color='color',
#             size=80,
#             zoom=15,
#         )

#     with col_legend:
#         st.markdown("#### Legend")
#         st.markdown("""
#         🟢 **Vulkanplatz** (cluster 0)  
#         🔵 **Münsterhof** (cluster 1)  
#         🔴 **Dropped** (>75% missing)
#         """)
#         st.markdown("---")
#         st.markdown("#### Sensor details")
#         for _, row in sensor_locations.iterrows():
#             icon   = "🔴" if row['status'] == 'dropped' else ("🟢" if row['cluster'] == 0 else "🔵")
#             status = "dropped" if row['status'] == 'dropped' else f"{row['missing_%']}% missing"
#             st.markdown(
#                 f"{icon} **{row['sensor_ID']}** — {row['location']}  \n"
#                 f"<small style='color:grey'>{status}</small>",
#                 unsafe_allow_html=True
#             )

#     # ── Table below map ───────────────────────────────────────────────────────
#     with st.expander("📍 Sensor coordinates table"):
#         st.dataframe(
#             sensor_locations[['sensor_ID','location','latitude','longitude','missing_%','status']],
#             use_container_width=True
#         )