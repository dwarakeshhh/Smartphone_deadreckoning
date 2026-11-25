"""Real-Time Smartphone Inertial Motion Tracking - Streamlit dashboard.

Run from the project root:   streamlit run dashboard/app.py
It polls the FastAPI server (default http://localhost:8000); every number on
screen comes from sensor packets the server received from the phone.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_SERVER = os.getenv("DR_SERVER_URL", "http://localhost:8000")
TITLE = "Real-Time Smartphone Inertial Motion Tracking"

st.set_page_config(page_title=TITLE, page_icon="📱", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.banner {border-radius: 12px; padding: 14px 22px; color: #fff; font-weight: 700; font-size: 1.5rem; letter-spacing: .5px;}
.chip {display:inline-block; padding: 3px 12px; margin: 0 8px 0 0; border-radius: 999px; font-size: .8rem; font-weight:600; background: rgba(255,255,255,.22);}
.card {background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;}
.card .lbl {font-size: .75rem; text-transform: uppercase; letter-spacing: .6px; opacity: .7;}
.card .val {font-size: 1.55rem; font-weight: 700; line-height: 1.25;}
.card .sub {font-size: .75rem; opacity: .6;}
.sec {font-size: 1.05rem; font-weight: 700; margin: 14px 0 6px 0; opacity: .9;}
.warn {border-left: 5px solid #f59e0b; background: rgba(245,158,11,.12); padding: 8px 14px; border-radius: 6px; margin: 6px 0;}
</style>
""", unsafe_allow_html=True)

COLORS = {"LIVE TRACKING": "#16a34a", "PHONE CONNECTED": "#2563eb", "TRACKING STOPPED": "#475569",
          "WAITING FOR PHONE": "#6b7280", "PHONE DISCONNECTED": "#dc2626"}


def card(label: str, value, sub: str = "") -> str:
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def api(method: str, path: str, **kw):
    url = st.session_state.get("server_url", DEFAULT_SERVER).rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=2.5, **kw)
        return r.json() if r.content else {}
    except Exception as e:                      # noqa: BLE001
        return {"_error": str(e)}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Server")
    st.session_state["server_url"] = st.text_input("Server URL", DEFAULT_SERVER)
    refresh = st.slider("Refresh interval (s)", 0.2, 2.0, 0.5, 0.1)

    st.header("Session controls")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", width="stretch", help="Start a new tracking session (phone should be still for ~1.5 s)"):
        api("POST", "/tracking/start")
    if c2.button("■ Stop", width="stretch"):
        api("POST", "/tracking/stop")
    if st.button("Reset Position", width="stretch", help="Set X/Y to 0 and restart the plotted path; keeps distance/steps"):
        api("POST", "/reset", params={"scope": "position"})
    if st.button("Reset Session", width="stretch", help="Forget everything and re-calibrate"):
        api("POST", "/reset", params={"scope": "session"})
    if st.button("Clear Data", width="stretch", help="Clear the plotted path and the in-memory sample buffer"):
        api("POST", "/reset", params={"scope": "data"})

    st.header("Recording (for ML / evaluation)")
    rec_name = st.text_input("Session name", "walking_forward")
    rec_label = st.selectbox("Activity label", ["", "stationary", "walking", "running", "other"], index=2)
    r1, r2 = st.columns(2)
    if r1.button("● Record", width="stretch"):
        res = api("POST", "/recording/start", json={"name": rec_name, "label": rec_label or None})
        st.session_state["rec_msg"] = res.get("detail") or res.get("_error") or f"recording → {res.get('meta', {}).get('file')}"
    if r2.button("■ Save", width="stretch"):
        res = api("POST", "/recording/stop")
        st.session_state["rec_msg"] = res.get("detail") or res.get("_error") or f"saved {res.get('file')} ({res.get('rows')} rows)"
    if st.session_state.get("rec_msg"):
        st.caption(st.session_state["rec_msg"])
    st.caption("Files go to data/raw/ (see docs/protocol.md).")

st.title(TITLE)
st.caption("IMU dead reckoning from the phone's accelerometer, gyroscope and magnetometer. "
           "The path is an ESTIMATE (not GPS) and drifts with time.")


def make_figure(traj: dict) -> go.Figure:
    fig = go.Figure()
    x, y = traj["x"], traj["y"]
    fig.add_trace(go.Scatter(x=[-1, 1], y=[-1, 1], mode="markers", marker={"opacity": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Estimated trajectory", line={"width": 3, "color": "#2563eb"},
                             marker={"size": 4, "color": "#2563eb"}, customdata=traj["t"],
                             hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>t=%{customdata:.1f} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=[traj["start"][0]], y=[traj["start"][1]], mode="markers+text", name="Start", text=["Start"],
                             textposition="bottom right", marker={"size": 15, "color": "#16a34a", "symbol": "diamond"}))
    fig.add_trace(go.Scatter(x=[traj["current"][0]], y=[traj["current"][1]], mode="markers+text", name="Current position",
                             text=["Now"], textposition="top right", marker={"size": 15, "color": "#dc2626", "symbol": "circle"}))
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 30, "b": 10}, uirevision="trajectory",
                      xaxis={"title": "Estimated X (m)  [East]", "zeroline": True, "gridcolor": "rgba(128,128,128,.25)"},
                      yaxis={"title": "Estimated Y (m)  [North]", "scaleanchor": "x", "scaleratio": 1, "zeroline": True,
                             "gridcolor": "rgba(128,128,128,.25)"},
                      legend={"orientation": "h", "y": 1.08}, title="Live estimated trajectory (dead reckoning, NOT GPS)")
    return fig


def render() -> None:
    s = api("GET", "/state")
    if "_error" in s:
        st.markdown(f'<div class="banner" style="background:#dc2626">SERVER UNREACHABLE</div>', unsafe_allow_html=True)
        st.error(f"Cannot reach {st.session_state['server_url']}  →  start it with run_server.bat  ({s['_error'][:120]})")
        return
    ui, q, stt, sens = s["ui_status"], s["quality"], s["stats"], s["sensors"]
    color = COLORS.get(ui, "#d97706")
    chips = ""
    if s["connected"]:
        chips += '<span class="chip">PHONE CONNECTED</span>'
    if s["tracking"] and s["phase"] == "TRACKING":
        chips += '<span class="chip">TRACKING ACTIVE</span>'
    st.markdown(f'<div class="banner" style="background:{color}">{"● " if ui == "LIVE TRACKING" else ""}{ui}'
                f'<div style="margin-top:6px">{chips}</div></div>', unsafe_allow_html=True)
    if s["simulated"]:
        st.markdown('<div class="warn"><b>SIMULATED / REPLAYED DATA</b> - these packets are not coming from the phone '
                    f'(source = {s["source"]}).</div>', unsafe_allow_html=True)
    if s["phase"] == "CALIBRATING":
        st.progress(min(1.0, q["calibration_progress"]), text="Calibrating - keep the phone still on the start point"
                    + (" (motion detected, waiting)" if q["calibration_waiting_for_still"] else ""))

    # A. CONNECTION ---------------------------------------------------------
    st.markdown('<div class="sec">A · Connection</div>', unsafe_allow_html=True)
    a = st.columns(4)
    a[0].markdown(card("Phone", "Connected" if s["connected"] else "Disconnected"), unsafe_allow_html=True)
    a[1].markdown(card("Phone IP address", s["client_ip"] or "-"), unsafe_allow_html=True)
    a[2].markdown(card("Packets received", f'{s["packets_received"]:,}', f'{s["packets_rejected"]} rejected'), unsafe_allow_html=True)
    a[3].markdown(card("Sampling rate", f'{s["sampling_rate_hz"]:.1f} Hz', "measured at the server"), unsafe_allow_html=True)

    # B/C/D -----------------------------------------------------------------
    left, mid, right = st.columns([1.5, 1, 1.3])
    with left:
        st.markdown('<div class="sec">B · Live sensor data</div>', unsafe_allow_html=True)
        df = pd.DataFrame({"Sensor": ["Accelerometer (m/s²)", "Gyroscope (rad/s)", "Magnetometer (µT)"],
                           "X": [sens["acc"][0], sens["gyro"][0], sens["mag"][0]],
                           "Y": [sens["acc"][1], sens["gyro"][1], sens["mag"][1]],
                           "Z": [sens["acc"][2], sens["gyro"][2], sens["mag"][2]],
                           "|magnitude|": [sens["acc_magnitude"], sens["gyro_magnitude"],
                                           round(sum(v * v for v in sens["mag"]) ** 0.5, 2)]})
        st.dataframe(df, hide_index=True, width="stretch")
    with mid:
        st.markdown('<div class="sec">C · Orientation</div>', unsafe_allow_html=True)
        o = sens["orientation_deg"]
        c = st.columns(3)
        c[0].markdown(card("Roll", f'{o["roll"]:.1f}°'), unsafe_allow_html=True)
        c[1].markdown(card("Pitch", f'{o["pitch"]:.1f}°'), unsafe_allow_html=True)
        c[2].markdown(card("Yaw", f'{o["yaw"]:.1f}°', f'heading {o["heading"]:.0f}°'), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="sec">D · Movement</div>', unsafe_allow_html=True)
        d = st.columns(4)
        d[0].markdown(card("State", s["movement_state"]), unsafe_allow_html=True)
        d[1].markdown(card("Steps", stt["steps"]), unsafe_allow_html=True)
        d[2].markdown(card("Speed", f'{stt["current_speed_mps"]:.2f}', "m/s"), unsafe_allow_html=True)
        d[3].markdown(card("Distance", f'{stt["total_distance_m"]:.2f}', "m (integrated)"), unsafe_allow_html=True)

    # E/F -------------------------------------------------------------------
    e, f = st.columns([2.3, 1])
    with e:
        st.markdown('<div class="sec">E · Live trajectory</div>', unsafe_allow_html=True)
        traj = api("GET", "/trajectory", params={"max_points": 1500})
        if "_error" not in traj:
            st.plotly_chart(make_figure(traj), width="stretch", key="traj_chart")
    with f:
        st.markdown('<div class="sec">F · Statistics</div>', unsafe_allow_html=True)
        st.markdown(card("Total distance (integrated)", f'{stt["total_distance_m"]:.2f} m',
                         f'step-based: {stt["step_distance_m"]:.1f} m @ {stt["step_length_m"]} m/step'), unsafe_allow_html=True)
        g = st.columns(2)
        g[0].markdown(card("Current speed", f'{stt["current_speed_mps"]:.2f} m/s'), unsafe_allow_html=True)
        g[1].markdown(card("Max speed", f'{stt["max_speed_mps"]:.2f} m/s'), unsafe_allow_html=True)
        g = st.columns(2)
        g[0].markdown(card("Steps", stt["steps"]), unsafe_allow_html=True)
        g[1].markdown(card("Duration", f'{stt["duration_s"]:.1f} s'), unsafe_allow_html=True)
        g = st.columns(2)
        g[0].markdown(card("Current X", f'{stt["x_m"]:.2f} m'), unsafe_allow_html=True)
        g[1].markdown(card("Current Y", f'{stt["y_m"]:.2f} m'), unsafe_allow_html=True)

    # Summary after stop ----------------------------------------------------
    if s["phase"] == "STOPPED":
        st.markdown('<div class="sec">SESSION SUMMARY</div>', unsafe_allow_html=True)
        m = st.columns(6)
        m[0].markdown(card("Total distance", f'{stt["total_distance_m"]:.2f} m'), unsafe_allow_html=True)
        m[1].markdown(card("Steps", stt["steps"]), unsafe_allow_html=True)
        m[2].markdown(card("Average speed", f'{stt["average_speed_mps"]:.2f} m/s'), unsafe_allow_html=True)
        m[3].markdown(card("Maximum speed", f'{stt["max_speed_mps"]:.2f} m/s'), unsafe_allow_html=True)
        m[4].markdown(card("Tracking duration", f'{stt["duration_s"]:.1f} s'), unsafe_allow_html=True)
        m[5].markdown(card("Estimated final position", f'({stt["final_position_m"][0]:.2f}, {stt["final_position_m"][1]:.2f}) m'),
                      unsafe_allow_html=True)

    # Quality ---------------------------------------------------------------
    st.markdown('<div class="sec">Error / quality information</div>', unsafe_allow_html=True)
    qc = st.columns(5)
    qc[0].markdown(card("Zero-velocity state", "STILL" if q["stationary"] else "moving",
                        f'{q["zupt_count"]} ZUPT corrections'), unsafe_allow_html=True)
    qc[1].markdown(card("Time since last ZUPT", f'{q["seconds_since_zupt"]:.1f} s',
                        f'heuristic uncertainty ≈ {q["position_uncertainty_heuristic_m"]} m'), unsafe_allow_html=True)
    qc[2].markdown(card("Accel / mag trust", f'{q["acc_trust"]:.2f} / {q["mag_trust"]:.2f}',
                        "mag in use" if q["mag_in_use"] else "magnetometer not used"), unsafe_allow_html=True)
    qc[3].markdown(card("Data quality", f'{q["gaps"]} gaps · {q["missed_samples"]} missed', f'{q["out_of_order"]} out-of-order'),
                   unsafe_allow_html=True)
    ml = "—" if not s["ml_model_loaded"] else f'{s["ml_state"] or "…"}'
    qc[4].markdown(card("ML movement class", ml, "optional model" if s["ml_model_loaded"] else "no model trained yet"),
                   unsafe_allow_html=True)
    st.markdown(f'<div class="warn">{q["warning"]}</div>', unsafe_allow_html=True)
    if q["last_error"]:
        st.caption(f'Last rejected-packet reason: {q["last_error"]}')
    if s["recording"].get("active"):
        st.info(f'Recording → {s["recording"].get("meta", {}).get("file")}  ({s["recording"]["rows"]} rows)')


st.fragment(run_every=refresh)(render)()
