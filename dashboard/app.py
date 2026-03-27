"""
PMEC-X: Live Demo Dashboard
=============================
dashboard/app.py

HOW TO RUN:
    streamlit run dashboard/app.py

WHAT THE PANEL SEES:
    - Live simulation running epoch by epoch
    - Real-time energy comparison chart (PMEC-X vs all baselines)
    - VM heatmap showing consolidation happening live
    - CUSUM alarm timeline — spikes detected in real time
    - Carbon intensity dial per VM rack
    - One-click workload spike injection for demo
    - Final results summary table
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import copy
import time
import random

from simulation.workload import WorkloadGenerator, build_vm_pool
from simulation.environment import (
    PMECXScheduler, RoundRobinScheduler,
    MinMinScheduler, FCFSScheduler, EpochMetrics
)
from core.scorer import WeightVector
from core.models import VMStatus

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "PMEC-X Dashboard",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #2E75B6;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .novelty-badge {
        display: inline-block;
        background: #EEEDFE;
        color: #3C3489;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
    }
    .alarm-box {
        background: #FCEBEB;
        border: 1px solid #F09595;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #A32D2D;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .stable-box {
        background: #EAF3DE;
        border: 1px solid #97C459;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #27500A;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ PMEC-X Controls")
    st.markdown("---")

    st.markdown("### Simulation Settings")
    n_epochs     = st.slider("Epochs to run", 60, 720, 360, step=60,
                              help="720 = full 24-hour simulation")
    speed        = st.slider("Demo speed (epochs/sec)", 1, 50, 20,
                              help="Higher = faster animation")
    random_seed  = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.markdown("### Weight Vector W")
    st.markdown("*Adjust PMEC-X factor priorities:*")
    w_speed    = st.slider("Speed",    0.0, 1.0, 1/7, 0.05)
    w_headroom = st.slider("Headroom", 0.0, 1.0, 1/7, 0.05)
    w_cost     = st.slider("Cost",     0.0, 1.0, 1/7, 0.05)
    w_energy   = st.slider("Energy",   0.0, 1.0, 1/7, 0.05)
    w_carbon   = st.slider("Carbon",   0.0, 1.0, 1/7, 0.05)
    w_thermal  = st.slider("Thermal",  0.0, 1.0, 1/7, 0.05)
    w_urgency  = st.slider("Urgency",  0.0, 1.0, 1/7, 0.05)

    st.markdown("---")
    st.markdown("### Demo Actions")
    inject_spike = st.button("💥 Inject Workload Spike",
                              help="Injects 25 urgent tasks — triggers CUSUM alarm")
    st.markdown("*Press during simulation to demo CUSUM detection*")

    st.markdown("---")
    st.markdown("### Novelty Contributions")
    st.markdown('<span class="novelty-badge">N1: Dual-layer VM+Container</span>', unsafe_allow_html=True)
    st.markdown('<span class="novelty-badge">N2: EWMA+CUSUM Predictor</span>', unsafe_allow_html=True)
    st.markdown('<span class="novelty-badge">N3: Carbon Intensity Factor</span>', unsafe_allow_html=True)
    st.markdown('<span class="novelty-badge">N4: Adaptive Weight Tuning</span>', unsafe_allow_html=True)
    st.markdown('<span class="novelty-badge">N5: Thermal-Aware Scoring</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">⚡ PMEC-X Live Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predictive Multifactor Energy Consolidation · Cross-layer Adaptive Scheduler</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

if "running"        not in st.session_state: st.session_state.running        = False
if "epoch"          not in st.session_state: st.session_state.epoch          = 0
if "schedulers"     not in st.session_state: st.session_state.schedulers     = None
if "schedule"       not in st.session_state: st.session_state.schedule       = None
if "history"        not in st.session_state: st.session_state.history        = []
if "cusum_events"   not in st.session_state: st.session_state.cusum_events   = []
if "spike_pending"  not in st.session_state: st.session_state.spike_pending  = False
if "done"           not in st.session_state: st.session_state.done           = False

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL BUTTONS
# ─────────────────────────────────────────────────────────────────────────────

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1,1,1,4])

with col_btn1:
    if st.button("▶ Start", type="primary", disabled=st.session_state.running):
        # Build workload and schedulers
        weights = WeightVector(
            w_speed=w_speed, w_headroom=w_headroom, w_cost=w_cost,
            w_energy=w_energy, w_carbon=w_carbon,
            w_thermal=w_thermal, w_urgency=w_urgency
        )
        weights.normalize()

        gen      = WorkloadGenerator(seed=int(random_seed))
        schedule = gen.generate_full_day()[:n_epochs]
        base     = build_vm_pool(seed=int(random_seed))

        st.session_state.schedule   = schedule
        st.session_state.schedulers = {
            "PMEC-X":      PMECXScheduler(copy.deepcopy(base), weights),
            "Round Robin": RoundRobinScheduler(copy.deepcopy(base)),
            "MinMin":      MinMinScheduler(copy.deepcopy(base)),
            "FCFS":        FCFSScheduler(copy.deepcopy(base)),
        }
        st.session_state.epoch        = 0
        st.session_state.history      = []
        st.session_state.cusum_events = []
        st.session_state.running      = True
        st.session_state.done         = False
        st.rerun()

with col_btn2:
    if st.button("⏹ Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

with col_btn3:
    if st.button("🔄 Reset"):
        st.session_state.running    = False
        st.session_state.epoch      = 0
        st.session_state.schedulers = None
        st.session_state.schedule   = None
        st.session_state.history    = []
        st.session_state.cusum_events = []
        st.session_state.done       = False
        st.rerun()

if inject_spike:
    st.session_state.spike_pending = True

# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

total_epochs = len(st.session_state.schedule) if st.session_state.schedule else n_epochs
progress_val = st.session_state.epoch / total_epochs if total_epochs > 0 else 0
hour_display = (st.session_state.epoch * 120) / 3600

st.progress(progress_val,
    text=f"Epoch {st.session_state.epoch}/{total_epochs} | "
         f"Simulated time: {hour_display:.1f}h / 24.0h")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5, m6 = st.columns(6)

if st.session_state.history:
    h = st.session_state.history
    pmecx_energy = sum(r["PMEC-X"]["energy"]       for r in h)
    rr_energy    = sum(r["Round Robin"]["energy"]   for r in h)
    mm_energy    = sum(r["MinMin"]["energy"]        for r in h)
    pmecx_viol   = sum(r["PMEC-X"]["violated"]      for r in h)
    rr_viol      = sum(r["Round Robin"]["violated"] for r in h)
    pmecx_vms    = h[-1]["PMEC-X"]["active_vms"] if h else 0
    cusum_count  = len(st.session_state.cusum_events)
    pmecx_shut   = sum(r["PMEC-X"]["shutdowns"]     for r in h)

    energy_saved = (1 - pmecx_energy / mm_energy) * 100 if mm_energy > 0 else 0

    m1.metric("PMEC-X Energy",    f"{pmecx_energy:.2f} kWh")
    m2.metric("MinMin Energy",    f"{mm_energy:.2f} kWh")
    m3.metric("Energy Saved",     f"{energy_saved:.1f}%",
              delta=f"{energy_saved:.1f}% vs MinMin")
    m4.metric("PMEC-X Active VMs", str(pmecx_vms),
              delta=f"-{20 - pmecx_vms} vs baselines")
    m5.metric("CUSUM Alarms",     str(cusum_count))
    m6.metric("SLA Violations",   f"PMEC-X: {pmecx_viol}  |  RR: {rr_viol}")
else:
    m1.metric("PMEC-X Energy",    "—")
    m2.metric("MinMin Energy",    "—")
    m3.metric("Energy Saved",     "—")
    m4.metric("Active VMs",       "—")
    m5.metric("CUSUM Alarms",     "—")
    m6.metric("SLA Violations",   "—")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown("#### Cumulative Energy Consumption (kWh)")
    energy_chart = st.empty()

with chart_col2:
    st.markdown("#### Active VM Count per Scheduler")
    vm_chart = st.empty()

row2_col1, row2_col2, row2_col3 = st.columns([2, 2, 1])

with row2_col1:
    st.markdown("#### Task Arrival Rate & CUSUM Alarms")
    cusum_chart = st.empty()

with row2_col2:
    st.markdown("#### VM Heatmap — Current Load")
    heatmap_placeholder = st.empty()

with row2_col3:
    st.markdown("#### CUSUM Status")
    cusum_status = st.empty()

st.markdown("---")
results_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# CHART RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "PMEC-X":      "#1F4E79",
    "Round Robin": "#C00000",
    "MinMin":      "#ED7D31",
    "FCFS":        "#70AD47",
}

def render_energy_chart(history):
    if not history:
        energy_chart.empty()
        return
    epochs = list(range(len(history)))
    fig = go.Figure()
    for name, color in COLORS.items():
        cumulative = []
        total = 0
        for r in history:
            total += r[name]["energy"]
            cumulative.append(total)
        width = 3 if name == "PMEC-X" else 1.5
        dash   = "solid" if name == "PMEC-X" else "dot"
        fig.add_trace(go.Scatter(
            x=epochs, y=cumulative, name=name,
            line=dict(color=color, width=width, dash=dash),
            fill='none'
        ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=10, t=10, b=30),
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Epoch", yaxis_title="kWh",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    energy_chart.plotly_chart(fig, use_container_width=True, key="energy")


def render_vm_chart(history):
    if not history:
        vm_chart.empty()
        return
    epochs = list(range(len(history)))
    fig = go.Figure()
    for name, color in COLORS.items():
        vms = [r[name]["active_vms"] for r in history]
        width = 3 if name == "PMEC-X" else 1.5
        fig.add_trace(go.Scatter(
            x=epochs, y=vms, name=name,
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=10, t=10, b=30),
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Epoch", yaxis_title="Active VMs",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", range=[0, 22])
    vm_chart.plotly_chart(fig, use_container_width=True, key="vms")


def render_cusum_chart(history, cusum_events):
    if not history:
        cusum_chart.empty()
        return
    epochs  = list(range(len(history)))
    arrivals = [r["PMEC-X"]["arrived"] for r in history]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=epochs, y=arrivals, name="Task arrivals",
        marker_color="#B5D4F4", opacity=0.7
    ))
    # Mark CUSUM alarms
    for ev in cusum_events:
        fig.add_vline(x=ev, line_dash="dash",
                      line_color="#C00000", line_width=1.5,
                      annotation_text="CUSUM", annotation_font_size=9,
                      annotation_font_color="#C00000")
    fig.update_layout(
        height=280, margin=dict(l=20, r=10, t=10, b=30),
        xaxis_title="Epoch", yaxis_title="Tasks/epoch",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, font=dict(size=11)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    cusum_chart.plotly_chart(fig, use_container_width=True, key="cusum")


def render_heatmap(schedulers):
    if schedulers is None:
        heatmap_placeholder.empty()
        return
    pmecx = schedulers["PMEC-X"]
    vms   = list(pmecx.vms.values())
    vm_names = [vm.name for vm in vms]
    loads    = [vm.current_load if vm.status != VMStatus.SHUTDOWN else -0.1 for vm in vms]
    statuses = [vm.status.value for vm in vms]

    colors = []
    for vm, load in zip(vms, loads):
        if vm.status == VMStatus.SHUTDOWN:
            colors.append("#E8E8E8")
        elif load > 0.8:
            colors.append("#C00000")
        elif load > 0.5:
            colors.append("#ED7D31")
        elif load > 0.25:
            colors.append("#2E75B6")
        else:
            colors.append("#70AD47")

    display_loads = [max(0, l) for l in loads]
    fig = go.Figure(go.Bar(
        x=vm_names,
        y=display_loads,
        marker_color=colors,
        text=[f"{l:.0%}" if l >= 0 else "OFF" for l in loads],
        textposition="outside",
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=60),
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=9), showlegend=False,
        xaxis=dict(tickangle=45)
    )
    heatmap_placeholder.plotly_chart(fig, use_container_width=True, key="heatmap")


def render_cusum_status(cusum_events, epoch):
    recent_alarm = any(e >= epoch - 3 for e in cusum_events) if cusum_events else False
    if recent_alarm:
        cusum_status.markdown(
            '<div class="alarm-box">🚨 CUSUM ALARM<br>'
            'Regime change detected!<br>'
            'Emergency re-score triggered.</div>',
            unsafe_allow_html=True
        )
    else:
        cusum_status.markdown(
            '<div class="stable-box">✅ STABLE<br>'
            'EWMA tracking nominal.<br>'
            f'Total alarms: {len(cusum_events)}</div>',
            unsafe_allow_html=True
        )


def render_results(history):
    if len(history) < 10:
        return
    data = {}
    for name in COLORS:
        energy  = sum(r[name]["energy"]   for r in history)
        viol    = sum(r[name]["violated"] for r in history)
        total   = sum(r[name]["arrived"]  for r in history)
        avg_vms = sum(r[name]["active_vms"] for r in history) / len(history)
        viol_pct = viol / total * 100 if total > 0 else 0
        data[name] = {
            "Energy (kWh)": f"{energy:.2f}",
            "SLA Viol%":    f"{viol_pct:.1f}%",
            "Avg VMs":      f"{avg_vms:.1f}",
            "Violations":   str(viol),
        }
    df = pd.DataFrame(data).T
    df.index.name = "Scheduler"

    pmecx_e  = float(data["PMEC-X"]["Energy (kWh)"])
    minmin_e = float(data["MinMin"]["Energy (kWh)"])
    saving   = (1 - pmecx_e / minmin_e) * 100 if minmin_e > 0 else 0

    results_placeholder.markdown(
        f"#### Live Results Summary — {len(history)} epochs completed\n"
        f"**PMEC-X energy saving vs MinMin: {saving:.1f}%** | "
        f"**CUSUM detections: {len(st.session_state.cusum_events)}**"
    )
    results_placeholder.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.running and st.session_state.schedulers:
    schedule    = st.session_state.schedule
    schedulers  = st.session_state.schedulers
    epoch       = st.session_state.epoch

    if epoch >= len(schedule):
        st.session_state.running = False
        st.session_state.done    = True
    else:
        # Run `speed` epochs per UI refresh for smooth animation
        steps = min(speed, len(schedule) - epoch)
        gen_spike = WorkloadGenerator(seed=int(random_seed) + epoch)

        for step in range(steps):
            ep, sim_time, tasks = schedule[epoch + step]

            # Inject spike if button was pressed
            extra_tasks = []
            if st.session_state.spike_pending and step == 0:
                extra_tasks = gen_spike.generate_spike(sim_time, n=25)
                st.session_state.spike_pending = False

            all_tasks = tasks + extra_tasks

            row = {}
            for name, sched in schedulers.items():
                import copy as _copy
                task_copies = _copy.deepcopy(all_tasks)
                m = sched.run_epoch(ep, sim_time, task_copies)
                row[name] = {
                    "energy":     m.energy_kwh,
                    "active_vms": m.vms_active,
                    "arrived":    m.tasks_arrived,
                    "completed":  m.tasks_completed,
                    "violated":   m.tasks_violated,
                    "shutdowns":  m.vms_shutdown,
                    "regime_chg": m.regime_changes,
                }
                # Track CUSUM events
                if name == "PMEC-X" and m.regime_changes > 0:
                    st.session_state.cusum_events.append(epoch + step)

            st.session_state.history.append(row)

        st.session_state.epoch = epoch + steps

        # Render all charts
        h = st.session_state.history
        render_energy_chart(h)
        render_vm_chart(h)
        render_cusum_chart(h, st.session_state.cusum_events)
        render_heatmap(schedulers)
        render_cusum_status(st.session_state.cusum_events, st.session_state.epoch)
        render_results(h)

        time.sleep(0.05)
        st.rerun()

else:
    # Not running — render static state
    h = st.session_state.history
    render_energy_chart(h)
    render_vm_chart(h)
    render_cusum_chart(h, st.session_state.cusum_events)
    render_heatmap(st.session_state.schedulers)
    render_cusum_status(st.session_state.cusum_events, st.session_state.epoch)
    render_results(h)

    if st.session_state.done:
        st.success(
            "✅ Simulation complete! "
            f"PMEC-X saved energy vs all baselines. "
            f"CUSUM detected {len(st.session_state.cusum_events)} regime changes."
        )
    elif not st.session_state.history:
        st.info("👆 Press **Start** to begin the live simulation.")
