from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from proc_rosetta.devices import default_device
from proc_rosetta.inference import LoadedCheckpoint, list_trusted_checkpoints, load_trusted_checkpoint
from proc_rosetta_ui.app_state import initialize_state, reset_inference_for_checkpoint


PETRI_WARNING = (
    "The current external Petri-net encoder path does not use visible transition labels. "
    "PNML-derived decodes use canonical labels and do not preserve source activity names."
)


def configure_page(title: str, icon: str) -> None:
    st.set_page_config(
        page_title=f"{title} · ProcRosetta",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state(st.session_state)
    st.markdown(APP_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="pr-hero">
          <div class="pr-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trusted_checkpoint_directory() -> Path:
    configured = os.environ.get("PROC_ROSETTA_CHECKPOINT_DIR")
    return Path(configured).expanduser().resolve() if configured else (ROOT / "checkpoints").resolve()


@st.cache_resource(show_spinner="Loading trusted checkpoint…")
def _load_checkpoint_cached(
    path: str,
    directory: str,
    device: str,
    file_fingerprint: str,
) -> LoadedCheckpoint:
    return load_trusted_checkpoint(path, trusted_directory=directory, device=device)


def checkpoint_sidebar(*, required: bool = True) -> LoadedCheckpoint | None:
    directory = trusted_checkpoint_directory()
    paths = list_trusted_checkpoints(directory)
    with st.sidebar:
        st.markdown("### Active checkpoint")
        st.caption("Trusted server-side files only · checkpoint upload is disabled")
        if not paths:
            st.error(f"No `.pt` checkpoints found in `{directory}`.")
            if required:
                st.stop()
            return None
        saved = st.session_state.get("active_checkpoint_path")
        default_index = next((i for i, path in enumerate(paths) if str(path) == saved), 0)
        selected = st.selectbox(
            "Checkpoint",
            options=paths,
            index=default_index,
            format_func=lambda path: path.name,
            key="checkpoint_selector",
        )
        device = st.selectbox(
            "Inference device",
            [default_device(), "cpu"] if default_device() != "cpu" else ["cpu"],
            key="checkpoint_device",
        )
        st.session_state["active_checkpoint_path"] = str(selected)
        try:
            stat = selected.stat()
            checkpoint = _load_checkpoint_cached(
                str(selected),
                str(directory),
                device,
                f"{stat.st_size}:{stat.st_mtime_ns}",
            )
        except Exception as exc:
            st.error(f"Checkpoint load failed: {type(exc).__name__}: {exc}")
            if required:
                st.stop()
            return None
        reset_inference_for_checkpoint(st.session_state, checkpoint.metadata.identifier)
        metadata = checkpoint.metadata
        st.caption(
            f"Epoch {metadata.epoch or '—'} · {metadata.latent_dimension}D latent · "
            f"{metadata.maximum_activities} activities"
        )
        with st.expander("Checkpoint facts"):
            st.json(
                {
                    "identifier": metadata.identifier,
                    "type": metadata.checkpoint_type,
                    "epoch": metadata.epoch,
                    "latent_dimension": metadata.latent_dimension,
                    "hidden_dimension": metadata.hidden_dimension,
                    "maximum_activities": metadata.maximum_activities,
                    "maximum_tree_arity": metadata.maximum_tree_arity,
                    "best_validation_loss": metadata.best_validation_loss,
                }
            )
        st.divider()
        st.caption("Five views · one shared workspace")
    return checkpoint


def checkpoint_metadata_dict(checkpoint: LoadedCheckpoint) -> dict[str, Any]:
    return asdict(checkpoint.metadata)


def status_pill(label: str, ok: bool | None) -> str:
    state = "neutral" if ok is None else ("good" if ok else "bad")
    icon = "•" if ok is None else ("✓" if ok else "×")
    return f'<span class="pr-pill {state}">{icon} {label}</span>'


APP_CSS = """
<style>
:root {
  --pr-canvas: #f7f9fc;
  --pr-ink: #172033;
  --pr-muted: #667085;
  --pr-panel: #ffffff;
  --pr-border: #dfe5ee;
  --pr-violet: #6755d9;
  --pr-teal: #087f73;
  --pr-orange: #b86213;
  --pr-shadow: 0 8px 24px rgba(24, 38, 64, .055);
}
[data-testid="stAppViewContainer"] {
  background: var(--pr-canvas);
  color: var(--pr-ink);
}
[data-testid="stHeader"] { background: rgba(247, 249, 252, .94); }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--pr-border); }
[data-testid="stSidebarContent"] { color: var(--pr-ink); }
.block-container { max-width: 1480px; padding-top: 2.25rem; padding-bottom: 4rem; }
.pr-hero { padding: .4rem 0 1.8rem; border-bottom: 1px solid var(--pr-border); margin-bottom: 1.5rem; }
.pr-eyebrow { color: var(--pr-teal); font: 700 .74rem/1.3 sans-serif; letter-spacing: .16em; text-transform: uppercase; }
.pr-hero h1 { color: var(--pr-ink); font: 650 clamp(2rem, 4vw, 3.6rem)/1.04 sans-serif; letter-spacing: -.045em; margin: .35rem 0 .65rem; }
.pr-hero p { color: var(--pr-muted); font-size: 1.05rem; max-width: 880px; margin: 0; }
.pr-card { background: var(--pr-panel); border: 1px solid var(--pr-border); border-radius: 14px; padding: 1rem 1.15rem; box-shadow: var(--pr-shadow); }
.pr-kicker { color: var(--pr-muted); font-size: .78rem; letter-spacing: .09em; text-transform: uppercase; }
.pr-value { color: var(--pr-ink); font-size: 1.65rem; font-weight: 650; letter-spacing: -.03em; }
.pr-pill { display:inline-flex; align-items:center; gap:.28rem; border:1px solid var(--pr-border); border-radius:999px; padding:.32rem .62rem; margin:.15rem .22rem .15rem 0; font-size:.78rem; }
.pr-pill.good { color:#067064; background:#e9f8f4; border-color:#b9e6dc; }
.pr-pill.bad { color:#b42318; background:#fff1f0; border-color:#f2c8c4; }
.pr-pill.neutral { color:#536078; background:#f3f5f8; }
div[data-testid="stMetric"] { background:var(--pr-panel); border:1px solid var(--pr-border); padding:.8rem 1rem; border-radius:12px; box-shadow:var(--pr-shadow); }
div[data-testid="stDataFrame"] { background:#ffffff; border:1px solid var(--pr-border); border-radius:12px; overflow:hidden; }
[data-testid="stGraphVizChart"], [data-testid="stVegaLiteChart"] { background:#ffffff; border:1px solid var(--pr-border); border-radius:12px; padding:.6rem; }
[data-testid="stFileUploaderDropzone"] { background:#ffffff; border-color:var(--pr-border); }
[data-testid="stExpander"] { background:#ffffff; border-color:var(--pr-border); }
.stButton button[kind="primary"] { background:#6755d9; border-color:#6755d9; }
.stButton button[kind="primary"]:hover { background:#5846c4; border-color:#5846c4; }
.stButton button[kind="secondary"] { background:#ffffff; color:var(--pr-ink); border-color:var(--pr-border); }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:.45rem .8rem; color:var(--pr-muted); }
.stTabs [aria-selected="true"] { color:var(--pr-violet); background:#eeebff; }
.pr-card div:not(.pr-kicker):not(.pr-value) { color:var(--pr-muted); }
a { color:#5846c4; }
hr { border-color:var(--pr-border); }
code { color:#5846c4 !important; background:#f0eefc !important; }
</style>
"""
