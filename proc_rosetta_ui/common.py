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
  --pr-ink: #e9eefb;
  --pr-muted: #9ba8bf;
  --pr-panel: rgba(18, 27, 47, .72);
  --pr-border: rgba(151, 166, 200, .18);
  --pr-violet: #8d7bff;
  --pr-teal: #30c6b0;
  --pr-orange: #ffad5a;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 82% 4%, rgba(124,110,246,.13), transparent 30rem),
    radial-gradient(circle at 12% 28%, rgba(48,198,176,.08), transparent 26rem),
    #0b1120;
}
[data-testid="stSidebar"] { background: #0d1527; border-right: 1px solid var(--pr-border); }
.block-container { max-width: 1480px; padding-top: 2.25rem; padding-bottom: 4rem; }
.pr-hero { padding: .4rem 0 1.8rem; border-bottom: 1px solid var(--pr-border); margin-bottom: 1.5rem; }
.pr-eyebrow { color: var(--pr-teal); font: 700 .74rem/1.3 sans-serif; letter-spacing: .16em; text-transform: uppercase; }
.pr-hero h1 { color: var(--pr-ink); font: 650 clamp(2rem, 4vw, 3.6rem)/1.04 sans-serif; letter-spacing: -.045em; margin: .35rem 0 .65rem; }
.pr-hero p { color: var(--pr-muted); font-size: 1.05rem; max-width: 880px; margin: 0; }
.pr-card { background: var(--pr-panel); border: 1px solid var(--pr-border); border-radius: 14px; padding: 1rem 1.15rem; }
.pr-kicker { color: var(--pr-muted); font-size: .78rem; letter-spacing: .09em; text-transform: uppercase; }
.pr-value { color: var(--pr-ink); font-size: 1.65rem; font-weight: 650; letter-spacing: -.03em; }
.pr-pill { display:inline-flex; align-items:center; gap:.28rem; border:1px solid var(--pr-border); border-radius:999px; padding:.32rem .62rem; margin:.15rem .22rem .15rem 0; font-size:.78rem; }
.pr-pill.good { color:#78e5c8; background:rgba(48,198,176,.10); }
.pr-pill.bad { color:#ff9d9d; background:rgba(255,110,110,.09); }
.pr-pill.neutral { color:#bdc8dc; }
div[data-testid="stMetric"] { background:var(--pr-panel); border:1px solid var(--pr-border); padding:.8rem 1rem; border-radius:12px; }
div[data-testid="stDataFrame"] { border:1px solid var(--pr-border); border-radius:12px; overflow:hidden; }
.stButton button[kind="primary"] { background:linear-gradient(110deg, #6958eb, #4f83ef); border:0; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:.45rem .8rem; }
code { color:#b9afff !important; }
</style>
"""
