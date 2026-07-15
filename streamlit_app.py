from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from proc_rosetta_ui.app_state import workspace
from proc_rosetta_ui.common import checkpoint_sidebar, configure_page, page_header


configure_page("Multimodal process studio", "◈")
checkpoint = checkpoint_sidebar(required=False)
page_header(
    "ProcRosetta / shared latent space",
    "See every transformation.",
    "Inspect event logs, process trees, and Petri nets before encoding; compare their shared "
    "representations; then watch a grammar-constrained decoder turn latent vectors back into "
    "validated process models.",
)

items = list(workspace(st.session_state).values())
encoded = sum(bool(item.encoding and item.encoding.mu and not item.encoding.errors) for item in items)
decoded = sum(bool(item.decodes) for item in items)
groups = len({item.process_group for item in items if item.process_group})

metrics = st.columns(4)
metrics[0].metric("Workspace artifacts", len(items))
metrics[1].metric("Embeddings ready", encoded)
metrics[2].metric("Decoded artifacts", decoded)
metrics[3].metric("Process groups", groups)

st.markdown("### Studio workflow")
cards = st.columns([1, 1, 1, 1, 1])
views = [
    ("01", "Artifact workspace", "Parse, preview, group, canonicalize, and encode.", "pages/01_workspace.py"),
    ("02", "Translation studio", "Decode latent means and validate every output stage.", "pages/02_translation.py"),
    ("03", "Latent explorer", "Inspect PCA, similarity, agreement, and neighbors.", "pages/03_latent_explorer.py"),
    ("04", "Quality evaluation", "Run progressive decode and behavior evaluation.", "pages/04_evaluation.py"),
    ("05", "Checkpoint dashboard", "Compare trusted runs and inspect training history.", "pages/05_checkpoint.py"),
]
for column, (number, title, copy, path) in zip(cards, views):
    with column:
        st.markdown(
            f'<div class="pr-card"><div class="pr-kicker">View {number}</div>'
            f'<div class="pr-value" style="font-size:1.18rem;margin:.35rem 0">{title}</div>'
            f'<div style="color:#9ba8bf;min-height:4.8rem">{copy}</div></div>',
            unsafe_allow_html=True,
        )
        st.page_link(path, label="Open view →", use_container_width=True)

st.markdown("### Representation path")
st.code(
    "artifact bytes  →  parsed source  →  canonical model input  →  μ, log σ²  →  "
    "shared latent comparison  →  grammar-masked process tree  →  derived Petri net  →  behavior",
    language=None,
)
if checkpoint is None:
    st.warning(
        "No trusted server-side checkpoint is available. Artifact parsing still works after a "
        "checkpoint is installed in the configured directory; checkpoint upload is intentionally disabled."
    )

