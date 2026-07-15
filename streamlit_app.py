from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st


navigation = st.navigation(
    [
        st.Page("proc_rosetta_ui/home_page.py", title="Studio overview", icon="🏠", default=True),
        st.Page("pages/01_workspace.py", title="Artifact workspace", icon="📂"),
        st.Page("pages/02_translation.py", title="Translation studio", icon="🔀"),
        st.Page("pages/03_latent_explorer.py", title="Latent explorer", icon="🧭"),
        st.Page("pages/05_checkpoint.py", title="Checkpoint dashboard", icon="📊"),
    ]
)
navigation.run()
