from __future__ import annotations

from html import escape

import streamlit as st

from proc_rosetta.inference import DecodeResult


def render_decoder_steps(result: DecodeResult) -> None:
    colors = {
        "operator": "#8d7bff",
        "arity": "#5677d8",
        "activity": "#30c6b0",
        "tau": "#8892a8",
        "boundary": "#ffad5a",
    }

    def category(token: str) -> str:
        if token in {"SEQ", "XOR", "AND", "LOOP"}:
            return "operator"
        if token.startswith("ARITY_"):
            return "arity"
        if token.startswith("A") and token[1:].isdigit():
            return "activity"
        if token == "TAU":
            return "tau"
        return "boundary"

    ribbon = " ".join(
        f'<span style="display:inline-block;background:{colors[category(token)]};color:white;'
        f'padding:.28rem .48rem;margin:.14rem;border-radius:.38rem;font:600 .78rem monospace">'
        f"{escape(token)}</span>"
        for token in result.token_names
    )
    st.markdown(ribbon, unsafe_allow_html=True)
    rows = [
        {
            "Step": step.step_index,
            "Grammar state": step.grammar_state,
            "Chosen": step.chosen_token,
            "Pre-budget score": step.selected_pre_budget_probability,
            "Conditional score": step.selected_conditional_probability,
            "Pre-budget top": step.pre_budget_top_token,
            "Budget active": step.budget_mask_active,
            "Argmax overridden": step.argmax_overridden,
            "Closure slack": step.completion_slack,
            "Valid choices": len(step.valid_next_tokens),
            "Open slots": step.open_child_slots,
            "Subtree position": step.subtree_position,
            "EOS": step.eos_emitted,
            "Top valid": ", ".join(f"{name} {score:.2%}" for name, score in step.top_valid_token_scores),
            "Masked high scores": ", ".join(
                f"{name} {score:.2%}" for name, score in step.invalid_high_scoring_tokens
            ),
        }
        for step in result.steps
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
