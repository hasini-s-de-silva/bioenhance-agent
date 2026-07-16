"""BioEnhance Agent — Streamlit interface.

Research prototype for evidence-grounded formulation strategy generation.

The panel order is deliberate: descriptors, then retrieved evidence, then the model's
reasoning. The user should be able to see what the model was given before they see
what it concluded.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.descriptors import (  # noqa: E402
    InvalidSmilesError,
    NameResolutionError,
    compute_descriptors,
    resolve_input,
)
from src.llm_agent import FormulationAgent  # noqa: E402
from src.retrieval import build_query, get_index, retrieve_relevant_evidence  # noqa: E402
from src.solubility_model import get_predictor  # noqa: E402

st.set_page_config(page_title="BioEnhance Agent", page_icon="🧪", layout="wide")

CONFIDENCE_COLOUR = {"high": "#1a7f37", "medium": "#9a6700", "low": "#cf222e"}
RISK_COLOUR = {"low": "#1a7f37", "moderate": "#9a6700", "high": "#cf222e"}


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource
def _predictor():
    return get_predictor()


@st.cache_resource
def _index():
    return get_index()


@st.cache_data(show_spinner=False)
def _descriptors_for(smiles: str):
    return compute_descriptors(smiles)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def descriptor_gauge(desc) -> go.Figure:
    """Show each descriptor against the threshold that would flag it."""
    metrics = [
        ("MW", desc.molecular_weight, 500, 1000),
        ("cLogP", desc.clogp, 5, 10),
        ("TPSA", desc.tpsa, 140, 250),
        ("HBD", desc.h_bond_donors, 5, 12),
        ("HBA", desc.h_bond_acceptors, 10, 20),
        ("RotB", desc.rotatable_bonds, 10, 20),
    ]
    labels = [m[0] for m in metrics]
    # Normalise each value against its own rule-of-five style threshold so a single
    # bar chart can carry descriptors on wildly different scales.
    fractions = [min(m[1] / m[2], 1.6) for m in metrics]
    colours = ["#cf222e" if f > 1 else "#1a7f37" for f in fractions]
    text = [f"{m[1]:.1f} / {m[2]}" for m in metrics]

    fig = go.Figure(
        go.Bar(x=fractions, y=labels, orientation="h", marker_color=colours, text=text,
               textposition="auto", hoverinfo="skip")
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#57606a",
                  annotation_text="threshold", annotation_position="top")
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="value / threshold",
        showlegend=False,
        title="Descriptor thresholds (red = flagged)",
    )
    return fig


def solubility_figure(sol) -> go.Figure:
    """Plot the predicted log S with the model's own RMSE as an error bar."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[sol.log_s],
            y=["log S"],
            error_x=dict(type="data", array=[sol.model_rmse], color="#57606a", width=8),
            mode="markers",
            marker=dict(size=18, color=RISK_COLOUR[sol.risk.value]),
            name="predicted",
            hovertemplate=f"log S {sol.log_s:.2f} ± {sol.model_rmse:.2f}<extra></extra>",
        )
    )
    fig.add_vrect(x0=-12, x1=-4, fillcolor="#cf222e", opacity=0.08, line_width=0,
                  annotation_text="poorly soluble", annotation_position="top left")
    fig.add_vrect(x0=-4, x1=-2, fillcolor="#9a6700", opacity=0.08, line_width=0,
                  annotation_text="moderate", annotation_position="top")
    fig.add_vrect(x0=-2, x1=2, fillcolor="#1a7f37", opacity=0.08, line_width=0,
                  annotation_text="soluble", annotation_position="top right")
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="predicted log S (mol/L), error bar = model held-out RMSE",
        xaxis_range=[-12, 2],
        showlegend=False,
        title="Baseline ESOL solubility model",
    )
    return fig


def render_compound_panel(desc, sol) -> None:
    st.subheader("1 · Compound assessment")
    st.caption("Every number here is calculated by RDKit or the ESOL model — none is LLM-generated.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecular weight", f"{desc.molecular_weight:.1f}")
    c2.metric("Calculated logP", f"{desc.clogp:.2f}")
    c3.metric("TPSA", f"{desc.tpsa:.1f}")
    c4.metric("Lipinski violations", desc.lipinski_violations)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H-bond donors", desc.h_bond_donors)
    c2.metric("H-bond acceptors", desc.h_bond_acceptors)
    c3.metric("Rotatable bonds", desc.rotatable_bonds)
    c4.metric("Fraction sp3", f"{desc.fraction_csp3:.2f}")

    risk = sol.risk.value
    st.markdown(
        f"**Predicted aqueous solubility:** {sol.classification} — "
        f"log S {sol.log_s:.2f} ± {sol.log_s_std:.2f} (≈ {sol.mg_per_ml:.4g} mg/mL)  \n"
        f"**Solubility risk:** :{'red' if risk == 'high' else 'orange' if risk == 'moderate' else 'green'}[{risk.upper()}]"
    )

    if sol.applicability_warning:
        st.warning(f"Applicability domain: {sol.applicability_warning}")

    st.info(
        f"The backing model's held-out RMSE is {sol.model_rmse:.2f} log units, so read "
        f"log S {sol.log_s:.2f} as roughly {sol.log_s - sol.model_rmse:.1f} to "
        f"{sol.log_s + sol.model_rmse:.1f}. It predicts intrinsic (neutral-species) "
        "solubility and does not model ionisation, salt forms or solid-state effects."
    )

    left, right = st.columns(2)
    left.plotly_chart(descriptor_gauge(desc), use_container_width=True)
    right.plotly_chart(solubility_figure(sol), use_container_width=True)

    with st.expander("Rule-based flags"):
        st.write(f"- Lipinski rule of five: **{'pass' if desc.lipinski_pass else 'fail'}** "
                 f"({desc.lipinski_violations} violations)")
        st.write(f"- Veber rules (RotB ≤ 10, TPSA ≤ 140): **{'pass' if desc.veber_pass else 'fail'}**")
        st.write(f"- Canonical SMILES: `{desc.canonical_smiles}`")


def render_evidence_panel(evidence, query: str) -> None:
    st.subheader("2 · Retrieved evidence")
    st.caption(
        f"Retrieved with **{_index().backend}**. These are the only documents the model "
        "is permitted to cite."
    )

    with st.expander("Retrieval query built from the calculated descriptors"):
        st.code(query, language=None)

    if not evidence:
        st.warning("No evidence retrieved.")
        return

    for item in evidence:
        d = item.doc
        st.markdown(
            f"**[{d.id}]** [{d.title}]({d.url})  \n"
            f"*{d.source}* ({d.year}) · PMID {d.pmid} · similarity {item.score:.3f}  \n"
            f"`{', '.join(d.tags)}`"
        )
        with st.expander("Abstract"):
            st.write(d.text)


def render_assessment_panel(result) -> None:
    st.subheader("3 · Formulation strategy assessment")
    st.caption(f"Generated by: **{result.mode}**")

    g = result.grounding
    if not g.is_grounded:
        st.error(
            f"Grounding check FAILED. The model cited sources that were never retrieved: "
            f"{', '.join(g.hallucinated_ids)}. Treat this output as unreliable."
        )
    else:
        st.success(
            f"Grounding check passed — all {len(g.cited_ids)} citation(s) resolve to "
            "retrieved documents."
        )

    summary = result.assessment.compound_summary
    st.markdown(f"**Solubility risk:** `{summary.solubility_risk.value}`")
    if summary.main_drivers:
        st.markdown("**Main drivers:**")
        for d in summary.main_drivers:
            st.markdown(f"- {d}")

    st.markdown("#### Ranked strategies")
    strategies = result.assessment.ranked_strategies
    if not strategies:
        st.info(
            "No solubility-enabling formulation is indicated for this compound. "
            "That is a result, not a gap — see the uncertainty statement below."
        )

    by_id = {r.doc.id: r.doc for r in result.retrieved}
    for s in strategies:
        colour = CONFIDENCE_COLOUR[s.confidence.value]
        with st.container(border=True):
            st.markdown(
                f"**{s.rank}. {s.strategy}** &nbsp; "
                f"<span style='color:{colour};font-weight:600'>confidence: {s.confidence.value}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*Reason:* {s.rationale}")

            if s.supporting_sources:
                cites = []
                for sid in s.supporting_sources:
                    doc = by_id.get(sid)
                    cites.append(f"[{sid}]({doc.url})" if doc else f"`{sid}` (UNRESOLVED)")
                st.markdown(f"*Evidence:* {', '.join(cites)}")
            else:
                st.markdown("*Evidence:* :red[none — hypothesis only]")

            if s.limitations:
                st.markdown("*Limitations:*")
                for lim in s.limitations:
                    st.markdown(f"- {lim}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Missing information")
        for item in result.assessment.missing_information:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("#### Recommended next experiments")
        for item in result.assessment.recommended_experiments:
            st.markdown(f"- {item}")

    st.markdown("#### Overall uncertainty")
    st.warning(result.assessment.overall_uncertainty)

    with st.expander("Raw structured output (JSON)"):
        st.json(result.assessment.model_dump(mode="json"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.title("🧪 BioEnhance Agent")
    st.markdown(
        "**A tool-augmented LLM assistant for evidence-grounded oral drug formulation "
        "research.** Research prototype for evidence-grounded formulation strategy "
        "generation — it does not replace formulation scientists and does not predict "
        "clinical bioequivalence."
    )

    # Query params make an assessment deep-linkable and scriptable, e.g.
    #   ?name=itraconazole&run=1&backend=rulebased
    qp = st.query_params

    with st.sidebar:
        st.header("Compound input")
        name = st.text_input("Drug name", value=qp.get("name", ""), placeholder="itraconazole")
        smiles = st.text_area(
            "or SMILES", value=qp.get("smiles", ""), placeholder="CC(=O)Oc1ccccc1C(=O)O", height=80
        )
        st.caption("A SMILES string overrides the name.")

        st.header("Optional context")
        dose = st.text_input("Dose", value=qp.get("dose", ""), placeholder="100 mg")
        ph = st.text_input("Target pH", value=qp.get("ph", ""), placeholder="6.8")
        dosage_form = st.selectbox(
            "Dosage form",
            ["", "oral tablet", "oral capsule", "oral suspension", "powder for reconstitution"],
        )
        constraints = st.text_area(
            "Development constraints",
            placeholder="avoid organic solvents; must be manufacturable by hot-melt extrusion",
            height=80,
        )

        st.header("System")
        top_k = st.slider("Evidence documents to retrieve", 3, 12, int(qp.get("top_k", 6)))
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        options = ["auto", "anthropic", "rulebased"]
        backend = st.radio(
            "Reasoning backend",
            options,
            index=options.index(qp.get("backend", "auto")) if qp.get("backend") in options else 0,
            help="'rulebased' is a deterministic non-LLM baseline that needs no API key.",
        )
        if not has_key:
            st.caption("⚠️ No ANTHROPIC_API_KEY found — the LLM backend is unavailable.")

        run = st.button("Assess compound", type="primary", use_container_width=True)

    run = run or qp.get("run") == "1"

    if not run:
        st.info("Enter a drug name or SMILES in the sidebar, then press **Assess compound**.")
        with st.expander("What this tool does, and what it does not"):
            st.markdown(
                "**Does:** calculates molecular descriptors with RDKit; estimates aqueous "
                "solubility with a random forest trained on the Delaney ESOL dataset; "
                "retrieves real open-access PubMed abstracts; asks an LLM to rank "
                "bioenhancement strategies using *only* that supplied material; then "
                "verifies every citation resolves to a retrieved document.\n\n"
                "**Does not:** measure anything, predict bioequivalence or clinical "
                "performance, model ionisation or solid-state behaviour, or account for "
                "dose, stability and manufacturability. Its recommendations are hypotheses "
                "to screen experimentally."
            )
        return

    try:
        resolved_smiles, resolved_name = resolve_input(name, smiles)
    except NameResolutionError as exc:
        st.error(str(exc))
        return
    except InvalidSmilesError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Calculating descriptors…"):
        desc = _descriptors_for(resolved_smiles)
        sol = _predictor().predict(desc)

    render_compound_panel(desc, sol)
    st.divider()

    with st.spinner("Retrieving evidence…"):
        query = build_query(desc, sol, constraints=constraints or None,
                            dosage_form=dosage_form or None)
        evidence = retrieve_relevant_evidence(query, top_k=top_k)

    render_evidence_panel(evidence, query)
    st.divider()

    agent = FormulationAgent(backend=backend)
    with st.spinner(f"Generating assessment with {agent.mode_label}…"):
        try:
            result = agent.run(
                name=resolved_name,
                smiles=resolved_smiles,
                dose=dose or None,
                ph=ph or None,
                dosage_form=dosage_form or None,
                constraints=constraints or None,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - surface the failure to the user
            st.error(f"Assessment failed: {exc}")
            return

    render_assessment_panel(result)


if __name__ == "__main__":
    main()
