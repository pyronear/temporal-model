"""Streamlit viewer over the eval reporting tree (read-only).

Reads only ``data/08_reporting/<source>/vit_dinov2_finetune/{results.json,details/,
sequences/}`` plus the frame images those records point at; it never runs the model.
Run with ``streamlit run src/temporal_model/eval/app.py`` (or ``make app``).

Left pane selects the source (train / val / pyro-annotator). The main pane lists the
source's sequences in an error-coloured, filterable table; selecting a row opens an
autoplaying (pausable) frame viewer with the YOLO bboxes overlaid, the per-tube
timeline, and each kept tube's stabilized crop synced to the current frame, plus the
temporal-model decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from temporal_model.eval.outcomes import performance_summary
from temporal_model.eval.render import (
    CORRECTNESS,
    correctness_label,
    crop_around_bbox,
    day_of,
    draw_bboxes,
    frame_bboxes_by_input_index,
    legend_html,
    processed_to_input_index,
    row_background,
    triggering_tube_ids,
    tube_color,
    tube_input_boxes,
    tube_timeline_df,
)

REPORTING = Path("data/08_reporting")
MODEL_NAME = "vit_dinov2_finetune"
PLAY_FPS = 1  # autoplay speed (frames/sec); fixed, no UI control


def reporting_dirs() -> list[Path]:
    """Every <source>/vit_dinov2_finetune reporting dir that has a results.json."""
    if not REPORTING.exists():
        return []
    return sorted(p.parent for p in REPORTING.glob(f"*/{MODEL_NAME}/results.json"))


def load_results() -> pd.DataFrame:
    """Concatenate results.json across all sources (empty frame if none)."""
    frames = []
    for d in reporting_dirs():
        rows = json.loads((d / "results.json").read_text())
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_details(source: str, key: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "details" / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_sequence_view(source: str, key: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "sequences" / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_model_config(source: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "model_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _tube_timeline_chart(
    alt,
    tube_rows,
    n,
    trigger,
    current,
    color_map,
    trigger_tube_id=None,
    would_ids=frozenset(),
):  # pragma: no cover
    """One colour-coded bar row per tube + trigger/current rules (Altair).

    The decisive trigger tube's bars get a thick dark outline; tubes that would
    also have crossed the threshold get a thinner grey outline.
    """
    order = [label for label, _ in tube_rows]

    def _level(label: str) -> str:
        tid = int(label[1:])
        if tid == trigger_tube_id:
            return "decisive"
        return "would" if tid in would_ids else "none"

    df = tube_timeline_df(tube_rows)
    df["level"] = df["tube"].map(_level)
    levels = ["decisive", "would", "none"]
    xscale = alt.Scale(domain=[0, n], nice=False)
    bars = (
        alt.Chart(df)
        .mark_bar(height=16, cornerRadius=3)
        .encode(
            x=alt.X(
                "frame:Q",
                title="frame",
                scale=xscale,
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            x2="frame_end:Q",
            y=alt.Y("tube:N", title=None, sort=order, axis=alt.Axis(labelFontSize=13)),
            color=alt.Color(
                "tube:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[color_map[o] for o in order]),
                legend=None,
            ),
            stroke=alt.Stroke(
                "level:N",
                scale=alt.Scale(domain=levels, range=["#111", "#666", "transparent"]),
                legend=None,
            ),
            strokeWidth=alt.StrokeWidth(
                "level:N",
                scale=alt.Scale(domain=levels, range=[2.5, 1.2, 0]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tube:N", title="tube"),
                alt.Tooltip("frame:Q", title="frame"),
                alt.Tooltip("confidence:Q", title="bbox score", format=".2f"),
            ],
        )
    )
    layers = [bars]
    if trigger is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [trigger + 0.5]}))
            .mark_rule(color="#c62828", size=2)
            .encode(x=alt.X("x:Q", scale=xscale, axis=None))
        )
    layers.append(
        alt.Chart(pd.DataFrame({"x": [current + 0.5]}))
        .mark_rule(color="#111", strokeDash=[4, 3], size=2)
        .encode(x=alt.X("x:Q", scale=xscale, title="frame"))
    )
    return alt.layer(*layers).properties(
        height=max(90, len(tube_rows) * 34),
        autosize={"type": "fit-x", "contains": "padding"},
    )


@st.fragment(run_every=1.0 / PLAY_FPS)
def _drilldown(source: str, key: str, row: pd.Series) -> None:  # pragma: no cover
    # A fragment with run_every so autoplay reruns ONLY this drill-down on a timer
    # (not the whole page). Selecting a new sequence replaces it cleanly.
    import altair as alt  # noqa: PLC0415

    details = load_details(source, key)
    view = load_sequence_view(source, key)
    frame_files = [Path(p) for p in view.get("frames", [])]
    bbmap = frame_bboxes_by_input_index(details)
    padded = details.get("preprocessing", {}).get("padded_frame_indices", [])
    kept = details.get("tubes", {}).get("kept", [])
    trigger_tube_id = details.get("decision", {}).get("trigger_tube_id")
    would_ids = triggering_tube_ids(details)

    def trigger_state(tid: int | None) -> str | None:
        """'decisive' for the firing tube, 'would' for others over threshold."""
        if tid is not None and tid == trigger_tube_id:
            return "decisive"
        return "would" if tid in would_ids else None

    n = len(frame_files)
    trig_raw = row["trigger_frame_index"]
    trig = (
        processed_to_input_index(int(trig_raw), padded) if pd.notna(trig_raw) else None
    )

    is_keep = row["decision"] == "keep"
    verdict = "💨 KEEP (smoke)" if is_keep else "🚫 DISCARD (no smoke)"
    vcol, idcol = st.columns([4, 1], vertical_alignment="center")
    vcol.subheader(verdict)
    idcol.code(key, language=None)  # native hover copy-to-clipboard button
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("model verdict", row["decision"])
    c2.metric("correctness", correctness_label(row["outcome"]))
    c3.metric("trigger frame", "—" if trig is None else str(trig))
    prob = row["probability"]
    c4.metric("probability", f"{prob:.3f}" if pd.notna(prob) else "—")
    if not n:
        return

    # Playback: compact toggle above a full-width timeline + slider. While playing
    # we advance the slider's session_state value BEFORE the slider widget is
    # instantiated (the only point a keyed value may be modified).
    frame_key = f"frame_{key}"
    st.session_state.setdefault(frame_key, 0)
    playing = st.toggle("▶ play", value=True, key=f"play_{key}")
    if playing:
        st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n
    cur = st.session_state[frame_key] % n

    tube_rows = [
        (
            f"T{t['tube_id']}",
            {idx: conf for idx, _, conf in tube_input_boxes(t, padded)},
        )
        for t in kept
    ]
    color_map = {f"T{t['tube_id']}": tube_color(t["tube_id"]) for t in kept}
    if tube_rows:
        st.altair_chart(
            _tube_timeline_chart(
                alt, tube_rows, n, trig, cur, color_map, trigger_tube_id, would_ids
            ),
            width="stretch",
        )
    else:
        st.info("no smoke tubes extracted")
    i = st.slider("frame", 0, n - 1, key=frame_key) if n > 1 else 0

    frame_col, tubes_col = st.columns([2, 1])
    frame_boxes = [
        (bbox, conf, tube_color(tid), trigger_state(tid))
        for bbox, conf, tid in bbmap.get(i, [])
    ]
    frame_col.image(
        draw_bboxes(frame_files[i], frame_boxes),
        caption=f"frame {i + 1}/{n} — {len(frame_boxes)} detection(s)",
        width="stretch",
    )

    # Each tube crop is the tube's fixed stabilized window, synced to frame i.
    tubes_col.markdown(f"**tubes @ frame {i}** (stabilized crop)")
    for tube in kept:
        at_frame = {idx: bbox for idx, bbox, _ in tube_input_boxes(tube, padded)}
        color = tube_color(tube["tube_id"])
        tprob = tube.get("probability")
        stat = (
            f"prob {tprob:.2f}" if tprob is not None else f"logit {tube['logit']:.2f}"
        )
        state = trigger_state(tube["tube_id"])
        if state == "decisive":
            trig_badge = " ⚡<b>triggered</b>"
            if trig is not None:
                trig_badge += f" (frame {trig})"
        elif state == "would":
            trig_badge = " <span style='color:#888'>⚡ would trigger</span>"
        else:
            trig_badge = ""
        chip = f"<b style='color:{color}'>● T{tube['tube_id']}</b>"
        tubes_col.markdown(f"{chip} · {stat}{trig_badge}", unsafe_allow_html=True)
        window = tube.get("stabilized_window")
        crop_box = tuple(window) if window else at_frame.get(i)
        if crop_box is not None:
            tubes_col.image(crop_around_bbox(frame_files[i], crop_box), width=220)
        else:
            tubes_col.caption("inactive at this frame")


def render_performance(df: pd.DataFrame) -> None:  # pragma: no cover - Streamlit UI
    """Three headline metric cards over the labeled (smoke/fp) rows of ``df``.

    ``df`` is expected to already be narrowed to one source. Renders nothing when
    that source has no labeled sequences.
    """
    s = performance_summary(df)
    if s["n_labeled"] == 0:
        return
    st.caption(
        f"Model performance — {s['n_labeled']} labeled sequences "
        f"({s['n_fp']} fp · {s['n_smoke']} smoke)"
    )
    cards = (
        ("Recall (smoke kept)", s["recall"], f"{s['kept_smoke']}/{s['n_smoke']}"),
        ("FP filtered", s["specificity"], f"{s['discarded_fp']}/{s['n_fp']}"),
        (
            "Precision",
            s["precision"],
            f"{s['kept_smoke']}/{s['kept_smoke'] + s['kept_fp']}",
        ),
    )
    for col, (label, value, frac) in zip(st.columns(3), cards, strict=True):
        if value is None:
            col.metric(label, "—")
        else:
            col.metric(label, f"{value:.1%}")
            col.caption(frac)


# Hover help (info-on-hover) for each headline model-config param. Kept quote-free
# so it embeds safely in an HTML title attribute.
MODEL_CONFIG_HELP = {
    "detector": "Stage-1 object detector that proposes smoke boxes per frame; "
    "value is its source (e.g. a HuggingFace repo).",
    "variant": "Packaged model variant name.",
    "train sha": "Git commit of the training run that produced this model.",
    "aggregation": "Rule combining per-tube scores into the sequence keep/discard "
    "decision (max_logit or logistic).",
    "threshold": "Decision threshold on the aggregated max-logit score.",
    "logistic threshold": "Probability threshold applied when aggregation is logistic.",
    "stabilize": "If true, each tube is cropped from one fixed window (stabilized) "
    "instead of tracking the per-frame bbox.",
    "context factor": "How much the bbox is expanded before cropping the classifier "
    "patch (more context).",
    "max frames": "The input sequence is truncated to its first N frames before "
    "detection; also caps frames per tube fed to the temporal classifier.",
    "pad": "Short input sequences are padded up to a minimum number of frames "
    "(duplicating first/last frames) before detection, using this strategy.",
}


def render_model_config(source: str) -> None:  # pragma: no cover - Streamlit UI
    """Sidebar panel: headline model fields with info-on-hover."""
    cfg = load_model_config(source)
    # Spacer below the source selector drops the panel toward the bottom while
    # keeping the source selector pinned at the top (the spacer sits after it).
    st.sidebar.markdown("<div style='margin-top:12vh'></div>", unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.caption("Model config · hover a field for details")
    if not cfg:
        st.sidebar.caption("model config unavailable")
        return
    detector = (cfg.get("detector") or {}).get("source", "—")
    decision = cfg.get("decision") or {}
    model_input = cfg.get("model_input") or {}
    infer = cfg.get("infer") or {}
    classifier = cfg.get("classifier") or {}
    sha = (cfg.get("train_git_sha") or "")[:8] or "—"
    fields = [
        ("detector", f"<code>{detector}</code>"),
        ("variant", cfg.get("variant", "—")),
        ("train sha", f"<code>{sha}</code>"),
        ("aggregation", decision.get("aggregation", "—")),
        ("threshold", decision.get("threshold", "—")),
        ("logistic threshold", decision.get("logistic_threshold", "—")),
        ("stabilize", model_input.get("stabilize", "—")),
        ("context factor", model_input.get("context_factor", "—")),
        ("max frames", classifier.get("max_frames", "—")),
        (
            "pad",
            f"{infer.get('pad_strategy', '—')} / min "
            f"{infer.get('pad_to_min_frames', '—')}",
        ),
    ]
    # Clean stacked rows: muted label over value, whole row hoverable (title) for
    # the param's description — no underline noise.
    rows = [
        f'<div title="{MODEL_CONFIG_HELP[label]}" style="cursor:help;'
        f'margin-bottom:6px;line-height:1.25">'
        f'<span style="color:#8a8a8a;font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:.04em">{label}</span><br>'
        f'<span style="font-size:0.9rem">{value}</span></div>'
        for label, value in fields
    ]
    st.sidebar.markdown("".join(rows), unsafe_allow_html=True)


def main() -> None:  # pragma: no cover - Streamlit UI
    st.set_page_config(page_title="Eval Qualitative Viewer", layout="wide")
    st.title("Eval Qualitative Viewer")

    df = load_results()
    if df.empty:
        st.warning("No results yet. Run `uv run dvc repro` (or the evaluate CLI).")
        return

    st.sidebar.header("Select")
    # pyro-annotator first (default), then any other source alphabetically.
    sources = sorted(
        df["source"].dropna().unique(),
        key=lambda s: (s != "pyro-annotator", s),
    )
    source = st.sidebar.selectbox("source", sources, key="source")
    render_model_config(source)
    view = df[df["source"] == source].reset_index(drop=True)
    has_org = view["organization_name"].notna().any()
    has_cam = view["camera_name"].notna().any()

    render_performance(view)

    has_day = False
    if "started_at" in view.columns:
        view = view.assign(day=view["started_at"].map(day_of))
        view = view.sort_values(
            "started_at", ascending=False, na_position="last"
        ).reset_index(drop=True)
        has_day = view["day"].ne("unknown").any()

    # Full-width title placeholder (filled after filtering) + a horizontal bar
    # where the legend stretches to fill, pushing the filter popover to the right.
    title_ph = st.empty()
    bar = st.container(horizontal=True, vertical_alignment="center")
    legend_box = bar.container()
    with bar.popover("🔎 filter"):
        if has_org:
            orgs = sorted(view["organization_name"].dropna().unique())
            org = st.selectbox("organization", ["all", *orgs], key="f_org")
        else:
            org = "all"
        if has_cam:
            cams = sorted(view["camera_name"].dropna().unique())
            camera = st.selectbox("camera", ["all", *cams], key="f_cam")
        else:
            camera = "all"
        f_gt = st.selectbox(
            "ground truth", ["all", "smoke", "fp", "unknown"], key="f_gt"
        )
        f_mv = st.selectbox("model verdict", ["all", "keep", "discard"], key="f_mv")
        f_corr = st.selectbox(
            "correctness", ["all", *CORRECTNESS.values()], key="f_corr"
        )
    if org != "all":
        view = view[view["organization_name"] == org]
    if camera != "all":
        view = view[view["camera_name"] == camera]
    if f_gt != "all":
        view = view[view["label"] == f_gt]
    if f_mv != "all":
        view = view[view["decision"] == f_mv]
    if f_corr != "all":
        view = view[view["outcome"].map(correctness_label) == f_corr]

    title_ph.subheader(f"{len(view)} sequences — {source}")
    if view.empty:
        return

    display = view.assign(correctness=view["outcome"].map(correctness_label)).rename(
        columns={
            "label": "ground truth",
            "decision": "model verdict",
            "camera_name": "camera",
        }
    )
    cols = ["ground truth", "model verdict", "correctness", "score", "probability"]
    if has_cam:
        cols = ["camera", *cols]
    if has_day:
        cols = ["day", *cols]

    def _style_row(r):
        bg = row_background(r["model verdict"], r["correctness"])
        return [f"background-color: {bg}; color: #111"] * len(cols)

    styled = display[cols].style.apply(_style_row, axis=1)
    legend_box.markdown(legend_html(), unsafe_allow_html=True)
    event = st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="seqtable",
    )
    # Persist the selection: autoplay's rerun doesn't carry the table's selection
    # event, so without this the viewer would snap back to the first row each tick.
    rows = event.selection.rows
    if rows and rows[0] < len(view):
        st.session_state["selected_key"] = view.iloc[rows[0]]["key"]
    selected = st.session_state.get("selected_key")
    if selected not in set(view["key"]):
        selected = view.iloc[0]["key"]
    _drilldown(source, selected, view[view["key"] == selected].iloc[0])


if __name__ == "__main__":  # pragma: no cover
    main()
