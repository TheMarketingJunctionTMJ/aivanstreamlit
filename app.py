from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from lib.anthropic_client import generate_text
from lib.export_docx import export_blog_docx
from lib.file_processing import extract_text_from_upload
from lib.prompts import (
    TONE_OPTIONS,
    insights_system_prompt,
    insights_user_prompt,
    outline_system_prompt,
    outline_user_prompt,
    revision_system_prompt,
    revision_user_prompt,
    section_system_prompt,
    section_user_prompt,
)

load_dotenv()

st.set_page_config(page_title="Streamlit Blog Studio", page_icon="✍️", layout="wide")


def init_state() -> None:
    defaults: dict[str, Any] = {
        "title": "",
        "topic": "",
        "audience": "",
        "keywords_text": "",
        "facts_text": "",
        "quotes_text": "",
        "research_notes": "",
        "tone": "Thought leadership",
        "target_words": 1200,
        "document_text": "",
        "document_insights": [],
        "outline_title": "",
        "outline": [],
        "sections_content": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def lines_to_list(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def current_inputs() -> dict[str, Any]:
    return {
        "title": st.session_state.title.strip() or st.session_state.topic.strip(),
        "topic": st.session_state.topic.strip(),
        "audience": st.session_state.audience.strip(),
        "keywords": lines_to_list(st.session_state.keywords_text),
        "facts": lines_to_list(st.session_state.facts_text),
        "quotes": lines_to_list(st.session_state.quotes_text),
        "research_notes": st.session_state.research_notes.strip(),
        "tone": st.session_state.tone,
        "target_words": int(st.session_state.target_words),
        "document_insights": st.session_state.document_insights,
    }


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


init_state()

st.title("✍️ Streamlit Blog Studio")
st.caption("Outline-first AI blog writing for internal content teams")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is missing. Add it in your environment before using the app.")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("1. Blog setup")
    st.text_input("Topic", key="topic", placeholder="e.g. How AI is changing recruitment marketing")
    st.text_input("Working title", key="title", placeholder="Optional title override")
    st.text_input("Audience", key="audience", placeholder="e.g. Talent leaders at mid-size companies")
    st.selectbox("Tone", TONE_OPTIONS, key="tone")
    st.slider("Target words", min_value=600, max_value=2500, step=100, key="target_words")
    st.text_area("SEO keywords (one per line)", key="keywords_text", height=120)

    st.subheader("2. Facts, quotes, and notes")
    st.text_area("Facts to include (one per line)", key="facts_text", height=140)
    st.text_area("Quotes to include verbatim (one per line)", key="quotes_text", height=140)
    st.text_area("Additional research notes", key="research_notes", height=140)

    st.subheader("3. Upload research")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, CSV, XLSX",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        if st.button("Extract insights from uploaded file"):
            try:
                raw_text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
                st.session_state.document_text = raw_text
                response = generate_text(
                    insights_system_prompt(),
                    insights_user_prompt(raw_text, st.session_state.topic or st.session_state.title or "blog topic"),
                    max_tokens=1800,
                )
                parsed = parse_json_response(response)
                st.session_state.document_insights = parsed.get("insights", [])[:12]
                st.success("Document insights extracted.")
            except Exception as exc:
                st.error(f"Could not process upload: {exc}")

    if st.session_state.document_insights:
        st.markdown("**Document insights**")
        for item in st.session_state.document_insights:
            st.write(f"- {item}")

with right:
    st.subheader("4. Outline")
    if st.button("Generate outline", type="primary"):
        try:
            inputs = current_inputs()
            if not inputs["topic"]:
                st.error("Please enter a topic first.")
            else:
                response = generate_text(
                    outline_system_prompt(),
                    outline_user_prompt(inputs),
                    max_tokens=2200,
                )
                parsed = parse_json_response(response)
                st.session_state.outline_title = parsed.get("title") or inputs["title"] or inputs["topic"]
                st.session_state.outline = parsed.get("outline", [])
                st.session_state.sections_content = {}
                st.success("Outline generated.")
        except Exception as exc:
            st.error(f"Could not generate outline: {exc}")

    if st.session_state.outline:
        st.text_input("Final article title", key="outline_title")
        st.caption("You can edit any heading or objective before generating sections.")

        updated_outline: list[dict[str, Any]] = []
        for idx, section in enumerate(st.session_state.outline):
            with st.expander(f"Section {idx + 1}: {section.get('heading', 'Untitled')}", expanded=(idx == 0)):
                heading = st.text_input(f"Heading {idx + 1}", value=section.get("heading", ""), key=f"heading_{idx}")
                objective = st.text_area(
                    f"Objective {idx + 1}",
                    value=section.get("objective", ""),
                    key=f"objective_{idx}",
                    height=80,
                )
                key_points_text = st.text_area(
                    f"Key points {idx + 1} (one per line)",
                    value="\n".join(section.get("keyPoints", [])),
                    key=f"keypoints_{idx}",
                    height=100,
                )
                suggested_words = st.number_input(
                    f"Suggested words {idx + 1}",
                    min_value=80,
                    max_value=800,
                    value=int(section.get("suggestedWords", 180)),
                    step=20,
                    key=f"words_{idx}",
                )
                updated_outline.append(
                    {
                        "id": section.get("id", f"s{idx+1}"),
                        "heading": heading,
                        "objective": objective,
                        "keyPoints": lines_to_list(key_points_text),
                        "suggestedWords": int(suggested_words),
                    }
                )
        st.session_state.outline = updated_outline

st.divider()
st.subheader("5. Generate article sections")

if st.session_state.outline:
    inputs = current_inputs()
    title = st.session_state.outline_title or inputs["title"] or inputs["topic"]

    for idx, section in enumerate(st.session_state.outline):
        key = section["id"]
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"### {section['heading']}")
            st.caption(section.get("objective", ""))
        with col2:
            if st.button("Generate", key=f"generate_{key}"):
                try:
                    section_text = generate_text(
                        section_system_prompt(),
                        section_user_prompt(inputs, section, title, st.session_state.outline),
                        max_tokens=min(3000, max(900, section["suggestedWords"] * 5)),
                    )
                    st.session_state.sections_content[key] = section_text
                    st.success(f"Generated {section['heading']}")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        current_content = st.session_state.sections_content.get(key, "")
        edited = st.text_area(
            f"Content for {section['heading']}",
            value=current_content,
            key=f"content_{key}",
            height=220,
        )
        st.session_state.sections_content[key] = edited

        revision_instruction = st.text_input(
            f"Revision instruction for {section['heading']}",
            key=f"rev_inst_{key}",
            placeholder="e.g. Make this more conversational and add one stronger example",
        )
        if st.button("Revise section", key=f"revise_{key}"):
            try:
                revised = generate_text(
                    revision_system_prompt(),
                    revision_user_prompt(section["heading"], st.session_state.sections_content.get(key, ""), revision_instruction),
                    max_tokens=2200,
                )
                st.session_state.sections_content[key] = revised
                st.rerun()
            except Exception as exc:
                st.error(f"Revision failed: {exc}")

        st.divider()

    if st.button("Generate all missing sections"):
        for section in st.session_state.outline:
            key = section["id"]
            if st.session_state.sections_content.get(key, "").strip():
                continue
            try:
                st.session_state.sections_content[key] = generate_text(
                    section_system_prompt(),
                    section_user_prompt(inputs, section, title, st.session_state.outline),
                    max_tokens=min(3000, max(900, section["suggestedWords"] * 5)),
                )
            except Exception as exc:
                st.error(f"Failed on {section['heading']}: {exc}")
                break
        st.rerun()

st.subheader("6. Export")
if st.session_state.outline:
    ordered_sections = [
        {
            "heading": section["heading"],
            "content": st.session_state.sections_content.get(section["id"], ""),
        }
        for section in st.session_state.outline
    ]

    combined_markdown = "\n\n".join(
        f"## {item['heading']}\n\n{item['content']}" for item in ordered_sections if item["content"].strip()
    )

    st.text_area("Combined article preview", value=combined_markdown, height=300)

    docx_bytes = export_blog_docx(title or "blog-article", ordered_sections)
    st.download_button(
        "Download DOCX",
        data=docx_bytes,
        file_name="blog-article.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
