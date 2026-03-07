from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from lib.anthropic_client import generate_text
from lib.export_docx import export_blog_docx
from lib.export_pdf import export_blog_pdf
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
        "logo_bytes": None,
        "logo_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def lines_to_list(value: str) -> list[str]:
    return [line.strip() for line in str(value).splitlines() if line.strip()]


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


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def to_bullet_lines(items: list[str]) -> str:
    cleaned = []
    for item in items:
        text = clean_text(item).lstrip("- ").strip()
        if text:
            cleaned.append(f"- {text}")
    return "\n".join(cleaned)


def from_bullet_lines(value: str) -> list[str]:
    cleaned = []
    for line in lines_to_list(value):
        text = clean_text(line).lstrip("- ").strip()
        if text:
            cleaned.append(text)
    return cleaned


def calc_text_area_height(text: str, min_height: int = 140, line_px: int = 28, extra_lines: int = 2) -> int:
    line_count = max(1, str(text or "").count("\n") + 1 + extra_lines)
    return max(min_height, line_px * line_count)


def normalise_outline() -> None:
    cleaned_outline: list[dict[str, Any]] = []

    for idx, section in enumerate(st.session_state.outline):
        section_id = str(section.get("id") or f"s{idx+1}")
        cleaned_outline.append(
            {
                "id": section_id,
                "heading": clean_text(section.get("heading", "")),
                "objective": clean_text(section.get("objective", "")),
                "keyPoints": [
                    clean_text(point).lstrip("- ").strip()
                    for point in section.get("keyPoints", [])
                    if clean_text(point).lstrip("- ").strip()
                ],
                "suggestedWords": int(section.get("suggestedWords", 180)),
            }
        )
        st.session_state.sections_content.setdefault(section_id, "")

    st.session_state.outline = cleaned_outline


def apply_pending_content_updates() -> None:
    pending_generation = st.session_state.pop("pending_generation_update", None)
    if pending_generation:
        section_id = str(pending_generation.get("section_id", "")).strip()
        generated_text = clean_text(pending_generation.get("content", ""))

        if section_id:
            content_key = f"content_{section_id}"
            st.session_state[content_key] = generated_text
            st.session_state.sections_content[section_id] = generated_text
            st.session_state["generation_success_message"] = "Section generated."

    pending_revision = st.session_state.pop("pending_revision_update", None)
    if pending_revision:
        section_id = str(pending_revision.get("section_id", "")).strip()
        revised_text = clean_text(pending_revision.get("content", ""))

        if section_id:
            content_key = f"content_{section_id}"
            st.session_state[content_key] = revised_text
            st.session_state.sections_content[section_id] = revised_text
            st.session_state["revision_success_message"] = "Section revised."


init_state()
apply_pending_content_updates()
normalise_outline()

# Top logo area
top_left, top_right = st.columns([1, 5])

with top_left:
    logo_upload = st.file_uploader(
        "Upload logo",
        type=["png", "jpg", "jpeg", "webp"],
        key="logo_uploader",
        help="Optional logo shown at the top of the app",
    )
    if logo_upload is not None:
        st.session_state.logo_bytes = logo_upload.getvalue()
        st.session_state.logo_name = logo_upload.name

with top_right:
    if st.session_state.logo_bytes:
        st.image(st.session_state.logo_bytes, width=180)

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
    st.text_area("SEO keywords (one per line)", key="keywords_text", height=140)

    st.subheader("2. Facts, quotes, and notes")
    st.text_area("Facts to include (one per line)", key="facts_text", height=160)
    st.text_area("Quotes to include verbatim (one per line)", key="quotes_text", height=160)
    st.text_area("Additional research notes", key="research_notes", height=160)

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
                    insights_user_prompt(
                        raw_text,
                        st.session_state.topic or st.session_state.title or "blog topic",
                    ),
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
                st.session_state.outline_title = clean_text(
                    parsed.get("title") or inputs["title"] or inputs["topic"]
                )
                st.session_state.outline = parsed.get("outline", [])
                st.session_state.sections_content = {}

                keys_to_delete = [
                    key
                    for key in list(st.session_state.keys())
                    if key.startswith("content_") or key.startswith("rev_inst_")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                normalise_outline()
                st.success("Outline generated.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not generate outline: {exc}")

    if st.session_state.outline:
        st.text_input("Final article title", key="outline_title")
        st.caption("You can edit any heading or objective before generating sections.")

        updated_outline: list[dict[str, Any]] = []
        for idx, section in enumerate(st.session_state.outline):
            with st.expander(f"Section {idx + 1}: {section.get('heading', 'Untitled')}", expanded=(idx == 0)):
                heading = st.text_input(
                    f"Heading {idx + 1}",
                    value=clean_text(section.get("heading", "")),
                    key=f"heading_{idx}",
                )

                objective_value = clean_text(section.get("objective", ""))
                objective_height = calc_text_area_height(
                    objective_value,
                    min_height=140,
                    line_px=28,
                    extra_lines=3,
                )
                objective = st.text_area(
                    f"Objective {idx + 1}",
                    value=objective_value,
                    key=f"objective_{idx}",
                    height=objective_height,
                )

                key_points_value = to_bullet_lines(section.get("keyPoints", []))
                key_points_height = calc_text_area_height(
                    key_points_value,
                    min_height=170,
                    line_px=30,
                    extra_lines=3,
                )
                key_points_text = st.text_area(
                    f"Key points {idx + 1}",
                    value=key_points_value,
                    key=f"keypoints_{idx}",
                    height=key_points_height,
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
                        "heading": clean_text(heading),
                        "objective": clean_text(objective),
                        "keyPoints": from_bullet_lines(key_points_text),
                        "suggestedWords": int(suggested_words),
                    }
                )
        st.session_state.outline = updated_outline
        normalise_outline()

if st.session_state.get("generation_success_message"):
    st.success(st.session_state.pop("generation_success_message"))

if st.session_state.get("revision_success_message"):
    st.success(st.session_state.pop("revision_success_message"))

st.divider()
st.subheader("5. Generate article sections")

if st.session_state.outline:
    inputs = current_inputs()
    title = clean_text(st.session_state.outline_title or inputs["title"] or inputs["topic"])

    for idx, section in enumerate(st.session_state.outline):
        key = section["id"]
        content_key = f"content_{key}"
        revision_key = f"rev_inst_{key}"

        if content_key not in st.session_state:
            st.session_state[content_key] = st.session_state.sections_content.get(key, "")

        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"### {section['heading']}")
            st.caption(section.get("objective", ""))

            if section.get("keyPoints"):
                st.markdown(
                    "\n".join(f"- {clean_text(point)}" for point in section["keyPoints"])
                )

        with col2:
            if st.button("Generate", key=f"generate_{key}"):
                try:
                    section_text = clean_text(
                        generate_text(
                            section_system_prompt(),
                            section_user_prompt(inputs, section, title, st.session_state.outline),
                            max_tokens=min(3000, max(900, int(section["suggestedWords"]) * 5)),
                        )
                    )
                    st.session_state["pending_generation_update"] = {
                        "section_id": key,
                        "content": section_text,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        current_content = st.session_state.get(content_key, "")
        content_height = calc_text_area_height(
            current_content,
            min_height=420,
            line_px=28,
            extra_lines=10,
        )

        edited = st.text_area(
            f"Content for {section['heading']}",
            key=content_key,
            height=content_height,
        )
        st.session_state.sections_content[key] = clean_text(edited)

        revision_instruction = st.text_input(
            f"Revision instruction for {section['heading']}",
            key=revision_key,
            placeholder="e.g. Make this more conversational and add one stronger example",
        )

        if st.button("Revise section", key=f"revise_{key}"):
            try:
                source_text = clean_text(st.session_state.sections_content.get(key, ""))

                revised = clean_text(
                    generate_text(
                        revision_system_prompt(),
                        revision_user_prompt(
                            section["heading"],
                            source_text,
                            revision_instruction,
                        ),
                        max_tokens=2200,
                    )
                )

                st.session_state["pending_revision_update"] = {
                    "section_id": key,
                    "content": revised,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Revision failed: {exc}")

        st.divider()

    if st.button("Generate all missing sections"):
        try:
            generated_any = False
            for section in st.session_state.outline:
                key = section["id"]
                existing = clean_text(st.session_state.sections_content.get(key, ""))
                if existing:
                    continue

                section_text = clean_text(
                    generate_text(
                        section_system_prompt(),
                        section_user_prompt(inputs, section, title, st.session_state.outline),
                        max_tokens=min(3000, max(900, int(section["suggestedWords"]) * 5)),
                    )
                )

                st.session_state.sections_content[key] = section_text
                st.session_state[f"content_{key}"] = section_text
                generated_any = True

            if generated_any:
                st.success("Generated all missing sections.")
            else:
                st.info("No missing sections to generate.")
            st.rerun()

        except Exception as exc:
            st.error(f"Failed while generating missing sections: {exc}")

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
        f"## {item['heading']}\n\n{item['content']}"
        for item in ordered_sections
        if item["content"].strip()
    )

    preview_height = calc_text_area_height(
        combined_markdown,
        min_height=500,
        line_px=26,
        extra_lines=8,
    )

    st.text_area("Combined article preview", value=combined_markdown, height=preview_height)

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        docx_bytes = export_blog_docx(title or "blog-article", ordered_sections)
        st.download_button(
            "Download DOCX",
            data=docx_bytes,
            file_name="blog-article.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    with export_col2:
        pdf_bytes = export_blog_pdf(title or "blog-article", ordered_sections)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="blog-article.pdf",
            mime="application/pdf",
        )
