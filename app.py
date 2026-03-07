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
    section_metadata_system_prompt,
    section_metadata_user_prompt,
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
        "new_section_topic": "",
        "new_section_position": 1,
        "next_section_id": 1,
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


def section_defaults(section: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "id": str(section.get("id", f"s{idx + 1}")),
        "heading": clean_text(section.get("heading", "")),
        "objective": clean_text(section.get("objective", "")),
        "keyPoints": [clean_text(x) for x in section.get("keyPoints", []) if clean_text(x)],
        "suggestedWords": int(section.get("suggestedWords", 180)),
    }


def ensure_outline_state() -> None:
    refreshed_outline: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    max_numeric_id = 0

    for idx, section in enumerate(st.session_state.outline):
        item = section_defaults(section, idx)
        section_id = item["id"]

        if not section_id or section_id in used_ids:
            max_numeric_id += 1
            section_id = f"s{max_numeric_id}"
            item["id"] = section_id
        else:
            if section_id.startswith("s") and section_id[1:].isdigit():
                max_numeric_id = max(max_numeric_id, int(section_id[1:]))

        used_ids.add(section_id)
        st.session_state.sections_content.setdefault(
            section_id,
            st.session_state.sections_content.get(section_id, ""),
        )
        refreshed_outline.append(item)

    st.session_state.outline = refreshed_outline
    st.session_state.next_section_id = max(
        int(st.session_state.get("next_section_id", 1)),
        max_numeric_id + 1,
    )


def new_section_id() -> str:
    next_id = int(st.session_state.get("next_section_id", 1))
    section_id = f"s{next_id}"
    st.session_state.next_section_id = next_id + 1
    return section_id


def generate_section_metadata(
    heading: str,
    outline: list[dict[str, Any]],
    section_position: int,
) -> dict[str, Any]:
    response = generate_text(
        section_metadata_system_prompt(),
        section_metadata_user_prompt(current_inputs(), heading, outline, section_position),
        max_tokens=1200,
    )
    parsed = parse_json_response(response)
    return {
        "heading": clean_text(parsed.get("heading") or heading),
        "objective": clean_text(parsed.get("objective", "")),
        "keyPoints": [clean_text(x) for x in parsed.get("keyPoints", []) if clean_text(x)],
        "suggestedWords": max(80, min(800, int(parsed.get("suggestedWords", 180)))),
    }


def apply_pending_section_refresh() -> None:
    pending_section_refresh = st.session_state.pop("pending_section_refresh", None)
    if not pending_section_refresh:
        return

    target_id = str(pending_section_refresh.get("id", ""))
    metadata = pending_section_refresh.get("metadata", {})

    for idx, section in enumerate(st.session_state.get("outline", [])):
        if str(section.get("id")) == target_id:
            updated_section = {
                **section_defaults(section, idx),
                "heading": clean_text(metadata.get("heading") or section.get("heading", "")),
                "objective": clean_text(metadata.get("objective") or section.get("objective", "")),
                "keyPoints": [
                    clean_text(x)
                    for x in metadata.get("keyPoints", section.get("keyPoints", []))
                    if clean_text(x)
                ],
                "suggestedWords": int(
                    metadata.get("suggestedWords", section.get("suggestedWords", 180))
                ),
            }

            st.session_state.outline[idx] = updated_section

            # Safe here because widgets have not been created yet in this run
            st.session_state[f"heading_{target_id}"] = updated_section["heading"]
            st.session_state[f"objective_{target_id}"] = updated_section["objective"]
            st.session_state[f"keypoints_{target_id}"] = "\n".join(updated_section["keyPoints"])
            st.session_state[f"words_{target_id}"] = updated_section["suggestedWords"]
            st.session_state["section_refresh_success"] = f"Updated section: {updated_section['heading']}"
            break


init_state()
ensure_outline_state()

if st.session_state.pop("clear_new_section_inputs", False):
    st.session_state["new_section_topic"] = ""
    st.session_state["new_section_position"] = 1

apply_pending_section_refresh()
ensure_outline_state()

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
                st.session_state.outline = [
                    section_defaults(section, idx)
                    for idx, section in enumerate(parsed.get("outline", []))
                ]
                st.session_state.sections_content = {}

                keys_to_delete = [
                    key
                    for key in list(st.session_state.keys())
                    if key.startswith("content_")
                    or key.startswith("rev_inst_")
                    or key.startswith("heading_s")
                    or key.startswith("objective_s")
                    or key.startswith("keypoints_s")
                    or key.startswith("words_s")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                ensure_outline_state()
                st.success("Outline generated.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not generate outline: {exc}")

    ensure_outline_state()

    if st.session_state.outline:
        st.text_input("Final article title", key="outline_title")
        st.caption("Edit headings, then use Update details to refresh the objective and key points for that section.")

        if st.session_state.get("section_refresh_success"):
            st.success(st.session_state.pop("section_refresh_success"))

        add_section_expander = st.expander("Add a new section", expanded=False)
        with add_section_expander:
            st.text_input(
                "New section topic or headline",
                key="new_section_topic",
                placeholder="e.g. What candidates now expect from AI-assisted hiring",
            )

            max_position = len(st.session_state.outline) + 1
            current_position = min(
                max(1, int(st.session_state.get("new_section_position", 1))),
                max_position,
            )

            if st.session_state.get("new_section_position") != current_position:
                st.session_state["new_section_position"] = current_position

            st.number_input(
                "Insert position",
                min_value=1,
                max_value=max_position,
                step=1,
                key="new_section_position",
                help="1 inserts at the top. The last position inserts at the end.",
            )

            if st.button("Generate and insert section", key="insert_new_section"):
                try:
                    new_heading = clean_text(st.session_state.get("new_section_topic", ""))
                    if not new_heading:
                        st.error("Please enter a topic or headline for the new section.")
                    else:
                        insert_at = int(st.session_state.get("new_section_position", 1)) - 1
                        preview_outline = [
                            section_defaults(section, idx)
                            for idx, section in enumerate(st.session_state.outline)
                        ]
                        metadata = generate_section_metadata(new_heading, preview_outline, insert_at + 1)
                        new_section = {
                            "id": new_section_id(),
                            "heading": metadata["heading"],
                            "objective": metadata["objective"],
                            "keyPoints": metadata["keyPoints"],
                            "suggestedWords": metadata["suggestedWords"],
                        }
                        st.session_state.outline.insert(insert_at, new_section)
                        ensure_outline_state()
                        st.session_state["clear_new_section_inputs"] = True
                        st.success("New section added.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not add the new section: {exc}")

        updated_outline: list[dict[str, Any]] = []

        for idx, section in enumerate(st.session_state.outline):
            section = section_defaults(section, idx)
            section_id = section["id"]

            with st.expander(f"Section {idx + 1}: {section.get('heading', 'Untitled')}", expanded=(idx == 0)):
                heading = st.text_input(
                    f"Heading {idx + 1}",
                    value=section["heading"],
                    key=f"heading_{section_id}",
                )

                button_col1, button_col2 = st.columns([1, 1])

                with button_col1:
                    if st.button("Update details", key=f"refresh_meta_{section_id}"):
                        try:
                            refreshed_heading = clean_text(
                                st.session_state.get(f"heading_{section_id}", heading)
                            )

                            if not refreshed_heading:
                                st.error("Please enter a heading first.")
                            else:
                                preview_outline = []
                                for preview_idx, preview_section in enumerate(st.session_state.outline):
                                    base = section_defaults(preview_section, preview_idx)
                                    base_id = base["id"]
                                    draft_heading = clean_text(
                                        st.session_state.get(f"heading_{base_id}", base["heading"])
                                    )
                                    if preview_idx == idx:
                                        base["heading"] = refreshed_heading
                                    else:
                                        base["heading"] = draft_heading or base["heading"]
                                    preview_outline.append(base)

                                metadata = generate_section_metadata(
                                    refreshed_heading,
                                    preview_outline,
                                    idx + 1,
                                )

                                st.session_state["pending_section_refresh"] = {
                                    "id": section_id,
                                    "metadata": metadata,
                                }
                                st.rerun()

                        except Exception as exc:
                            st.error(f"Could not update this section: {exc}")

                with button_col2:
                    if st.button("Delete section", key=f"delete_section_{section_id}"):
                        st.session_state.outline.pop(idx)
                        st.session_state.sections_content.pop(section_id, None)

                        keys_to_remove = [
                            f"content_{section_id}",
                            f"rev_inst_{section_id}",
                            f"heading_{section_id}",
                            f"objective_{section_id}",
                            f"keypoints_{section_id}",
                            f"words_{section_id}",
                        ]
                        for key in keys_to_remove:
                            st.session_state.pop(key, None)

                        ensure_outline_state()
                        st.success("Section removed.")
                        st.rerun()

                objective = st.text_area(
                    f"Objective {idx + 1}",
                    value=section["objective"],
                    key=f"objective_{section_id}",
                    height=80,
                )

                key_points_text = st.text_area(
                    f"Key points {idx + 1} (one per line)",
                    value="\n".join(section["keyPoints"]),
                    key=f"keypoints_{section_id}",
                    height=100,
                )

                suggested_words = st.number_input(
                    f"Suggested words {idx + 1}",
                    min_value=80,
                    max_value=800,
                    value=int(section["suggestedWords"]),
                    step=20,
                    key=f"words_{section_id}",
                )

                updated_outline.append(
                    {
                        "id": section_id,
                        "heading": clean_text(st.session_state.get(f"heading_{section_id}", heading)),
                        "objective": clean_text(objective),
                        "keyPoints": [clean_text(x) for x in lines_to_list(key_points_text)],
                        "suggestedWords": int(suggested_words),
                    }
                )

        st.session_state.outline = updated_outline
        ensure_outline_state()

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

        with col2:
            if st.button("Generate", key=f"generate_{key}"):
                try:
                    section_text = clean_text(
                        generate_text(
                            section_system_prompt(),
                            section_user_prompt(inputs, section, title, st.session_state.outline),
                            max_tokens=min(3000, max(900, section["suggestedWords"] * 5)),
                        )
                    )
                    st.session_state[content_key] = section_text
                    st.session_state.sections_content[key] = section_text
                    st.success(f"Generated {section['heading']}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        edited = st.text_area(
            f"Content for {section['heading']}",
            key=content_key,
            height=220,
        )
        st.session_state.sections_content[key] = clean_text(edited)

        revision_instruction = st.text_input(
            f"Revision instruction for {section['heading']}",
            key=revision_key,
            placeholder="e.g. Make this more conversational and add one stronger example",
        )

        if st.button("Revise section", key=f"revise_{key}"):
            try:
                revised = clean_text(
                    generate_text(
                        revision_system_prompt(),
                        revision_user_prompt(
                            section["heading"],
                            st.session_state.sections_content.get(key, ""),
                            revision_instruction,
                        ),
                        max_tokens=2200,
                    )
                )
                st.session_state[content_key] = revised
                st.session_state.sections_content[key] = revised
                st.rerun()
            except Exception as exc:
                st.error(f"Revision failed: {exc}")

        st.divider()

    if st.button("Generate all missing sections"):
        for section in st.session_state.outline:
            key = section["id"]
            content_key = f"content_{key}"

            if st.session_state.sections_content.get(key, "").strip():
                continue

            try:
                section_text = clean_text(
                    generate_text(
                        section_system_prompt(),
                        section_user_prompt(inputs, section, title, st.session_state.outline),
                        max_tokens=min(3000, max(900, section["suggestedWords"] * 5)),
                    )
                )
                st.session_state[content_key] = section_text
                st.session_state.sections_content[key] = section_text
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
        f"## {item['heading']}\n\n{item['content']}"
        for item in ordered_sections
        if item["content"].strip()
    )

    st.text_area("Combined article preview", value=combined_markdown, height=300)

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
