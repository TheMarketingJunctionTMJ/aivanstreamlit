from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from lib.anthropic_client import generate_text
from lib.export_docx import export_blog_docx
from lib.export_pdf import export_blog_pdf
from lib.file_processing import extract_text_from_upload
from lib.prompts import (
    LANGUAGE_OPTIONS,
    TONE_OPTIONS,
    ai_friendly_blog_system_prompt,
    ai_friendly_blog_user_prompt,
    ai_friendly_outline_system_prompt,
    ai_friendly_outline_user_prompt,
    evaluate_system_prompt,
    evaluate_user_prompt,
    insights_system_prompt,
    insights_user_prompt,
    outline_system_prompt,
    outline_user_prompt,
    revision_system_prompt,
    revision_user_prompt,
    section_system_prompt,
    section_user_prompt,
    verify_system_prompt,
    verify_user_prompt,
)

load_dotenv()

st.set_page_config(
    page_title="Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "blog_mode": "Writer Version",
        "title": "",
        "topic": "",
        "topic_input": "",
        "pending_ai_title": "",
        "ai_title_checkbox": False,
        "ai_title_checkbox_prev": False,
        "audience": "",
        "keywords_text": "",
        "facts_text": "",
        "quotes_text": "",
        "research_notes": "",
        "tone": "Thought leadership",
        "language": "UK English",
        "target_words": 1200,
        "add_hiring_section": False,
        "document_text": "",
        "document_insights": [],
        "quoted_lines": [],
        "evaluated_evidence": {},
        "verified_evidence": {},
        "outline_title": "",
        "outline": [],
        "sections_content": {},
        "logo_bytes": None,
        "logo_name": "",
        "sections_workspace_ready": False,
        "new_section_prompt": "",
        "seo_keyword_suggestions": [],
        "selected_seo_keywords": [],
        "show_seo_keyword_dialog": False,
        "pending_outline_generation": False,
        "ai_friendly_draft": "",
        "ai_friendly_draft_editor": "",
        "pending_ai_friendly_generation": False,
        "ai_outline_title": "",
        "ai_outline": [],
        "ai_new_section_prompt": "",
        "pending_ai_outline_generation": False,
        "full_blog_revision_prompt": "",
        "ai_full_blog_revision_prompt": "",
        "pending_full_blog_revision": None,
        "pending_ai_full_blog_revision": None,
        "processing_message": "",
        "writer_full_draft": "",
        "writer_full_draft_editor": "",
        "pending_writer_full_generation": False,
        "writer_export_title": "",
        "writer_export_title_source": "",
        "ai_export_title": "",
        "ai_export_title_source": "",
        "evidence_cache_key": "",
        "performance_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value




def record_timing(step: str, started_at: float) -> None:
    elapsed = round(max(0.0, time.perf_counter() - started_at), 2)
    history = list(st.session_state.get("performance_log", []))
    history.append({"step": clean_text(step), "seconds": elapsed})
    st.session_state.performance_log = history[-20:]


def evidence_cache_key(inputs: dict[str, Any]) -> str:
    payload = {
        "topic": clean_text(inputs.get("topic") or inputs.get("title")),
        "language": clean_text(inputs.get("language")),
        "audience": clean_text(inputs.get("audience")),
        "evidence": build_evidence_bundle(inputs),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parallel_section_workers(section_count: int) -> int:
    return max(1, min(3, int(section_count or 1)))


def generate_single_section(inputs: dict[str, Any], title: str, section: dict[str, Any], outline_snapshot: list[dict[str, Any]]) -> tuple[str, str]:
    tighter_section = dict(section)
    tighter_section["objective"] = (
        f"{clean_text(section.get('objective', ''))} "
        f"Write approximately {int(section['suggestedWords'])} words. "
        f"Do not exceed the target by more than 120 words."
    ).strip()

    section_text = sanitise_section_content(
        generate_text(
            section_system_prompt(inputs["language"]),
            section_user_prompt(inputs, tighter_section, title, outline_snapshot),
            max_tokens=section_max_tokens(int(section["suggestedWords"])),
        ),
        section.get("heading", ""),
    )
    return str(section["id"]), section_text

def lines_to_list(value: str) -> list[str]:
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def get_manual_title() -> str:
    return clean_text(
        st.session_state.get("topic_input", "")
        or st.session_state.get("title", "")
        or st.session_state.get("topic", "")
    )


def current_inputs() -> dict[str, Any]:
    manual_title = get_manual_title()
    generated_title = clean_text(
        st.session_state.ai_outline_title
        or st.session_state.outline_title
    )
    return {
        "title": manual_title or generated_title,
        "topic": manual_title or generated_title,
        "audience": st.session_state.audience.strip(),
        "keywords": lines_to_list(st.session_state.keywords_text),
        "facts": lines_to_list(st.session_state.facts_text),
        "quotes": lines_to_list(st.session_state.quotes_text),
        "research_notes": st.session_state.research_notes.strip(),
        "tone": st.session_state.tone,
        "language": st.session_state.language,
        "target_words": int(st.session_state.target_words),
        "document_insights": st.session_state.document_insights,
        "verified_evidence": st.session_state.verified_evidence,
        "blog_mode": st.session_state.blog_mode,
        "add_hiring_section": bool(st.session_state.add_hiring_section),
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


def normalise_heading_for_compare(text: str) -> str:
    text = clean_text(text).lower()
    chars: list[str] = []
    previous_was_space = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_was_space = False
        else:
            if not previous_was_space:
                chars.append(" ")
                previous_was_space = True
    return " ".join("".join(chars).split())


def strip_leading_heading(content: str, heading: str) -> str:
    content = str(content or "")
    heading = clean_text(heading)
    if not content.strip() or not heading:
        return clean_text(content)

    lines = content.splitlines()
    first_non_empty_index = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_empty_index = idx
            break

    if first_non_empty_index is None:
        return clean_text(content)

    first_line = lines[first_non_empty_index].strip()
    candidate = first_line.lstrip("#").strip()
    candidate = candidate.strip("*_").strip()
    candidate = candidate.rstrip(":")

    if normalise_heading_for_compare(candidate) == normalise_heading_for_compare(heading):
        remaining_lines = lines[first_non_empty_index + 1:]
        while remaining_lines and not remaining_lines[0].strip():
            remaining_lines.pop(0)
        return clean_text("\n".join(remaining_lines))

    return clean_text(content)


def sanitise_section_content(content: str, heading: str) -> str:
    return strip_leading_heading(clean_text(content), heading)


def section_max_tokens(suggested_words: int) -> int:
    target = max(80, int(suggested_words))
    return min(1400, max(250, int(target * 1.35)))


def count_words_in_sections(ordered_sections: list[dict[str, str]]) -> int:
    total_text = " ".join(
        clean_text(section.get("content", ""))
        for section in ordered_sections
        if clean_text(section.get("content", ""))
    )
    return len(total_text.split())


def build_export_sections_with_appendix(
    ordered_sections: list[dict[str, str]],
    keywords: list[str],
) -> list[dict[str, str]]:
    total_words = count_words_in_sections(ordered_sections)
    cleaned_keywords = [clean_text(keyword) for keyword in keywords if clean_text(keyword)]

    appendix_lines = [f"Total number of words: {total_words}"]

    if cleaned_keywords:
        appendix_lines.append("SEO keywords used: " + ", ".join(cleaned_keywords))
    else:
        appendix_lines.append("SEO keywords used: None provided")

    export_sections = list(ordered_sections)
    export_sections.append(
        {
            "heading": "Article metadata",
            "content": "\n\n".join(appendix_lines),
        }
    )
    return export_sections


def markdown_to_export_sections(markdown_text: str, fallback_title: str) -> list[dict[str, str]]:
    text = str(markdown_text or "").strip()
    if not text:
        return [{"heading": fallback_title or "Article", "content": ""}]

    sections: list[dict[str, str]] = []
    current_heading = fallback_title or "Article"
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_heading, current_lines, sections
        content = clean_text("\\n".join(current_lines).strip())
        heading = clean_text(current_heading) or "Section"
        if content:
            sections.append({"heading": heading, "content": content})
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            current_lines.append("")
            continue

        md_heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if md_heading_match:
            heading_text = clean_text(md_heading_match.group(2))
            if heading_text:
                if current_lines:
                    flush_section()
                else:
                    if not sections:
                        current_heading = heading_text
                        continue
                current_heading = heading_text
                continue

        bold_heading_match = re.match(r"^\*\*(.+?)\*\*$", line)
        if bold_heading_match and len(line) < 120:
            heading_text = clean_text(bold_heading_match.group(1))
            if heading_text:
                if current_lines:
                    flush_section()
                current_heading = heading_text
                continue

        current_lines.append(raw_line.rstrip())

    if current_lines or not sections:
        flush_section()

    if not sections:
        sections.append(
            {
                "heading": fallback_title or "Article",
                "content": clean_text(text),
            }
        )

    return sections

def sections_to_markdown(sections: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"## {clean_text(section.get('heading', 'Section'))}\n\n{clean_text(section.get('content', ''))}"
        for section in sections
        if clean_text(section.get("content", ""))
    ).strip()


def apply_revised_markdown_to_writer_sections(
    revised_markdown: str,
    outline_sections: list[dict[str, Any]],
) -> None:
    parsed_sections = markdown_to_export_sections(
        revised_markdown,
        st.session_state.outline_title or st.session_state.topic or "Article",
    )

    parsed_map = {
        normalise_heading_for_compare(section["heading"]): section["content"]
        for section in parsed_sections
        if clean_text(section.get("heading")) and clean_text(section.get("content"))
    }

    used_keys: set[str] = set()

    for section in outline_sections:
        section_id = str(section["id"])
        heading = clean_text(section.get("heading", ""))
        lookup_key = normalise_heading_for_compare(heading)
        matched_content = parsed_map.get(lookup_key)

        if matched_content:
            cleaned_content = sanitise_section_content(matched_content, heading)
            st.session_state.sections_content[section_id] = cleaned_content
            used_keys.add(lookup_key)

    leftover_sections = [
        section
        for section in parsed_sections
        if normalise_heading_for_compare(section["heading"]) not in used_keys
    ]

    if leftover_sections and outline_sections:
        last_section = outline_sections[-1]
        last_id = str(last_section["id"])
        existing_content = clean_text(st.session_state.sections_content.get(last_id, ""))
        extra_text = "\n\n".join(
            f"## {section['heading']}\n\n{section['content']}" for section in leftover_sections
        ).strip()

        merged_content = clean_text(
            f"{existing_content}\n\n{extra_text}" if existing_content and extra_text else existing_content or extra_text
        )
        merged_content = sanitise_section_content(
            merged_content,
            clean_text(last_section.get("heading", "")),
        )
        st.session_state.sections_content[last_id] = merged_content

    for section in outline_sections:
        section_id = str(section["id"])
        st.session_state.pop(f"content_{section_id}", None)
        st.session_state.pop(f"rev_inst_{section_id}", None)


def queue_full_blog_revision_for_writer(revised_markdown: str) -> None:
    cleaned = clean_text(revised_markdown)
    st.session_state["writer_full_draft"] = cleaned
    st.session_state["writer_full_draft_editor"] = cleaned
    st.session_state["writer_export_title"] = ""
    st.session_state["writer_export_title_source"] = ""



def queue_full_blog_revision_for_ai(revised_markdown: str) -> None:
    st.session_state["pending_ai_full_blog_revision"] = {
        "revised_markdown": clean_text(revised_markdown),
    }
    st.session_state["ai_export_title"] = ""
    st.session_state["ai_export_title_source"] = ""


def make_section_id() -> str:
    return f"s{uuid.uuid4().hex[:8]}"


def build_manual_section(prompt: str) -> dict[str, Any]:
    prompt = clean_text(prompt)
    words = [word for word in prompt.replace(",", " ").split() if word]
    heading = " ".join(words[:12]).strip().rstrip(".") or "New section"
    heading = heading[:1].upper() + heading[1:] if heading else "New section"
    return {
        "id": make_section_id(),
        "heading": heading,
        "objective": prompt,
        "keyPoints": [prompt],
        "suggestedWords": 180,
    }


def build_evidence_bundle(inputs: dict[str, Any]) -> str:
    parts: list[str] = []

    if inputs["facts"]:
        parts.append("User facts:\n" + "\n".join(f"- {clean_text(x)}" for x in inputs["facts"] if clean_text(x)))

    if inputs["quotes"]:
        parts.append("User quotes:\n" + "\n".join(f"- {clean_text(x)}" for x in inputs["quotes"] if clean_text(x)))

    if inputs["research_notes"]:
        parts.append("Research notes:\n" + clean_text(inputs["research_notes"]))

    if st.session_state.document_text.strip():
        parts.append("Uploaded document text:\n" + st.session_state.document_text[:12000])

    return "\n\n".join(parts).strip()


def suggest_seo_keywords(inputs: dict[str, Any]) -> list[str]:
    topic = clean_text(inputs["topic"] or inputs["title"] or "blog topic")
    audience = clean_text(inputs["audience"])

    system_prompt = (
        f"You are an SEO strategist writing in {inputs['language']}. "
        "Return valid JSON only."
    )

    user_prompt = f"""
Suggest 12 SEO keywords or keyword phrases for this blog article.

Topic: {topic}
Audience: {audience}
Tone: {inputs['tone']}

Return JSON in this shape:
{{
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ]
}}

Rules:
- Return exactly 12 suggestions.
- Mix short-tail and long-tail phrases.
- Keep them relevant to the topic and audience.
- No numbering.
- No markdown.
- JSON only.
"""
    response = generate_text(system_prompt, user_prompt, max_tokens=700)
    parsed = parse_json_response(response)
    return [clean_text(item) for item in parsed.get("keywords", []) if clean_text(item)][:12]



def generate_ai_title(topic: str, audience: str, language: str, content: str = "") -> str:
    cleaned_topic = clean_text(topic) or "blog topic"
    cleaned_audience = clean_text(audience) or "general audience"
    cleaned_content = clean_text(content)

    system_prompt = (
        f"You are an expert blog editor writing in {language}. "
        "Create one strong, publication-ready blog title and return only the title."
    )

    if cleaned_content:
        user_prompt = f"""
Create one compelling blog title after reading the content first.

Working topic: {cleaned_topic}
Audience: {cleaned_audience}

Content to read before writing the title:
{cleaned_content[:12000]}

Rules:
- Read the content first and base the title on the actual subject matter.
- Return exactly one title only.
- Make it clear, natural, and publication-ready.
- Use title case where appropriate.
- Do not add quotation marks.
- Do not add numbering, bullets, or commentary.
- Keep it under 75 characters where possible.
"""
    else:
        user_prompt = f"""
Create one compelling blog title for this topic.

Topic or working title: {cleaned_topic}
Audience: {cleaned_audience}

Rules:
- Return exactly one title only.
- Make it clear, natural, and publication-ready.
- Use title case where appropriate.
- Do not add quotation marks.
- Do not add numbering, bullets, or commentary.
- Keep it under 75 characters where possible.
"""

    response = generate_text(system_prompt, user_prompt, max_tokens=120)
    return clean_text(response.splitlines()[0])


def get_ai_title_source_text() -> str:
    inputs = current_inputs()
    parts: list[str] = []

    evidence_text = build_evidence_bundle(inputs)
    if evidence_text:
        parts.append(evidence_text)

    if st.session_state.writer_full_draft.strip():
        parts.append("Current writer draft:\n" + st.session_state.writer_full_draft.strip()[:12000])

    if st.session_state.ai_friendly_draft.strip():
        parts.append("Current AI-friendly draft:\n" + st.session_state.ai_friendly_draft.strip()[:12000])

    return "\n\n".join(part for part in parts if clean_text(part)).strip()


def resolve_export_title(mode: str, fallback_title: str, content: str) -> str:
    fallback_title = clean_text(fallback_title) or ("AI-friendly blog" if mode == "AI Friendly" else "Blog article")

    cached_title_key = "ai_export_title" if mode == "AI Friendly" else "writer_export_title"
    cached_source_key = "ai_export_title_source" if mode == "AI Friendly" else "writer_export_title_source"
    content_signature = clean_text(content)[:12000]

    if not st.session_state.ai_title_checkbox:
        st.session_state[cached_title_key] = ""
        st.session_state[cached_source_key] = ""
        return fallback_title

    cached_title = clean_text(st.session_state.get(cached_title_key, ""))
    cached_source = st.session_state.get(cached_source_key, "")
    if cached_title and cached_source == content_signature:
        return cached_title

    ai_title = generate_ai_title(
        topic=fallback_title,
        audience=clean_text(st.session_state.audience),
        language=clean_text(st.session_state.language) or "UK English",
        content=content,
    )
    final_title = clean_text(ai_title) or fallback_title
    st.session_state[cached_title_key] = final_title
    st.session_state[cached_source_key] = content_signature
    return final_title
def run_evan_light(inputs: dict[str, Any]) -> None:
    evidence_text = build_evidence_bundle(inputs)

    if not evidence_text:
        st.session_state.evaluated_evidence = {}
        st.session_state.verified_evidence = {}
        st.session_state.evidence_cache_key = ""
        return

    cache_key = evidence_cache_key(inputs)
    if (
        cache_key
        and cache_key == st.session_state.get("evidence_cache_key", "")
        and st.session_state.get("evaluated_evidence")
        and st.session_state.get("verified_evidence")
    ):
        return

    started_at = time.perf_counter()
    evaluated_response = generate_text(
        evaluate_system_prompt(inputs["language"]),
        evaluate_user_prompt(
            evidence_text,
            inputs["topic"] or inputs["title"] or "blog topic",
            inputs["language"],
        ),
        max_tokens=1800,
    )
    evaluated = parse_json_response(evaluated_response)
    record_timing("Evidence evaluation", started_at)

    started_at = time.perf_counter()
    verified_response = generate_text(
        verify_system_prompt(inputs["language"]),
        verify_user_prompt(
            json.dumps(evaluated, ensure_ascii=False),
            evidence_text,
            inputs["topic"] or inputs["title"] or "blog topic",
            inputs["language"],
        ),
        max_tokens=1800,
    )
    verified = parse_json_response(verified_response)
    record_timing("Evidence verification", started_at)

    st.session_state.evaluated_evidence = evaluated
    st.session_state.verified_evidence = verified
    st.session_state.evidence_cache_key = cache_key


def run_outline_generation() -> None:
    st.session_state.pending_outline_generation = False

    inputs = current_inputs()
    if not inputs["topic"]:
        st.error("Please enter a topic first.")
        return

    run_evan_light(inputs)
    inputs = current_inputs()

    started_at = time.perf_counter()
    response = generate_text(
        outline_system_prompt(inputs["language"]),
        outline_user_prompt(inputs),
        max_tokens=2200,
    )
    record_timing("Writer outline generation", started_at)
    parsed = parse_json_response(response)
    generated_title = clean_text(parsed.get("title") or inputs["title"] or inputs["topic"])
    manual_title = get_manual_title()
    st.session_state.outline_title = generated_title if st.session_state.ai_title_checkbox else (manual_title or generated_title)
    st.session_state.outline = parsed.get("outline", [])
    st.session_state.sections_content = {}
    st.session_state.sections_workspace_ready = False
    st.session_state.new_section_prompt = ""
    st.session_state.ai_friendly_draft = ""
    st.session_state.ai_friendly_draft_editor = ""

    keys_to_delete = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("content_") or key.startswith("rev_inst_")
    ]
    for key in keys_to_delete:
        del st.session_state[key]

    normalise_outline()
    st.session_state.show_seo_keyword_dialog = False
    st.success("Outline generated.")
    st.rerun()


def run_writer_full_generation() -> None:
    st.session_state.pending_writer_full_generation = False

    inputs = current_inputs()
    if not inputs["topic"]:
        st.error("Please enter a topic first.")
        return

    run_evan_light(inputs)
    inputs = current_inputs()

    started_at = time.perf_counter()
    response = generate_text(
        outline_system_prompt(inputs["language"]),
        outline_user_prompt(inputs),
        max_tokens=2200,
    )
    record_timing("Writer outline generation", started_at)
    parsed = parse_json_response(response)
    generated_title = clean_text(parsed.get("title") or inputs["title"] or inputs["topic"])
    manual_title = get_manual_title()
    st.session_state.outline_title = generated_title if st.session_state.ai_title_checkbox else (manual_title or generated_title)
    st.session_state.outline = parsed.get("outline", [])
    st.session_state.sections_content = {}
    st.session_state.sections_workspace_ready = True
    st.session_state.new_section_prompt = ""
    normalise_outline()

    title = clean_text(get_manual_title() or st.session_state.outline_title or inputs["title"] or inputs["topic"])
    generate_missing_sections(inputs, title)

    ordered_sections = [
        {
            "heading": section["heading"],
            "content": sanitise_section_content(
                st.session_state.sections_content.get(section["id"], ""),
                section["heading"],
            ),
        }
        for section in st.session_state.outline
    ]
    draft = sections_to_markdown(ordered_sections)
    st.session_state.writer_full_draft = draft
    st.session_state.writer_full_draft_editor = draft
    st.session_state.writer_export_title = ""
    st.session_state.writer_export_title_source = ""
    if st.session_state.ai_title_checkbox:
        fallback_title = clean_text(st.session_state.topic_input or st.session_state.title or st.session_state.topic or title)
        st.session_state.writer_export_title = resolve_export_title("Writer Version", fallback_title, draft)
    st.success("Blog generated.")
    st.rerun()


def run_ai_outline_generation() -> None:
    st.session_state.pending_ai_outline_generation = False

    inputs = current_inputs()
    topic_seed = clean_text(inputs["topic"] or inputs["title"] or st.session_state.ai_outline_title)

    if not topic_seed:
        st.error("Please enter a topic first.")
        return

    if not inputs["topic"]:
        inputs["topic"] = topic_seed
    if not inputs["title"]:
        inputs["title"] = topic_seed

    run_evan_light(inputs)
    inputs = current_inputs()

    response = generate_text(
        ai_friendly_outline_system_prompt(inputs["language"]),
        ai_friendly_outline_user_prompt(inputs),
        max_tokens=2400,
    )
    parsed = parse_json_response(response)

    generated_title = clean_text(parsed.get("title") or inputs["title"] or inputs["topic"] or topic_seed)
    manual_title = get_manual_title()
    st.session_state.ai_outline_title = generated_title if st.session_state.ai_title_checkbox else (manual_title or generated_title)
    st.session_state.ai_outline = parsed.get("outline", [])
    st.session_state.ai_new_section_prompt = ""
    st.session_state.ai_friendly_draft = ""
    st.session_state.ai_friendly_draft_editor = ""

    normalise_ai_outline()
    st.session_state.show_seo_keyword_dialog = False
    st.success("AI-friendly outline generated.")
    st.rerun()


def run_ai_friendly_generation() -> None:
    st.session_state.pending_ai_friendly_generation = False

    inputs = current_inputs()
    topic_seed = clean_text(
        inputs["topic"] or inputs["title"] or st.session_state.get("ai_outline_title", "")
    )

    if not topic_seed:
        st.error("Please enter a topic first.")
        return

    if not inputs["topic"]:
        inputs["topic"] = topic_seed
    if not inputs["title"]:
        inputs["title"] = topic_seed

    run_evan_light(inputs)
    inputs = current_inputs()

    if not inputs["topic"]:
        inputs["topic"] = topic_seed
    if not inputs["title"]:
        inputs["title"] = topic_seed

    started_at = time.perf_counter()
    outline_response = generate_text(
        ai_friendly_outline_system_prompt(inputs["language"]),
        ai_friendly_outline_user_prompt(inputs),
        max_tokens=2400,
    )
    record_timing("AI-friendly outline generation", started_at)
    outline_parsed = parse_json_response(outline_response)

    generated_outline_title = clean_text(
        outline_parsed.get("title") or inputs["title"] or inputs["topic"] or topic_seed
    )
    manual_title = get_manual_title()
    outline_title = generated_outline_title if st.session_state.ai_title_checkbox else (manual_title or generated_outline_title)
    outline = outline_parsed.get("outline", [])

    st.session_state.ai_outline_title = outline_title
    st.session_state.ai_outline = outline
    normalise_ai_outline()

    seo_keywords = [clean_text(keyword) for keyword in inputs["keywords"] if clean_text(keyword)]

    keyword_instruction = ""
    keyword_footer = ""

    if seo_keywords:
        keyword_instruction = (
            "\n\nAdditional SEO requirement:\n"
            "Use every SEO keyword or phrase below naturally at least once where relevant.\n"
            "Do not keyword-stuff, but do not omit them.\n"
            + "\n".join(f"- {keyword}" for keyword in seo_keywords)
        )

        keyword_footer = "\n\n## SEO keywords used\n\n" + "\n".join(
            f"- {keyword}" for keyword in seo_keywords
        )

    started_at = time.perf_counter()
    response = generate_text(
        ai_friendly_blog_system_prompt(inputs["language"]),
        ai_friendly_blog_user_prompt(
            inputs,
            outline_title=outline_title,
            outline=st.session_state.ai_outline,
        ) + keyword_instruction,
        max_tokens=3800,
    )
    record_timing("AI-friendly blog generation", started_at)

    final_draft = response.strip()
    if keyword_footer:
        final_draft = f"{final_draft}{keyword_footer}"

    st.session_state.ai_friendly_draft = final_draft
    st.session_state.ai_friendly_draft_editor = final_draft
    st.session_state.ai_export_title = ""
    st.session_state.ai_export_title_source = ""
    if st.session_state.ai_title_checkbox:
        fallback_title = clean_text(st.session_state.topic_input or st.session_state.title or st.session_state.topic or outline_title)
        st.session_state.ai_export_title = resolve_export_title("AI Friendly", fallback_title, final_draft)
    st.success("AI-friendly blog generated.")
    st.rerun()


def generate_new_section_from_prompt(one_liner: str) -> dict[str, Any]:
    inputs = current_inputs()
    article_title = clean_text(st.session_state.outline_title or inputs["title"] or inputs["topic"] or "the article")
    existing_headings = [clean_text(section.get("heading", "")) for section in st.session_state.outline if clean_text(section.get("heading", ""))]

    system_prompt = (
        f"You are an expert blog strategist writing in {inputs['language']}. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Create exactly one new blog section for this article.

Article title: {article_title}
Topic: {inputs["topic"]}
Audience: {inputs["audience"]}
Tone: {inputs["tone"]}
Target words for whole article: {inputs["target_words"]}
Existing section headings: {json.dumps(existing_headings, ensure_ascii=False)}
Requested new section idea: {one_liner}

Return JSON in this shape:
{{
  "heading": "...",
  "objective": "...",
  "keyPoints": ["...", "...", "..."],
  "suggestedWords": 180
}}

Rules:
- Make the new section distinct from the existing headings.
- Keep the heading concise and publication-ready.
- Objective should be one clear sentence.
- keyPoints should contain 3 short bullets.
- suggestedWords must be an integer between 120 and 350.
- No markdown, no commentary, JSON only.
"""
    response = generate_text(system_prompt, user_prompt, max_tokens=900)
    parsed = parse_json_response(response)
    return {
        "id": make_section_id(),
        "heading": clean_text(parsed.get("heading", "") or one_liner),
        "objective": clean_text(parsed.get("objective", "") or one_liner),
        "keyPoints": [clean_text(point) for point in parsed.get("keyPoints", []) if clean_text(point)],
        "suggestedWords": int(parsed.get("suggestedWords", 180)),
    }


def generate_new_ai_section_from_prompt(one_liner: str) -> dict[str, Any]:
    inputs = current_inputs()
    article_title = clean_text(
        st.session_state.ai_outline_title
        or inputs["title"]
        or inputs["topic"]
        or "the article"
    )
    topic_value = clean_text(inputs["topic"] or article_title)
    existing_headings = [
        clean_text(section.get("heading", ""))
        for section in st.session_state.ai_outline
        if clean_text(section.get("heading", ""))
    ]

    system_prompt = (
        f"You are an expert blog strategist writing in {inputs['language']}. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Create exactly one new AI-friendly blog section for this article.

Article title: {article_title}
Topic: {topic_value}
Audience: {inputs["audience"]}
Tone: {inputs["tone"]}
Target words for whole article: {inputs["target_words"]}
Existing section headings: {json.dumps(existing_headings, ensure_ascii=False)}
Requested new section idea: {one_liner}

Return JSON in this shape:
{{
  "heading": "...",
  "objective": "...",
  "keyPoints": ["...", "...", "..."],
  "suggestedWords": 180
}}

Rules:
- Make the heading question-led or answer-led where appropriate for an AI-friendly blog.
- Make the new section distinct from the existing headings.
- Keep the heading concise and publication-ready.
- Objective should be one clear sentence.
- keyPoints should contain 3 short bullets.
- suggestedWords must be an integer between 120 and 350.
- No markdown, no commentary, JSON only.
"""
    response = generate_text(system_prompt, user_prompt, max_tokens=900)
    parsed = parse_json_response(response)
    return {
        "id": make_section_id(),
        "heading": clean_text(parsed.get("heading", "") or one_liner),
        "objective": clean_text(parsed.get("objective", "") or one_liner),
        "keyPoints": [clean_text(point) for point in parsed.get("keyPoints", []) if clean_text(point)],
        "suggestedWords": int(parsed.get("suggestedWords", 180)),
    }


def delete_section(section_id: str) -> None:
    st.session_state.outline = [
        section for section in st.session_state.outline if str(section.get("id")) != str(section_id)
    ]
    st.session_state.sections_content.pop(section_id, None)
    st.session_state.pop(f"content_{section_id}", None)
    st.session_state.pop(f"rev_inst_{section_id}", None)
    normalise_outline()
    if not st.session_state.outline:
        st.session_state.sections_workspace_ready = False


def delete_ai_section(section_id: str) -> None:
    st.session_state.ai_outline = [
        section for section in st.session_state.ai_outline if str(section.get("id")) != str(section_id)
    ]
    normalise_ai_outline()


def normalise_outline() -> None:
    cleaned_outline: list[dict[str, Any]] = []

    valid_section_ids: set[str] = set()
    for idx, section in enumerate(st.session_state.outline):
        section_id = str(section.get("id") or f"s{idx+1}")
        valid_section_ids.add(section_id)
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
                "suggestedWords": max(80, int(section.get("suggestedWords", 180))),
            }
        )
        st.session_state.sections_content.setdefault(section_id, "")

    stale_ids = [section_id for section_id in list(st.session_state.sections_content.keys()) if section_id not in valid_section_ids]
    for stale_id in stale_ids:
        st.session_state.sections_content.pop(stale_id, None)
        st.session_state.pop(f"content_{stale_id}", None)
        st.session_state.pop(f"rev_inst_{stale_id}", None)

    st.session_state.outline = cleaned_outline


def normalise_ai_outline() -> None:
    cleaned_outline: list[dict[str, Any]] = []

    for idx, section in enumerate(st.session_state.ai_outline):
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
                "suggestedWords": max(80, int(section.get("suggestedWords", 180))),
            }
        )

    st.session_state.ai_outline = cleaned_outline


def generate_missing_sections(inputs: dict[str, Any], title: str) -> bool:
    missing_sections = []
    outline_snapshot = [dict(section) for section in st.session_state.outline]

    for section in outline_snapshot:
        key = str(section["id"])
        existing = clean_text(st.session_state.sections_content.get(key, ""))
        if existing:
            continue
        missing_sections.append(section)

    if not missing_sections:
        return False

    generated_any = False
    max_workers = parallel_section_workers(len(missing_sections))
    started_at = time.perf_counter()

    if max_workers == 1:
        for section in missing_sections:
            key, section_text = generate_single_section(inputs, title, section, outline_snapshot)
            st.session_state.sections_content[key] = section_text
            st.session_state[f"content_{key}"] = section_text
            generated_any = True
    else:
        ordered_results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_single_section, inputs, title, section, outline_snapshot): str(section["id"])
                for section in missing_sections
            }
            for future in as_completed(futures):
                key, section_text = future.result()
                ordered_results[key] = section_text

        for section in outline_snapshot:
            key = str(section["id"])
            if key in ordered_results:
                st.session_state.sections_content[key] = ordered_results[key]
                st.session_state[f"content_{key}"] = ordered_results[key]
                generated_any = True

    record_timing(f"Section generation ({len(missing_sections)} sections)", started_at)
    return generated_any


def apply_pending_content_updates() -> None:
    pending_generation = st.session_state.pop("pending_generation_update", None)
    if pending_generation:
        section_id = str(pending_generation.get("section_id", "")).strip()
        generated_text = clean_text(pending_generation.get("content", ""))

        if section_id:
            matched_section = next(
                (section for section in st.session_state.outline if str(section.get("id")) == section_id),
                None,
            )
            heading = clean_text((matched_section or {}).get("heading", ""))
            generated_text = sanitise_section_content(generated_text, heading)
            content_key = f"content_{section_id}"
            st.session_state[content_key] = generated_text
            st.session_state.sections_content[section_id] = generated_text
            st.session_state["generation_success_message"] = "Section generated."

    pending_revision = st.session_state.pop("pending_revision_update", None)
    if pending_revision:
        section_id = str(pending_revision.get("section_id", "")).strip()
        revised_text = clean_text(pending_revision.get("content", ""))

        if section_id:
            matched_section = next(
                (section for section in st.session_state.outline if str(section.get("id")) == section_id),
                None,
            )
            heading = clean_text((matched_section or {}).get("heading", ""))
            revised_text = sanitise_section_content(revised_text, heading)
            content_key = f"content_{section_id}"
            st.session_state[content_key] = revised_text
            st.session_state.sections_content[section_id] = revised_text
            st.session_state["revision_success_message"] = "Section revised."

    pending_full_blog_revision = st.session_state.pop("pending_full_blog_revision", None)
    if pending_full_blog_revision:
        revised_markdown = clean_text(pending_full_blog_revision.get("revised_markdown", ""))
        if revised_markdown:
            apply_revised_markdown_to_writer_sections(
                revised_markdown,
                st.session_state.outline,
            )
            st.session_state["revision_success_message"] = "Full blog revised."

    pending_ai_full_blog_revision = st.session_state.pop("pending_ai_full_blog_revision", None)
    if pending_ai_full_blog_revision:
        revised_markdown = clean_text(pending_ai_full_blog_revision.get("revised_markdown", ""))
        if revised_markdown:
            st.session_state.ai_friendly_draft = revised_markdown
            st.session_state.ai_friendly_draft_editor = revised_markdown
            st.session_state.ai_export_title = ""
            st.session_state.ai_export_title_source = ""
            st.session_state["revision_success_message"] = "Full blog revised."


def switch_blog_mode(mode: str) -> None:
    if st.session_state.blog_mode == mode:
        return
    st.session_state.blog_mode = mode


def set_processing(message: str = "Processing your blog...") -> None:
    st.session_state.processing_message = clean_text(message) or "Processing your blog..."


def clear_processing() -> None:
    st.session_state.processing_message = ""


def apply_pending_ai_title() -> None:
    st.session_state.pop("pending_ai_title", "")


def render_processing_overlay() -> None:
    message = clean_text(st.session_state.get("processing_message", ""))
    if not message:
        return

    st.markdown(
        f"""
        <style>
        .processing-overlay-backdrop {{
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.45);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(4px);
        }}

        .processing-overlay-modal {{
            width: min(560px, 92vw);
            background: white;
            border-radius: 28px;
            padding: 2rem 1.8rem;
            box-shadow: 0 28px 70px rgba(15, 23, 42, 0.28);
            text-align: center;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }}

        .processing-spinner {{
            width: 78px;
            height: 78px;
            margin: 0 auto 1.2rem auto;
            border: 7px solid #dbeafe;
            border-top: 7px solid #1368e8;
            border-radius: 50%;
            animation: processing-spin 0.9s linear infinite;
        }}

        .processing-title {{
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }}

        .processing-copy {{
            font-size: 1rem;
            color: #64748b;
            line-height: 1.5;
        }}

        @keyframes processing-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>

        <div class="processing-overlay-backdrop">
            <div class="processing-overlay-modal">
                <div class="processing-spinner"></div>
                <div class="processing-title">Processing...</div>
                <div class="processing-copy">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_controls() -> None:
    st.markdown("### Writing mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Writer",
            key="compact_writer_mode",
            use_container_width=True,
            type="primary" if st.session_state.blog_mode == "Writer Version" else "secondary",
        ):
            switch_blog_mode("Writer Version")
            st.rerun()
    with col2:
        if st.button(
            "AI-Friendly",
            key="compact_ai_mode",
            use_container_width=True,
            type="primary" if st.session_state.blog_mode == "AI Friendly" else "secondary",
        ):
            switch_blog_mode("AI Friendly")
            st.rerun()


init_state()
apply_pending_content_updates()
apply_pending_ai_title()
normalise_outline()
normalise_ai_outline()

render_processing_overlay()

if st.session_state.get("pending_outline_generation"):
    try:
        with st.spinner("Generating outline..."):
            run_outline_generation()
    finally:
        clear_processing()

if st.session_state.get("pending_writer_full_generation"):
    try:
        with st.spinner("Generating full blog..."):
            run_writer_full_generation()
    finally:
        clear_processing()

if st.session_state.get("pending_ai_outline_generation"):
    try:
        with st.spinner("Generating AI-friendly outline..."):
            run_ai_outline_generation()
    finally:
        clear_processing()

if st.session_state.get("pending_ai_friendly_generation"):
    try:
        with st.spinner("Generating AI-friendly blog..."):
            run_ai_friendly_generation()
    finally:
        clear_processing()

if st.session_state.show_seo_keyword_dialog:
    @st.dialog("Choose recommended SEO keywords")
    def seo_keywords_dialog() -> None:
        st.write("Your SEO keywords box is empty. Pick any recommended keywords below, then continue.")

        suggestions = st.session_state.get("seo_keyword_suggestions", [])
        selected = st.session_state.get("selected_seo_keywords", [])

        if not suggestions:
            st.info("No SEO suggestions are available right now.")
        else:
            cols_per_row = 3
            for start in range(0, len(suggestions), cols_per_row):
                cols = st.columns(cols_per_row)
                for idx, keyword in enumerate(suggestions[start:start + cols_per_row]):
                    is_selected = keyword in selected
                    button_label = f"✅ {keyword}" if is_selected else keyword
                    button_type = "primary" if is_selected else "secondary"

                    with cols[idx]:
                        if st.button(
                            button_label,
                            key=f"seo_pick_{start}_{idx}",
                            use_container_width=True,
                            type=button_type,
                        ):
                            if is_selected:
                                st.session_state.selected_seo_keywords = [
                                    item for item in st.session_state.selected_seo_keywords if item != keyword
                                ]
                            else:
                                st.session_state.selected_seo_keywords = [
                                    *st.session_state.selected_seo_keywords,
                                    keyword,
                                ]
                            st.rerun()

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_seo_keyword_dialog = False
                st.rerun()

        with action_col2:
            if st.button("Continue", use_container_width=True, type="primary"):
                selected_keywords = st.session_state.get("selected_seo_keywords", [])

                if not selected_keywords:
                    st.warning("Please select at least one SEO keyword.")
                else:
                    st.session_state.keywords_text = "\n".join(selected_keywords)
                    st.session_state.selected_seo_keywords = selected_keywords
                    st.session_state.show_seo_keyword_dialog = False
                    if st.session_state.blog_mode == "Writer Version":
                        set_processing("Generating your full blog draft...")
                        st.session_state.pending_writer_full_generation = True
                    else:
                        set_processing("Generating your AI-friendly blog draft...")
                        st.session_state.pending_ai_friendly_generation = True
                    st.rerun()

    seo_keywords_dialog()

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }

    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 1.1rem;
        max-width: 1440px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(15, 23, 42, 0.08);
        min-width: 285px !important;
        max-width: 285px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.8rem;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }

    h1, h2, h3 {
        color: #0f172a;
        letter-spacing: -0.02em;
    }

    .sidebar-brand {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 0.45rem;
    }

    .sidebar-icon {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        background: #1368e8;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 18px;
        flex-shrink: 0;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 2px;
    }

    .sidebar-subtitle {
        font-size: 0.93rem;
        color: #64748b;
        line-height: 1.2;
    }

    .sidebar-label {
        font-size: 0.78rem;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.65rem;
        margin-bottom: 0.25rem;
    }

    .history-chip {
        background: #eef3f9;
        padding: 0.72rem 0.85rem;
        border-radius: 12px;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.2;
        border: 1px solid rgba(15, 23, 42, 0.04);
    }

    .main-wrap {
        max-width: 820px;
        margin: 0 auto;
        padding-top: 0.05rem;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 760;
        color: #0f172a;
        margin-bottom: 0.45rem;
    }

    .page-subtitle {
        font-size: 1.02rem;
        color: #64748b;
        margin-bottom: 0.9rem;
    }

    .form-card {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.1rem 1.15rem 0.95rem 1.15rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.95rem;
    }

    .result-card {
        background: #eaf3ff;
        border: 1px solid rgba(59, 130, 246, 0.12);
        border-radius: 20px;
        padding: 1rem 1.15rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.15rem;
    }

    .result-copy {
        font-size: 0.98rem;
        color: #5b6b80;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border-radius: 14px !important;
        min-height: 46px !important;
        background: #ffffff !important;
        border: 1px solid #d8e0ea !important;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 16px !important;
        background: #ffffff !important;
        border: 1px solid #d8e0ea !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 14px !important;
        min-height: 46px !important;
        background: #ffffff !important;
        border: 1px solid #d8e0ea !important;
    }

    div[data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #d8e0ea !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px !important;
        min-height: 46px !important;
        font-weight: 700 !important;
    }

    .big-primary .stButton > button {
        background: #1368e8 !important;
        color: white !important;
        border: none !important;
        min-height: 48px !important;
        font-size: 1rem !important;
    }

    .small-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -0.2rem;
        margin-bottom: 0.45rem;
    }

    .muted-divider {
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is missing. Add it in your environment before using the app.")

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-icon">✎</div>
            <div>
                <div class="sidebar-title">Alvan</div>
                <div class="sidebar-subtitle">Blog Writer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_mode_controls()

    st.markdown("<div class='sidebar-label'>Specify Audience</div>", unsafe_allow_html=True)
    st.text_input(
        "Specify Audience",
        value="Marketing Junction",
        key="audience",
        label_visibility="collapsed",
        placeholder="Describe your target audience",
    )

    st.markdown("<div class='sidebar-label'>Language</div>", unsafe_allow_html=True)
    st.radio(
        "Language",
        options=LANGUAGE_OPTIONS,
        key="language",
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-label'>Keywords</div>", unsafe_allow_html=True)
    st.text_input(
        "Keywords",
        key="keywords_sidebar_compact",
        value=st.session_state.keywords_text.replace("\n", ", "),
        placeholder="keyword1, keyword2...",
        label_visibility="collapsed",
    )
    sidebar_keywords = clean_text(st.session_state.get("keywords_sidebar_compact", ""))
    if sidebar_keywords:
        st.session_state.keywords_text = "\n".join(
            [item.strip() for item in sidebar_keywords.split(",") if item.strip()]
        )

    st.markdown("<div class='sidebar-label'>Word count</div>", unsafe_allow_html=True)
    word_range_options = {
        "600-900": 800,
        "750-1500": 1200,
        "1200-1800": 1500,
        "1800-2500": 2200,
    }

    default_label = "750-1500"
    current_target = int(st.session_state.target_words)
    if current_target <= 900:
        default_label = "600-900"
    elif current_target <= 1500:
        default_label = "750-1500"
    elif current_target <= 1800:
        default_label = "1200-1800"
    else:
        default_label = "1800-2500"

    selected_range = st.selectbox(
        "Word count",
        options=list(word_range_options.keys()),
        index=list(word_range_options.keys()).index(default_label),
        label_visibility="collapsed",
    )
    st.session_state.target_words = word_range_options[selected_range]

    st.checkbox(
        "Hiring impact section",
        key="add_hiring_section",
    )

    st.markdown("<div class='sidebar-label'>History</div>", unsafe_allow_html=True)
    history_value = clean_text(
        st.session_state.topic
        or st.session_state.title
        or st.session_state.ai_outline_title
        or "No topic yet"
    )
    st.markdown(
        f"<div class='history-chip'>{history_value}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)
st.markdown("<div class='page-title'>Create a Blog Post</div>", unsafe_allow_html=True)

if not st.session_state.topic_input:
    st.session_state.topic_input = clean_text(
        st.session_state.topic
        or st.session_state.title
        or st.session_state.ai_outline_title
        or st.session_state.outline_title
    )

st.markdown("<div class='form-card'>", unsafe_allow_html=True)

top_row_col1, top_row_col2 = st.columns([5, 1.2], gap="small")
with top_row_col1:
    st.text_input(
        "Blog Title / Topic",
        key="topic_input",
        placeholder="why recruitment marketing matters",
    )
with top_row_col2:
    st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
    st.checkbox("AI Title", key="ai_title_checkbox")

ai_title_checked = bool(st.session_state.ai_title_checkbox)
st.session_state.ai_title_checkbox_prev = ai_title_checked

facts_col, quotes_col = st.columns(2)
with facts_col:
    st.text_area(
        "Key Facts & Figures",
        key="facts_text",
        height=105,
        placeholder="Statistical data, research findings, important facts...",
    )

with quotes_col:
    st.text_area(
        "Quotes & Insights",
        key="quotes_text",
        height=105,
        placeholder="Expert quotes, industry insights, original perspectives...",
    )

st.text_area(
    "Paste in your copies",
    key="research_notes",
    height=80,
    placeholder="Paste your content or additional notes here...",
)

st.markdown("#### Supporting Document")
uploaded = st.file_uploader(
    "Upload supporting document",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    key="supporting_file_uploader_main",
    label_visibility="collapsed",
)

if uploaded is not None:
    extract_col1, extract_col2 = st.columns([1.4, 4])
    with extract_col1:
        if st.button("Extract insights", use_container_width=True):
            try:
                set_processing("Extracting insights from your uploaded document...")
                render_processing_overlay()
                raw_text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
                st.session_state.document_text = raw_text
                response = generate_text(
                    insights_system_prompt(st.session_state.language),
                    insights_user_prompt(
                        raw_text,
                        st.session_state.topic or st.session_state.title or st.session_state.ai_outline_title or "blog topic",
                        st.session_state.language,
                    ),
                    max_tokens=1800,
                )
                parsed = parse_json_response(response)
                st.session_state.document_insights = parsed.get("insights", [])[:12]
                st.session_state.quoted_lines = parsed.get("quoted_lines", [])[:12]
                clear_processing()
                st.success("Document insights extracted.")
            except Exception as exc:
                clear_processing()
                st.error(f"Could not process upload: {exc}")
    with extract_col2:
        st.caption(uploaded.name)

st.markdown("<div class='big-primary'>", unsafe_allow_html=True)

if st.session_state.blog_mode == "Writer Version":
    if st.button("Generate Blog Article", type="primary", use_container_width=True):
        try:
            inputs = current_inputs()
            if not inputs["topic"]:
                st.error("Please enter a topic first.")
            elif not inputs["keywords"]:
                suggested_keywords = suggest_seo_keywords(inputs)
                st.session_state.seo_keyword_suggestions = suggested_keywords
                st.session_state.selected_seo_keywords = []
                st.session_state.show_seo_keyword_dialog = True
                st.rerun()
            else:
                set_processing("Generating your full blog draft...")
                st.session_state.pending_writer_full_generation = True
                st.rerun()
        except Exception as exc:
            clear_processing()
            st.error(f"Could not generate blog: {exc}")
else:
    if st.button("Generate Blog Article", type="primary", use_container_width=True):
        try:
            inputs = current_inputs()
            topic_seed = clean_text(inputs["topic"] or inputs["title"] or st.session_state.ai_outline_title)
            if not topic_seed:
                st.error("Please enter a topic first.")
            elif not inputs["keywords"]:
                suggested_keywords = suggest_seo_keywords(inputs)
                st.session_state.seo_keyword_suggestions = suggested_keywords
                st.session_state.selected_seo_keywords = []
                st.session_state.show_seo_keyword_dialog = True
                st.rerun()
            else:
                set_processing("Generating your AI-friendly blog draft...")
                st.session_state.pending_ai_friendly_generation = True
                st.rerun()
        except Exception as exc:
            clear_processing()
            st.error(f"Could not generate AI-friendly outline: {exc}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.get("generation_success_message"):
    st.success(st.session_state.pop("generation_success_message"))

if st.session_state.get("revision_success_message"):
    st.success(st.session_state.pop("revision_success_message"))

verified = st.session_state.get("verified_evidence", {}) or {}
if verified.get("verified_points") or verified.get("verified_quotes"):
    with st.expander("Verified evidence", expanded=False):
        if verified.get("verified_points"):
            st.markdown("**Verified points**")
            for item in verified.get("verified_points", [])[:10]:
                st.write(f"- {item}")
        if verified.get("verified_quotes"):
            st.markdown("**Verified quotes**")
            for item in verified.get("verified_quotes", [])[:6]:
                st.write(f"- {item}")
        if verified.get("unsupported_points"):
            st.markdown("**Unsupported or weak points**")
            for item in verified.get("unsupported_points", [])[:6]:
                st.write(f"- {item}")

if st.session_state.document_insights:
    with st.expander("Document insights", expanded=False):
        for item in st.session_state.document_insights:
            st.write(f"- {item}")


else:
    pass

st.markdown("## Export")

if st.session_state.blog_mode == "Writer Version":
    if st.session_state.writer_full_draft.strip():
        inputs = current_inputs()
        user_export_title = clean_text(
            st.session_state.topic_input or st.session_state.title or st.session_state.topic
        )
        title = resolve_export_title(
            "Writer Version",
            user_export_title or "Blog article",
            st.session_state.writer_full_draft,
        )
        export_sections = markdown_to_export_sections(st.session_state.writer_full_draft, title)
        export_sections = build_export_sections_with_appendix(export_sections, inputs["keywords"])

        preview_markdown = "\n\n".join(
            f"## {item['heading']}\n\n{item['content']}"
            for item in export_sections
            if item["content"].strip()
        )

        with st.expander("Preview article", expanded=False):
            preview_height = calc_text_area_height(
                preview_markdown,
                min_height=500,
                line_px=26,
                extra_lines=8,
            )
            st.text_area("Combined article preview", value=preview_markdown, height=preview_height)

            st.text_input(
                "Revision instruction for full blog",
                key="full_blog_revision_prompt",
                placeholder="e.g. Make the full blog more persuasive, tighten repetition, and improve the conclusion",
            )

            if st.button("Revise full blog", key="revise_full_writer_blog", use_container_width=True):
                try:
                    revision_instruction = clean_text(st.session_state.full_blog_revision_prompt)
                    if not revision_instruction:
                        st.warning("Enter a revision instruction for the full blog.")
                    else:
                        revised_full_blog = clean_text(
                            generate_text(
                                revision_system_prompt(inputs["language"]),
                                revision_user_prompt(
                                    title or "Full article",
                                    st.session_state.writer_full_draft,
                                    revision_instruction,
                                    inputs["language"],
                                ),
                                max_tokens=4200,
                            )
                        )
                        queue_full_blog_revision_for_writer(revised_full_blog)
                        st.rerun()
                except Exception as exc:
                    st.error(f"Full blog revision failed: {exc}")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            docx_bytes = export_blog_docx(title or "blog-article", export_sections)
            st.download_button(
                "Download DOCX",
                data=docx_bytes,
                file_name="blog-article.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        with export_col2:
            pdf_bytes = export_blog_pdf(title or "blog-article", export_sections)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="blog-article.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("Generate content first to enable export.")
else:
    if st.session_state.ai_friendly_draft.strip():
        inputs = current_inputs()
        user_export_title = clean_text(
            st.session_state.topic_input or st.session_state.title or st.session_state.topic
        )
        export_title = resolve_export_title(
            "AI Friendly",
            user_export_title or "AI-friendly blog",
            st.session_state.ai_friendly_draft,
        )
        export_sections = markdown_to_export_sections(st.session_state.ai_friendly_draft, export_title)
        export_sections = build_export_sections_with_appendix(export_sections, inputs["keywords"])

        preview_markdown = "\n\n".join(
            f"## {item['heading']}\n\n{item['content']}"
            for item in export_sections
            if item["content"].strip()
        )

        with st.expander("Preview article", expanded=False):
            preview_height = calc_text_area_height(
                preview_markdown,
                min_height=500,
                line_px=26,
                extra_lines=8,
            )
            st.text_area("Combined article preview", value=preview_markdown, height=preview_height)

            st.text_input(
                "Revision instruction for full blog",
                key="ai_full_blog_revision_prompt",
                placeholder="e.g. Make the blog clearer, reduce fluff, and strengthen the opening",
            )

            if st.button("Revise full blog", key="revise_full_ai_blog", use_container_width=True):
                try:
                    revision_instruction = clean_text(st.session_state.ai_full_blog_revision_prompt)
                    if not revision_instruction:
                        st.warning("Enter a revision instruction for the full blog.")
                    else:
                        revised_ai_blog = clean_text(
                            generate_text(
                                revision_system_prompt(inputs["language"]),
                                revision_user_prompt(
                                    export_title or "Full article",
                                    st.session_state.ai_friendly_draft,
                                    revision_instruction,
                                    inputs["language"],
                                ),
                                max_tokens=4200,
                            )
                        )
                        queue_full_blog_revision_for_ai(revised_ai_blog)
                        st.rerun()
                except Exception as exc:
                    st.error(f"Full blog revision failed: {exc}")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            docx_bytes = export_blog_docx(export_title or "blog-article", export_sections)
            st.download_button(
                "Download DOCX",
                data=docx_bytes,
                file_name="blog-article.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        with export_col2:
            pdf_bytes = export_blog_pdf(export_title or "blog-article", export_sections)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="blog-article.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("Generate the AI-friendly draft first to enable export.")

st.markdown("</div>", unsafe_allow_html=True)
