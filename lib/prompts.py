from __future__ import annotations

from typing import Iterable

TONE_OPTIONS = [
    "Thought leadership",
    "Conversational",
    "Professional",
    "Editorial",
]


def _bullet_lines(items: Iterable[str]) -> str:
    cleaned = [x.strip() for x in items if str(x).strip()]
    return "\n".join(f"- {x}" for x in cleaned) if cleaned else "- None provided"


def outline_system_prompt() -> str:
    return (
        "You are an expert B2B content strategist and blog editor. "
        "Create practical, high-quality blog outlines that are specific, structured, and useful for long-form content production. "
        "Do not write the full article. Return only the outline in clean JSON."
    )


def outline_user_prompt(inputs: dict) -> str:
    return f'''
Create a blog outline in valid JSON.

Required JSON shape:
{{
  "title": "string",
  "outline": [
    {{
      "id": "s1",
      "heading": "string",
      "objective": "string",
      "keyPoints": ["string"],
      "suggestedWords": 180
    }}
  ]
}}

Instructions:
- Create 4 to 6 sections.
- The opening section should act as the introduction.
- The final section should act as the conclusion.
- Make the structure logical and specific to the topic.
- Use the facts and quotes where they are relevant.
- Keep headings editorial, not generic.
- Suggested words across all sections should roughly match the target word count.

Topic: {inputs['topic']}
Working title: {inputs['title']}
Audience: {inputs['audience']}
Tone: {inputs['tone']}
Target words: {inputs['target_words']}
SEO keywords:
{_bullet_lines(inputs['keywords'])}
Facts:
{_bullet_lines(inputs['facts'])}
Quotes:
{_bullet_lines(inputs['quotes'])}
Research notes:
{inputs.get('research_notes') or 'None provided'}
Document insights:
{_bullet_lines(inputs.get('document_insights', []))}
'''.strip()


def insights_system_prompt() -> str:
    return (
        "You analyze uploaded research material for blog writing. "
        "Extract concrete, useful, non-redundant insights. Return only valid JSON."
    )


def insights_user_prompt(raw_text: str, topic: str) -> str:
    clipped = raw_text[:12000]
    return f'''
Read the material below and extract insights useful for writing a blog on this topic: {topic}

Return valid JSON in this shape:
{{
  "insights": ["string"],
  "quoted_lines": ["string"]
}}

Rules:
- Keep insights factual and concise.
- Prefer concrete findings, examples, numbers, and claims.
- Do not invent anything not present in the source.

Source material:
{clipped}
'''.strip()


def section_system_prompt() -> str:
    return (
        "You are a senior B2B blog writer. Write polished, engaging, non-generic article sections. "
        "Avoid fluffy openings, repetitive transitions, and empty filler. Use the provided facts and quotes faithfully."
    )


def section_user_prompt(inputs: dict, section: dict, title: str, outline: list[dict]) -> str:
    outline_text = "\n".join(
        f"- {item['heading']}: {item['objective']} ({item['suggestedWords']} words)"
        for item in outline
    )
    return f'''
Write only this one section of the blog article.

Blog title: {title}
Topic: {inputs['topic']}
Audience: {inputs['audience']}
Tone: {inputs['tone']}
Target total words: {inputs['target_words']}

Full outline:
{outline_text}

Current section heading: {section['heading']}
Current section objective: {section['objective']}
Required key points:
{_bullet_lines(section.get('keyPoints', []))}
Suggested words for this section: {section['suggestedWords']}

Facts that must be reflected where relevant:
{_bullet_lines(inputs['facts'])}
Quotes that should be used verbatim where relevant:
{_bullet_lines(inputs['quotes'])}
SEO keywords:
{_bullet_lines(inputs['keywords'])}
Research notes:
{inputs.get('research_notes') or 'None provided'}
Document insights:
{_bullet_lines(inputs.get('document_insights', []))}

Rules:
- Write only the section body, not the heading.
- Use specific, human editorial language.
- Do not start with generic filler like "In today's fast-paced world".
- If this is the introduction, open with a strong hook.
- If this is the conclusion, synthesize the argument instead of repeating the article.
- Do not fabricate statistics or quotes.
'''.strip()


def revision_system_prompt() -> str:
    return (
        "You are a precise content editor. Rewrite only the provided section while following the requested instruction. "
        "Keep the substance accurate and preserve useful facts."
    )


def revision_user_prompt(section_heading: str, section_content: str, instruction: str) -> str:
    return f'''
Section heading: {section_heading}
Revision instruction: {instruction}

Rewrite the section below:
{section_content}
'''.strip()
