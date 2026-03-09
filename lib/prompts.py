from __future__ import annotations

from typing import Iterable, Optional

TONE_OPTIONS = [
    "Thought leadership",
    "Conversational",
    "Professional",
    "Editorial",
]

LANGUAGE_OPTIONS = [
    "UK English",
    "US English",
]


def _bullet_lines(items: Iterable[str]) -> str:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return "\n".join(f"- {x}" for x in cleaned) if cleaned else "- None provided"


def _language_label(language: Optional[str] = None) -> str:
    return "US English" if str(language or "").strip() == "US English" else "UK English"


def _language_rules(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    if chosen_language == "US English":
        spelling_guidance = "Use American spelling, grammar, and punctuation."
    else:
        spelling_guidance = "Use British spelling, grammar, and punctuation."

    return f"""
Language rules:
- Write in {chosen_language}.
- {spelling_guidance}
- Keep the language choice consistent throughout the output.
""".strip()


def _style_rules(language: Optional[str] = None) -> str:
    return f"""
{_language_rules(language)}
- Do not use em dashes (—).
- Do not use en dashes (–).
- Use commas, full stops, or colons instead of long dashes.
- Avoid AI-sounding phrasing and empty filler.
- Avoid overdramatic phrasing.
- Keep the writing natural, polished, and human.
""".strip()


def evaluate_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You evaluate source material for blog writing. "
        "Identify the most useful material, candidate claims, examples, statistics, and direct quotes. "
        "Separate strong evidence from vague or weak material. "
        "Return only valid JSON. "
        f"Use {chosen_language} and do not use em dashes or en dashes anywhere."
    )


def evaluate_user_prompt(evidence_text: str, topic: str, language: Optional[str] = None) -> str:
    clipped = str(evidence_text or "")[:15000]
    return f'''
Evaluate the material below for writing a blog on this topic: {topic}

Return valid JSON in this shape:
{{
  "useful_points": ["string"],
  "candidate_claims": ["string"],
  "examples": ["string"],
  "statistics": ["string"],
  "quotes": ["string"],
  "weak_points": ["string"]
}}

Rules:
- Keep points factual and concise.
- Prefer concrete findings, examples, numbers, and clearly stated claims.
- Do not invent anything not present in the source.
- Use {_language_label(language)}.
- Do not use em dashes or en dashes.

{_style_rules(language)}

Source material:
{clipped}
'''.strip()


def verify_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You verify evidence for blog writing. "
        "Check whether each candidate point or quote is supported by the provided source material. "
        "Only keep material that is supported. "
        "Return only valid JSON. "
        f"Use {chosen_language} and do not use em dashes or en dashes anywhere."
    )


def verify_user_prompt(
    evaluated_material_json: str,
    evidence_text: str,
    topic: str,
    language: Optional[str] = None,
) -> str:
    clipped_candidates = str(evaluated_material_json or "")[:9000]
    clipped_source = str(evidence_text or "")[:15000]
    return f'''
Verify the candidate material below for writing a blog on this topic: {topic}

Return valid JSON in this shape:
{{
  "verified_points": ["string"],
  "verified_quotes": ["string"],
  "unsupported_points": ["string"]
}}

Rules:
- Only mark a point or quote as verified if it is supported by the source material.
- Keep wording faithful to the source where possible.
- Exclude anything vague, unsupported, or weakly implied.
- Do not invent support.
- Use {_language_label(language)}.
- Do not use em dashes or en dashes.

{_style_rules(language)}

Candidate material:
{clipped_candidates}

Source material:
{clipped_source}
'''.strip()


def outline_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You are an expert B2B content strategist and blog editor. "
        "Create practical, high-quality blog outlines that are specific, structured, and useful for long-form content production. "
        "Do not write the full article. Return only the outline in clean JSON. "
        f"Use {chosen_language} and do not use em dashes or en dashes anywhere."
    )


def outline_user_prompt(inputs: dict) -> str:
    verified = inputs.get("verified_evidence") or {}
    verified_points = verified.get("verified_points", [])
    verified_quotes = verified.get("verified_quotes", [])
    unsupported_points = verified.get("unsupported_points", [])

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
- Use {inputs['language']} in the title and all section headings.
- Do not use em dashes or en dashes in the title, headings, or objectives.
- Prioritise verified evidence where available.
- If verified evidence exists, do not build major sections around unsupported points.
- If little or no verified evidence exists, you may rely on general knowledge, but avoid overly specific statistics or named claims unless supplied.
- If add_hiring_section is true, include at least one dedicated section on the impact on hiring, talent strategy, recruitment, employer branding, or workforce implications where relevant to the topic.

{_style_rules(inputs['language'])}

Topic: {inputs['topic']}
Working title: {inputs['title']}
Audience: {inputs['audience']}
Tone: {inputs['tone']}
Target words: {inputs['target_words']}
Verified points:
{_bullet_lines(verified_points)}
Verified quotes:
{_bullet_lines(verified_quotes)}
Unsupported or weak points to avoid treating as established fact:
{_bullet_lines(unsupported_points)}
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
Add hiring section:
{'Yes' if inputs.get('add_hiring_section') else 'No'}
'''.strip()


def insights_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You analyse uploaded research material for blog writing. "
        "Extract concrete, useful, non-redundant insights. Return only valid JSON. "
        f"Use {chosen_language} and do not use em dashes or en dashes."
    )


def insights_user_prompt(raw_text: str, topic: str, language: Optional[str] = None) -> str:
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
- Use {_language_label(language)}.
- Do not use em dashes or en dashes.

{_style_rules(language)}

Source material:
{clipped}
'''.strip()


def section_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You are a senior B2B blog writer. Write polished, engaging, non-generic article sections. "
        "Avoid fluffy openings, repetitive transitions, and empty filler. "
        "Use the provided facts and quotes faithfully. "
        f"Write in {chosen_language}. Do not use em dashes or en dashes. "
        "Use commas, full stops, or colons instead."
    )


def section_user_prompt(inputs: dict, section: dict, title: str, outline: list) -> str:
    outline_text = "\n".join(
        f"- {item['heading']}: {item['objective']} ({item['suggestedWords']} words)"
        for item in outline
    )
    verified = inputs.get("verified_evidence") or {}
    verified_points = verified.get("verified_points", [])
    verified_quotes = verified.get("verified_quotes", [])
    unsupported_points = verified.get("unsupported_points", [])

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

Verified points to use where relevant:
{_bullet_lines(verified_points)}
Verified quotes to use verbatim where relevant:
{_bullet_lines(verified_quotes)}
Unsupported or weak points that must not be presented as established fact:
{_bullet_lines(unsupported_points)}
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
Add hiring section:
{'Yes' if inputs.get('add_hiring_section') else 'No'}

Rules:
- Write only the section body, not the heading.
- Use specific, human editorial language.
- Write in {inputs['language']}.
- Keep spelling, grammar, punctuation, and phrasing consistent with {inputs['language']}.
- Do not use em dashes (—).
- Do not use en dashes (–).
- Use commas, full stops, or colons instead.
- Do not start with generic filler like "In today's fast-paced world".
- If this is the introduction, open with a strong hook.
- If this is the conclusion, synthesise the argument instead of repeating the article.
- Do not fabricate statistics or quotes.
- Use verified evidence where relevant.
- Use verified quotes verbatim where relevant.
- Do not introduce unsupported numbers, named studies, or precise claims as fact.
- If no verified evidence is available, write from general knowledge and keep claims broad rather than overly specific.
- If add_hiring_section is true and this section is the hiring-related section, cover the impact on hiring, recruitment, talent strategy, employer brand, role design, or workforce planning in a practical way.
- Avoid repetitive sentence openings.
- Avoid robotic transitions such as "It is important to note" or "Another key point is".
- Keep the rhythm natural and readable.

{_style_rules(inputs['language'])}
'''.strip()


def revision_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You are a precise content editor. Rewrite only the provided section while following the requested instruction. "
        "Keep the substance accurate and preserve useful facts. "
        f"Write in {chosen_language}. Do not use em dashes or en dashes."
    )


def revision_user_prompt(section_heading: str, section_content: str, instruction: str, language: Optional[str] = None) -> str:
    return f'''
Section heading: {section_heading}
Revision instruction: {instruction}

Rewrite the section below.

Rules:
- Keep the meaning accurate.
- Use {_language_label(language)}.
- Do not use em dashes (—).
- Do not use en dashes (–).
- Use commas, full stops, or colons instead.
- Keep the writing natural and human.
- Do not introduce unsupported statistics, studies, or quotes.

{_style_rules(language)}

Section content:
{section_content}
'''.strip()


def ai_outline_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You are an expert content strategist for AI-friendly, search-friendly blogs. "
        "Create editable question-led article plans, not full articles. "
        "Return only valid JSON. "
        f"Use {chosen_language} and do not use em dashes or en dashes anywhere."
    )


def ai_outline_user_prompt(inputs: dict) -> str:
    verified = inputs.get("verified_evidence") or {}
    verified_points = verified.get("verified_points", [])
    verified_quotes = verified.get("verified_quotes", [])
    unsupported_points = verified.get("unsupported_points", [])

    return f'''
Create an AI-friendly blog outline in valid JSON.

Required JSON shape:
{{
  "title": "string",
  "outline": [
    {{
      "id": "aq1",
      "question": "string",
      "keyTakeaway": "string",
      "objective": "string",
      "suggestedWords": 180
    }}
  ]
}}

Instructions:
- Create 5 to 7 question-led sections.
- Each question must be specific, natural, and useful to a searcher.
- Each keyTakeaway must be a single concise sentence.
- Each objective must explain what that section should cover in practical terms.
- The opening question should act as the introduction.
- Include at least one question-led how-to section.
- Include one FAQ section objective near the end.
- Include one TL;DR or wrap-up section objective at the end.
- Suggested words across all sections should roughly match the target word count.
- Prioritise verified evidence where available.
- If verified evidence exists, do not build major sections around unsupported points.
- If add_hiring_section is true, include at least one dedicated question on hiring, recruitment, talent strategy, employer branding, or workforce implications where relevant.

{_style_rules(inputs['language'])}

Topic: {inputs['topic']}
Working title: {inputs['title']}
Audience: {inputs['audience']}
Tone: {inputs['tone']}
Target words: {inputs['target_words']}
Verified points:
{_bullet_lines(verified_points)}
Verified quotes:
{_bullet_lines(verified_quotes)}
Unsupported or weak points to avoid treating as established fact:
{_bullet_lines(unsupported_points)}
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
Add hiring section:
{'Yes' if inputs.get('add_hiring_section') else 'No'}
'''.strip()


def ai_friendly_blog_system_prompt(language: Optional[str] = None) -> str:
    chosen_language = _language_label(language)
    return (
        "You are an expert blog writer specialising in AI-friendly, search-friendly, answer-engine-friendly content. "
        "Write complete blog posts that are easy to scan, question-led, useful, and grounded in provided evidence. "
        "The output must be publication-ready markdown. "
        f"Use {chosen_language}. Do not use em dashes or en dashes anywhere."
    )


def ai_friendly_blog_user_prompt(inputs: dict, outline_title: str = "", outline_text: str = "") -> str:
    verified = inputs.get("verified_evidence") or {}
    verified_points = verified.get("verified_points", [])
    verified_quotes = verified.get("verified_quotes", [])
    unsupported_points = verified.get("unsupported_points", [])

    word_count = int(inputs["target_words"])
    topic = inputs["topic"] or inputs["title"] or "the topic"
    audience = inputs["audience"] or "a general professional audience"

    return f'''
Write a {word_count}-word blog post about "{topic}" for {audience}.

Make it AI-friendly using this structure.

Format requirements:
- Use a strong H1 title at the top.
- Include a short introduction after the title.
- Use question headings as H2s, such as "What is X?" or "How do I do Y?"
- Answer each question immediately with 1 to 2 clear sentences before expanding further.
- Start major sections with **Key takeaway:** followed by one concise sentence.
- Include at least one numbered step-by-step process.
- Include at least one practical "how-to" section with clear steps.
- End with exactly 5 FAQ pairs.
- End with a final TL;DR section.
- Use markdown formatting.
- If add_hiring_section is true, include at least one dedicated H2 section covering the impact on hiring.

Content requirements:
- Include 2 to 3 specific examples with real numbers or real results only when supported by the provided evidence.
- Do not make anything up.
- If exact numbers are not supported by the provided material, do not invent them and do not pretend they exist.
- Include actionable tips readers can use immediately.
- Use the provided facts, quotes, research notes, and document insights where relevant.
- Use verified evidence first.
- Do not present unsupported points as fact.
- Keep examples grounded and faithful to the material supplied.

Writing style:
- Conversational and easy to scan.
- No jargon unless it is explained simply.
- Short paragraphs, ideally 2 to 3 sentences maximum.
- Use bullet points where helpful.
- Natural and human, not robotic.
- No generic filler openings.
- No exaggerated claims.

Language and style rules:
- Write in {inputs['language']}.
- Do not use em dashes (—).
- Do not use en dashes (–).
- Use commas, full stops, or colons instead.

SEO and metadata:
- Naturally incorporate the SEO keywords where relevant.
- Keep the article useful first, optimised second.

Outline to follow closely:
Title to use if suitable:
{outline_title or inputs["title"] or inputs["topic"]}

Editable AI-friendly outline:
{outline_text or "- No outline provided"}

Important evidence rules:
- Only use verified quotes verbatim.
- Do not introduce unsupported statistics, named reports, or precise claims as established fact.
- If evidence is limited, keep claims broad, practical, and honest.

Inputs:
Working title:
{inputs['title'] or 'None provided'}

Topic:
{inputs['topic']}

Audience:
{inputs['audience']}

Tone:
{inputs['tone']}

Target words:
{inputs['target_words']}

SEO keywords:
{_bullet_lines(inputs['keywords'])}

Verified points:
{_bullet_lines(verified_points)}

Verified quotes:
{_bullet_lines(verified_quotes)}

Unsupported or weak points to avoid treating as fact:
{_bullet_lines(unsupported_points)}

Facts:
{_bullet_lines(inputs['facts'])}

Quotes:
{_bullet_lines(inputs['quotes'])}

Research notes:
{inputs.get('research_notes') or 'None provided'}

Document insights:
{_bullet_lines(inputs.get('document_insights', []))}
Add hiring section:
{'Yes' if inputs.get('add_hiring_section') else 'No'}

Return only the final blog post in markdown.
'''.strip()
