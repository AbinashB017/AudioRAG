import json
import re
from dataclasses import dataclass
from openai import OpenAI
from .config import Config

@dataclass
class EpisodeSummary:
    episode_id: str
    episode_title: str
    topics: list[str]
    concepts: list[str]
    description: str

_SYSTEM_PROMPT = """\
You are an expert podcast summarizer. Your job is to take a full transcript of an episode of the Fermi Podcast (which explores landmark scientific papers) and produce a highly structured summary.

The summary MUST be output as a valid JSON object matching this schema exactly:
{
  "topics": ["High-level topic 1", "High-level topic 2", ...],
  "concepts": ["Specific technical term 1", "Specific technical term 2", ...],
  "description": "A 2-3 sentence description of what the episode is about and who it's useful for."
}
No other text or markdown formatting. Just the JSON object.
"""

def generate_episode_summary(
    transcript: list[dict],
    episode_id: str,
    episode_title: str,
    config: Config
) -> EpisodeSummary:
    """Generate a structured summary of an entire episode using the LLM.
    
    The transcript is passed directly as it easily fits inside modern LLM context limits
    (longest episode is ~10k tokens).
    """
    client = OpenAI(
        api_key=config.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/fermi-podcast-companion",
            "X-Title": "Fermi Podcast Companion",
        },
    )
    
    full_text = " ".join(s["text"] for s in transcript)
    
    user_message = (
        f"Episode ID: {episode_id}\n"
        f"Episode Title: {episode_title}\n\n"
        f"TRANSCRIPT:\n{full_text}\n\n"
        f"Please provide the structured JSON summary."
    )
    
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0,
    )
    
    raw = response.choices[0].message.content or ""
    
    # Strip potential markdown block
    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md:
        raw = md.group(1).strip()
        
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback regex extraction if there's trailing garbage
        obj = re.search(r"\{.*\}", raw, re.DOTALL)
        if obj:
            data = json.loads(obj.group(0))
        else:
            raise ValueError(f"Could not extract JSON from LLM response: {raw[:300]!r}")

    return EpisodeSummary(
        episode_id=episode_id,
        episode_title=episode_title,
        topics=data.get("topics", []),
        concepts=data.get("concepts", []),
        description=data.get("description", "")
    )
