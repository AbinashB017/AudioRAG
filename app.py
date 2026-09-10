"""
app.py -- Phase 2: Streamlit Citation-to-Audio UI

Single-page Streamlit app that wraps the Phase 1 pipeline (retrieve +
generate_answer) with a chat interface and clickable audio citations.

Design decisions:
- Reuses retrieve() and generate_answer() verbatim -- no logic duplicated.
- st.session_state holds the full conversation history (same structure the
  CLI uses) and the last answer's citations so citation buttons survive reruns.
- Citations are rendered as st.button() calls; clicking one triggers
  st.audio() with start_time=start_sec so Streamlit seeks the player directly.
- Audio files are looked up by episode_id using a simple scan of cfg.audio_dir.
- The index is loaded once via @st.cache_resource so embeddings are not
  re-loaded on every user interaction.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fermi.config import load_config
from fermi.generate import Answer, format_timestamp, generate_answer
from fermi.index import get_or_create_collection
from fermi.retrieve import retrieve

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fermi Podcast Companion",
    page_icon="🎙️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached resources -- loaded once per server process
# ---------------------------------------------------------------------------

@st.cache_resource
def load_index():
    cfg = load_config()
    collection = get_or_create_collection(cfg.chroma_dir, cfg.embed_model)
    return cfg, collection


@st.cache_resource
def build_audio_map() -> dict[str, Path]:
    """Map episode_id -> audio file path by scanning podcasts/ relative to this script."""
    # Resolve relative to this file so it works both locally and on Streamlit Cloud
    audio_dir = Path(__file__).parent / "podcasts"
    mapping = {}
    from fermi.transcribe import episode_id_from_path
    for mp3 in audio_dir.glob("*.mp3"):
        mapping[episode_id_from_path(mp3)] = mp3
    return mapping


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []          # [{role, content}, ...]
    if "messages" not in st.session_state:
        st.session_state.messages = []         # display messages for st.chat_message
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []   # [Citation, ...] from most recent answer
    if "playing" not in st.session_state:
        st.session_state.playing = None        # (episode_id, start_sec) or None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    init_state()

    cfg, collection = load_index()
    audio_map = build_audio_map()

    # Count distinct episodes in the index
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    n_episodes = len({m["episode_id"] for m in all_meta}) if all_meta else 0

    st.title("🎙️ Fermi Podcast Companion")
    st.caption(
        f"Ask anything about the Great Papers podcast series. "
        f"Index: {collection.count()} chunks across {n_episodes} episodes."
    )

    # ---- Replay a citation if one was clicked --------------------------------
    if st.session_state.playing is not None:
        ep_id, start_sec = st.session_state.playing
        audio_path = audio_map.get(ep_id)
        if audio_path and audio_path.exists():
            st.audio(str(audio_path), start_time=int(start_sec))
        else:
            st.warning(f"Audio file for {ep_id} not found in {cfg.audio_dir}")

    # ---- Render conversation history ----------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Citation buttons (shown below the last assistant message) ----------
    if st.session_state.last_citations:
        st.markdown("---")
        st.markdown("**📎 Sources**")
        for i, citation in enumerate(st.session_state.last_citations):
            # Episode-level citations (recommendations) — no audio seek button
            if getattr(citation, "is_episode_level", False):
                with st.container(border=True):
                    col_info, col_btn = st.columns([4, 1])
                    with col_info:
                        st.markdown(f"**[{citation.ref}]** 📚 {citation.episode_title}")
                        st.caption("Full episode reference")
                    with col_btn:
                        if st.button("▶ Play", key=f"cite_{i}_{citation.chunk_id}"):
                            # Start playing from the beginning of the episode
                            st.session_state.playing = (citation.episode_id, 0.0)
                            st.rerun()
            else:
                ts_start = format_timestamp(citation.start_sec)
                ts_end   = format_timestamp(citation.end_sec)
                with st.container(border=True):
                    col_info, col_btn = st.columns([4, 1])
                    with col_info:
                        st.markdown(f"**[{citation.ref}]** {citation.episode_title}")
                        st.caption(f"⏱ {ts_start} → {ts_end}")
                        if citation.excerpt:
                            st.markdown(f"*\"{citation.excerpt}\"*")
                    with col_btn:
                        if st.button("▶ Play", key=f"cite_{i}_{citation.chunk_id}"):
                            st.session_state.playing = (citation.episode_id, citation.start_sec)
                            st.rerun()

    # ---- Chat input ---------------------------------------------------------
    if prompt := st.chat_input("Ask about the podcast episodes…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating…"):
                chunks = retrieve(prompt, collection, top_k=cfg.top_k)
                answer: Answer = generate_answer(
                    prompt, chunks, st.session_state.history, cfg
                )

            if answer.refused:
                st.warning("The podcast episodes don't appear to cover this topic.")

            st.markdown(answer.text)

        # Persist to history and display state
        st.session_state.messages.append({"role": "assistant", "content": answer.text})
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": answer.text})
        st.session_state.last_citations = answer.citations
        st.session_state.playing = None   # clear any previous audio player

        st.rerun()


if __name__ == "__main__" or True:
    main()
