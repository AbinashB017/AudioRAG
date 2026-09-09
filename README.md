# Fermi Podcast Companion

The Fermi Podcast Companion is an interactive, grounded learning assistant that allows listeners to search, query, and trace insights across the **Great Papers** podcast series. It answers complex questions and provides citations that let you click and instantly listen to the exact audio timestamp where the topic was discussed.

## 🚀 Quickstart

### Prerequisites
1. **Python 3.10+**
2. **FFmpeg**: Required for audio duration extraction.
   - Install and ensure `ffprobe` is available, or point to it in your `.env`.
3. **OpenRouter API Key**: Get one at [openrouter.ai](https://openrouter.ai).

### Setup
```powershell
# 1. Clone the repo and install dependencies
git clone https://github.com/fermi-podcast-companion/fermi
cd fermi
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY and FFMPEG_BIN paths
```

### One-Command Full Run
To build the complete index from raw audio and launch the Streamlit app, simply run:
```powershell
.\start.ps1
```
This script handles the full pipeline:
1. **Transcription & Indexing**: Converts `.mp3` files in `podcasts/` into chunks and embeds them in Chroma.
2. **Episode Summaries**: Generates high-level summaries for episode recommendations.
3. **UI**: Launches the interactive Streamlit chat interface.

---

## 🧪 Evaluation Suite

This system was built with an evidence-driven approach, evaluated against a rigorous suite of test cases to guarantee perfect factual grounding, prevent cross-episode collapse, and defend against hallucinations.

You can easily re-run the evaluations. The results are logged in `eval/results/`.

**To run the full 20-case baseline** (validating all 16 episodes, cross-episode spanning, and recommendations):
```powershell
python run_eval.py
```

*(Note: The original 14-case suite is perfectly preserved as the first 14 items inside `eval/cases.json`. Running the command above guarantees you see both the original cases and the new scale-up cases.)*

---

## 🎬 Demo Recording Guide

If you are recording a walkthrough of the product, here is the ideal path to showcase the core capabilities:

1. **Factual Grounding & Chunk-Level Citation:**
   - **Ask:** *"What is Shannon entropy and what does the formula actually compute?"*
   - **Highlight:** The system will answer accurately and provide inline citations `[1]`. Click the button below the answer to instantly seek the audio to the exact moment the speaker explains the concept.
2. **Episode-Level Recommendations:**
   - **Ask:** *"Which episode should I listen to if I want to learn about biology?"*
   - **Highlight:** The system will recommend episodes 3, 8, and 13. Note how the citations cleanly render as static, non-interactive `📚 From: ...` labels instead of broken audio-seek buttons.
3. **Far-Apart Cross-Episode Comparison:**
   - **Ask:** *"Compare Watson and Crick's approach to discovering the structure of DNA with Mendeleev's approach to discovering the periodic table. How did both rely on finding patterns?"*
   - **Highlight:** The system successfully retrieves evidence from Ep 03 and Ep 15, integrates them seamlessly, and cites both appropriately.
4. **Adversarial Refusal (Zero Hallucination):**
   - **Ask:** *"Which episode covers the discovery of the Higgs Boson?"*
   - **Highlight:** The system cleanly refuses the question, acknowledging that no episodes cover this topic, and produces zero misleading citations.
