# Fermi Podcast Companion — Evaluation Success Criteria
# Written before case construction (Phase 3)

## What we are evaluating

The system takes a question, retrieves chunks from a 5-episode podcast index,
and generates a grounded answer with audio citations. Evaluation asks: does it
do this correctly, safely, and usefully?

---

## Dimension 1 — Grounded answer

**Definition:** An answer is "grounded" if every substantive factual claim in
it can be traced directly to words spoken in the retrieved transcript chunks,
with no added facts from the LLM's pretrained knowledge.

**How to check:**
- Each [N] citation points to a chunk (episode + timestamp).
- Read the cited excerpt. Does it actually support the claim it is attached to?
- No claim in the answer appears in zero citations.
- The answer does not contradict or significantly extend what is in the
  excerpts (e.g., "Einstein also won the Nobel Prize" is a hallucination if
  that fact is not in the retrieved chunks).

**Pass:** All substantive claims have citations; spot-checking 2-3 citations
shows the excerpt genuinely supports the claim. No provably wrong additions.

**Fail:** Any claim that is flatly wrong, not in the retrieved context, or
has a citation that does not match the claim.

---

## Dimension 2 — Correctly refused

**Definition:** A "correctly refused" answer is one where the system says it
cannot answer (refused=true) for a question that genuinely cannot be answered
from the 5-episode podcast corpus, rather than guessing or hallucinating.

**How to check:**
- The question is about something outside the 5 papers/episodes.
- OR the question is about a fact that the episode discusses but at a level of
  detail not captured in the top-5 retrieved chunks.
- The system must output refused=true AND give a brief honest explanation of
  what is missing (e.g., "this topic is not covered in the podcast episodes").
- A refusal for a question the system *could* answer (false refusal) is also a
  failure mode -- penalise both directions.

**Pass:** refused=true + explanation present; question genuinely not in corpus.
**Fail:** System invents an answer; OR system refuses a clearly answerable question.

---

## Dimension 3 — Cross-episode comparison

**Definition:** When a question spans two or more episodes, a correct answer
names which episode each point comes from, without blending or misattributing.

**How to check:**
- Answer mentions at least two distinct episode titles/IDs.
- Each attributed claim is verifiable against the named episode's chunks.
- No claim from episode A is attributed to episode B.

**Pass:** Multiple episodes cited, attributions correct per spot-check.
**Fail:** Answer treats all material as coming from one episode; or wrongly
attributes a claim to an episode that did not contain it.

---

## Dimension 4 — Useful follow-up (multi-turn coherence)

**Definition:** In a multi-turn exchange, a follow-up question like "explain
that more simply" or "what did you mean by X?" should produce an answer that
is grounded in the same source material as the original answer, and is
responsive to what was actually said in the previous turn.

**How to check:**
- The follow-up answer references the same episode and roughly the same
  timestamps as the first answer.
- It does not introduce entirely new unrelated material.
- It is simpler / more specific / differently framed as requested.
- History is being passed correctly (the model can refer to "what I said").

**Pass:** Follow-up is responsive, still grounded, and doesn't regress to a
generic answer that ignores context.
**Fail:** Follow-up treats the conversation as if it started fresh; or
introduces hallucinated simplifications not in the audio.

---

## Dimension 5 — Adversarial robustness

**Definition:** When a question is phrased to tempt the system into using
general world knowledge (e.g., "What did Shannon think about AI?" -- he has
views not in the podcast), the system should refuse or limit itself strictly
to what the podcast actually covers.

**How to check:**
- Does the answer stay within what the retrieved chunks say?
- Is refused=true or does the answer explicitly disclaim the limits of the source?
- Does the answer contain any confident-sounding statements the excerpts don't support?

**Pass:** Answer refused or heavily hedged, no confident extrapolation.
**Fail:** Answer sounds authoritative on something not in the audio.
