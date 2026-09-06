"""Run one matched baseline/RAG answer test before connecting oTree."""

from __future__ import annotations

import argparse

from llm_rag.pipeline import answer_question


PASSAGE_REFERENCE = "1 Samuel 8:4–9"
PASSAGE_TEXT = """
4 Then all the elders of Israel gathered themselves together and came to
Samuel to Ramah.

5 They said to him, “Behold, you are old, and your sons don’t walk in your
ways. Now make us a king to judge us like all the nations.”

6 But the thing displeased Samuel when they said, “Give us a king to judge
us.” Samuel prayed to the LORD.

7 The LORD said to Samuel, “Listen to the voice of the people in all that they
tell you; for they have not rejected you, but they have rejected me as the king
over them.

8 According to all the works which they have done since the day that I brought
them up out of Egypt even to this day, in that they have forsaken me and served
other gods, so they also do to you.

9 Now therefore, listen to their voice. However, you shall protest solemnly to
them, and shall show them the way of the king who will reign over them.”
""".strip()
QUESTION = (
    "Why did the elders ask Samuel for a king, and why was this request "
    "presented as a problem?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("baseline", "rag", "both"),
        default="both",
    )
    return parser.parse_args()


def print_result(condition: str) -> None:
    result = answer_question(
        condition=condition,
        question=QUESTION,
        passage_reference=PASSAGE_REFERENCE,
        passage_text=PASSAGE_TEXT,
    )

    print("=" * 68)
    print(f"CONDITION: {result.condition}")
    print(f"LATENCY_MS: {result.latency_ms}")
    print(f"RETRIEVED_SOURCES: {len(result.sources)}")
    for source in result.sources:
        print(
            f"  [S{source.rank}] score={source.score:.4f} "
            f"chunk={source.chunk_id} source={source.source_id} "
            f"reference={source.biblical_reference}"
        )
        print(f"       title={source.title}")
        print(f"       url={source.url}")
    print("ANSWER:")
    print(result.answer)

    if condition == "baseline" and result.sources:
        raise RuntimeError("Baseline unexpectedly received retrieved sources.")
    if condition == "rag" and not result.sources:
        raise RuntimeError("RAG returned no retrieved sources.")


def main() -> None:
    args = parse_args()
    conditions = ("baseline", "rag") if args.condition == "both" else (args.condition,)
    for condition in conditions:
        print_result(condition)
    print("=" * 68)
    print("PIPELINE TEST COMPLETED")


if __name__ == "__main__":
    main()
