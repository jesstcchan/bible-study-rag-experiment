from llm_rag.llm import generate_answer


answer = generate_answer(
    question=(
        "Why did the elders ask Samuel for a king, and why was this request "
        "presented as a problem?"
    ),
    passage_reference="1 Samuel 8:4–9",
    passage_text=(
        "Temporary test text. Replace this with the approved World English "
        "Bible passage."
    ),
    source_context="NONE",
)

print(answer)