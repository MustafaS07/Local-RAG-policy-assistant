# evaluation/evaluate.py

from app.rag_pipeline import create_qa_chain

def evaluate_sample():
    qa_chain = create_qa_chain()

    test_questions = [
        "How many annual leaves do employees get?",
        "How many sick leaves are allowed?"
    ]

    for q in test_questions:
        result = qa_chain.invoke(q)
        print("Question:", q)
        print("Answer:", result["result"])
        print("-" * 50)


if __name__ == "__main__":
    evaluate_sample()
