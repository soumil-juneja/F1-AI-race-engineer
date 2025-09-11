import chromadb

client = chromadb.PersistentClient(path = "./f1_db")

collection = client.get_collection(name="f1_knowledge")

def ask_commentator(question):
    results = collection.query(
        query_texts = [question],
        n_results = 1
    )

    ans = results['documents'][0][0]

    return ans

if __name__ == "__main__":
    print("--- F1 AI Commentator v1.0 ---")
    print("Ask a question about any F1 term or type 'exit' to quit.")

    while(True):
        q = input("\nYour question: ")

        if q.lower() == 'exit':
            print("Exiting the F1 AI Commentator. Goodbye!")
            break

        answer = ask_commentator(q)
        print("\nF1 Commentator:")
        print(answer)

