import pandas as pd
import chromadb

def ingest():
    df = pd.read_csv('f1_glossary.csv')
    print("Successfully loaded the data")
    

    client = chromadb.PersistentClient(path = "./f1_db")

    collection_name = "f1_knowledge"
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Successfully connected to ChromaDB and using collection: '{collection_name}'")

    documents = [f"{row['Term']}: {row['Definition']}" for index, row in df.iterrows()]
    ids = [f"term_{index}" for index, row in df.iterrows()]
    print(f"Prepared {len(documents)} documents")

    collection.add(
        documents=documents,
        ids=ids
    )

    count = collection.count()
    print(f"Successfully ingested {count} documents into the collection '{collection_name}'")

if __name__ == "__main__":
    ingest()