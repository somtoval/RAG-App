from rag_pipeline_with_memory import RagPipeline

class Prediction:
    def pipeline_runner(self):
        rag = RagPipeline()
        qa_chain = rag.run_pipeline("Labour_Function.pdf")
        return qa_chain

    def predict_query(self, query, qa_chain):
        result = qa_chain({"question":query})['answer']
        return result


test = Prediction()
qa_chain = test.pipeline_runner()

query = input("Enter your Query Here: ")
while query != "quit":
    print(test.predict_query(query, qa_chain))
    query = input("Enter your Query Here: ")