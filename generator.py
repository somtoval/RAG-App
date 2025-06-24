from rag_pipeline import RagPipeline

class Prediction:
    def pipeline_runner(self):
        rag = RagPipeline()
        qa_chain = rag.run_pipeline("Labour_Function.pdf")
        return qa_chain

    def predict_query(self, query, qa_chain):
        result = qa_chain({"query": query})
        return result["result"]


test = Prediction()
qa_chain = test.pipeline_runner()
print(test.predict_query("Who is to work in the Toilet?", qa_chain))