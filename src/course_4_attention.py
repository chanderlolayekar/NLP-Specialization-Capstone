from transformers import pipeline

class AttentionModels:
    """
    Course 4: Attention Models
    Covers: Encoder-Decoder, Causal/Self-Attention, Transformers, LLMs
    """
    def __init__(self):
        # We use Hugging Face Transformers here for the industry-standard implementation
        # In the Coursera labs, you will build Self-Attention and T5/BERT models from scratch!
        self.translator = pipeline("translation_en_to_fr")
        self.summarizer = pipeline("summarization")
        self.qa_model = pipeline("question-answering")

    def machine_translate(self, text):
        """Translates complete sentences using Encoder-Decoder architectures."""
        return self.translator(text)

    def summarize_text(self, text):
        """Generates abstractive text summaries using self-attention mechanisms."""
        return self.summarizer(text, max_length=50, min_length=25, do_sample=False)

    def answer_question(self, question, context):
        """Performs reading comprehension and question-answering based on a provided text block."""
        return self.qa_model(question=question, context=context)
