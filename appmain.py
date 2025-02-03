import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from detoxify import Detoxify
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_length: int = 200
    min_length: int = 50
    temperature: float = 0.7  # Slightly increased for more creative outputs
    top_k: int = 50  # Increased for more diverse outputs
    top_p: float = 0.9
    repetition_penalty: float = 1.3  # Increased to better prevent repetition
    no_repeat_ngram_size: int = 3
    do_sample: bool = True


class TextGenerationApp:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M"):
        """Initialize the text generation application with specified model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            self.detoxifier = Detoxify("original")
            self.config = GenerationConfig()
            logger.info(f"Successfully initialized {model_name}")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def generate(self, prompt: str) -> Optional[str]:
        """Generate text based on the input prompt with error handling."""
        try:
            if not prompt.strip():
                return "Error: Empty prompt provided"

            full_prompt = f"Explain {prompt} in a simple, structured way with key details:\n\n"

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
                self.model = self.model.to("cuda")

            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text.replace(full_prompt, '').strip()
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Error: {str(e)}"

    def summarize(self, text: str, sentence_count: int = 3) -> str:
        """Summarize the input text with improved error handling."""
        try:
            if not text.strip():
                return "Error: Empty text provided"

            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, sentence_count)
            return " ".join(str(sentence) for sentence in summary)
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return f"Error: {str(e)}"

    def detect_bias(self, text: str) -> str:
        """Detect bias and toxicity in text with enhanced detection."""
        try:
            if not text.strip():
                return "Error: Empty text provided"

            toxic_score = self.detoxifier.predict(text)

            # Enhanced bias terms list
            bias_terms = {
                "demographic": ["he/she", "man/woman", "boy/girl"],
                "socioeconomic": ["rich/poor", "wealthy/poor", "privileged/unprivileged"],
                "racial": ["race", "ethnic", "racial"],
                "gender": ["gender", "masculine", "feminine"],
                "ability": ["disabled", "handicapped", "impaired"],
                "economic": ["poverty", "low-income", "welfare"]
            }

            # Check for bias terms in each category
            found_biases = []
            for category, terms in bias_terms.items():
                if any(term in text.lower() for term in terms):
                    found_biases.append(category)

            if toxic_score["toxicity"] > 0.3 or found_biases:
                return f"⚠️ Potential bias detected: {', '.join(found_biases)}" if found_biases else "⚠️ Potential toxicity detected"
            return "✅ Neutral content"
        except Exception as e:
            logger.error(f"Error in bias detection: {str(e)}")
            return f"Error: {str(e)}"

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment with confidence threshold."""
        try:
            if not text.strip():
                return "Error: Empty text provided"

            sentiment = self.sentiment_analyzer(text)[0]
            confidence = sentiment['score']

            if confidence < 0.6:
                return f"{sentiment['label']} (Low confidence: {confidence:.2f})"
            return f"{sentiment['label']} ({confidence:.2f})"
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return f"Error: {str(e)}"

    def run(self, prompt: str) -> Tuple[str, str, str, str]:
        """Run the complete analysis pipeline with proper error handling."""
        try:
            generated_text = self.generate(prompt)
            if generated_text.startswith("Error"):
                return generated_text, "Error: Generation failed", "Error: Generation failed", "Error: Generation failed"

            summarized_text = self.summarize(generated_text)
            bias_check = self.detect_bias(generated_text)
            sentiment = self.analyze_sentiment(generated_text)

            return generated_text, summarized_text, bias_check, sentiment
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            return (f"Error: {str(e)}", "Error: Pipeline failed",
                    "Error: Pipeline failed", "Error: Pipeline failed")


# Gradio Interface with improved error handling
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# AI Text Generator with Summarization & Analysis")

        generator = TextGenerationApp()

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Enter Topic",
                placeholder="Type a topic here...",
                lines=1
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            generated_output = gr.Textbox(
                label="Generated Content",
                interactive=False,
                lines=6
            )
            summarized_output = gr.Textbox(
                label="Summarized Content",
                interactive=False,
                lines=4
            )

        with gr.Row():
            bias_output = gr.Textbox(
                label="Bias Check",
                interactive=False,
                lines=2
            )
            sentiment_output = gr.Textbox(
                label="Sentiment",
                interactive=False,
                lines=2
            )

        def on_generate_click(prompt):
            try:
                return generator.run(prompt)
            except Exception as e:
                logger.error(f"Error in generate click handler: {str(e)}")
                return ("Error: Generation failed", "Error: Generation failed",
                        "Error: Generation failed", "Error: Generation failed")

        generate_btn.click(
            on_generate_click,
            inputs=[prompt_input],
            outputs=[generated_output, summarized_output, bias_output, sentiment_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)