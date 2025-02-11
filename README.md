# AI Text Generator with Analysis

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Overview

AI Text Generator is a sophisticated tool that leverages artificial intelligence to generate, analyze, and process text content. The application combines multiple AI models to provide comprehensive text analysis including sentiment detection, bias checking, and automatic summarization.

## Features

- **Advanced Text Generation**: Utilizes state-of-the-art language models
- **Intelligent Summarization**: Automatically creates concise summaries
- **Sentiment Analysis**: Determines emotional tone of content
- **Bias Detection**: Identifies potential biases in generated text
- **User-Friendly Interface**: Simple yet powerful Gradio-based UI

## Key Components
1. Text Generation:
○ Utilizes the GPT-Neo model (EleutherAI/gpt-neo-125M) to generate
detailed explanations or descriptions based on the user’s input prompt.
2. Text Summarization:
○ Uses the TextRank algorithm from the sumy library to summarize the
generated content into a more concise form.
3. Bias Detection:
○ The Detoxify model is used to check for toxic language in the generated text.
○ It also flags content containing common bias-related terms (e.g., gender,
race).
4. Sentiment Analysis:
○ A pre-trained DistilBERT model fine-tuned for sentiment analysis is
employed to classify the sentiment of the generated content as "positive,"
"negative," or "neutral."

## How to Use
1. Text Input:
○ Enter a topic or prompt in the provided input field (e.g., "Explain climate
change").
2. Generate Button:
○ Click the "Generate" button to initiate the text generation, summarization, bias
detection, and sentiment analysis.
3. Outputs:
○ The generated content is displayed in the "Generated Content" box.
○ The summarized content is shown in the "Summarized Content" box.
○ Any detected bias or toxicity is flagged in the "Bias Check" box.
○ The sentiment of the content (positive, negative, or neutral) is displayed in the
"Sentiment" box

## Installation

### Prerequisites

```bash
Python 3.8+
pip (Python package installer)
```

### Dependencies

Install required packages:

```bash
pip install torch transformers gradio detoxify sumy
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-text-generator.git
cd ai-text-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Basic Usage

1. Enter your prompt in the text input field
2. Click "Generate"
3. View results in the following sections:
   - Generated Text
   - Summary
   - Sentiment Analysis
   - Bias Check

### Example

```python
from text_generator import AITextGenerator

generator = AITextGenerator()
text = generator.generate("Explain artificial intelligence")
summary = generator.summarize(text)
sentiment = generator.analyze_sentiment(text)
```


## Code components Documentation

### Text Generation

```python
generate_text(prompt: str, max_length: int = 200) -> str
```

Generates text based on the input prompt.

**Parameters:**
- prompt (str): Input text to generate from
- max_length (int): Maximum length of generated text

**Returns:**
- str: Generated text

### Summarization

```python
summarize_text(text: str, sentences_count: int = 3) -> str
```

Creates a summary of the input text.

**Parameters:**
- text (str): Text to summarize
- sentences_count (int): Number of sentences in summary

**Returns:**
- str: Summarized text

### Sentiment Analysis

```python
analyze_sentiment(text: str) -> dict
```

Analyzes the sentiment of the input text.

**Parameters:**
- text (str): Text to analyze

**Returns:**
- dict: Sentiment analysis results
  - sentiment: str (positive/negative/neutral)
  - confidence: float (0-1)

## Configuration

### Model Settings

```yaml
# config.yaml
models:
  text_generation:
    model_name: "EleutherAI/gpt-neo-125M"
    temperature: 0.7
    max_length: 200
  
  sentiment_analysis:
    model_name: "distilbert-base-uncased-finetuned-sst-2-english"
    threshold: 0.5
```

