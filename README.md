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

