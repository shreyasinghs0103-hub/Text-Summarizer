import nltk
from flask import Flask, render_template, request
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Automatically download NLTK tokenizers on startup if missing
for resource in ['punkt', 'punkt_tab']:
  try:
    nltk.data.find(f'tokenizers/{resource}')
  except LookupError:
    nltk.download(resource, quiet=True)

app = Flask(__name__)


def summarize_text(text, num_sentences=3):
  """Generates a local summary using Sumy (LSA Algorithm)."""
  if not text or len(text.strip()) == 0:
    return ''

  parser = PlaintextParser.from_string(text, Tokenizer('english'))
  stemmer = Stemmer('english')
  summarizer = LsaSummarizer(stemmer)

  # Extract key sentences
  sentences = summarizer(parser.document, num_sentences)
  return ' '.join([str(s) for s in sentences])


@app.route('/', methods=['GET', 'POST'])
def index():
  summary = ''
  original_text = ''
  error = None

  if request.method == 'POST':
    original_text = request.form.get('text', '')
    word_count = len(original_text.split())

    if word_count < 10:
      error = 'Please enter at least 10 words to generate a summary.'
    else:
      try:
        summary = summarize_text(original_text, num_sentences=3)
      except Exception as e:
        error = f'Summarization error: {str(e)}'

  return render_template(
      'index.html', text=original_text, summary=summary, error=error
  )


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
