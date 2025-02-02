# A3: MM-to-EN-Language-Translator
Enjoy reading my A3 Assignment for NLP class.

## Author Info
Name: WIN MYINT@HTET WAI YAN HTUT (WILLIAM)
Student ID: st125326

## How to run the web app
1. Pull the github repository
2. Run
```sh
python app/app.py
```
3. Access the app using http://127.0.0.1:5000

## How to use website
1. Open a web browser and navigate to http://127.0.0.1:5000.
2. Enter a prompt and select a word limit.
2. Click "Generate Story" to see the result.

## Screenshot of my web app


## Task 1.1: Find relevant Dataset (1 points)
1. As a Myanmar Nationality, I choose ENGLISH - MYANMAR text pair as my dataset.
2. I obtained my dataset from https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/
3. The copyright holder of the translations of the ALT Parallel Corpus is the National Institute of Information and Communications Technology, Japan (NICT). 
4. The University of Computer Studies, Yangon, Myanmar helped NICT translate the English texts into the Myanmar texts.

## Task 1.2: Detailed process of dataset preparation (1 points)
1.  Initially the dataset contains english and multiple Asian language pairs. I firstly split it into English and Myanmar pair.
2. ENGLISH tokenization done using spacy tokenizer with the en_core_web_sm model.
3. MYANMAR tokenization done using custom tokenizer for Myanmar using the Viterbi algorithm for word segmentation. This is a sophisticated approach and well-suited for Myanmar, which requires word segmentation. (Full credit to Dr. Ye Kyaw Thu for the Viterbi algorithm implementation. [https://github.com/ye-kyaw-thu/myWord] )
4. Vocabulary buidling: Tokenized English and Myanmar text are used to build a vocabulary. The vocabulary is then used to convert text into numerical indices for further processing.













## Task 2.1: Detail the steps taken to preprocess the text data. (1 points)
1. Load the Dataset
2. Define the Tokenizer
- Use `torchtext.data.utils.get_tokenizer` to create a tokenizer.<br><br>
3. Tokenize the Text
- Define a function to tokenize each example in the dataset.
- Use the `map` method to apply the tokenization function to the entire dataset.<br><br>
4. Remove Unnecessary Columns
- Remove the original `'text'` column after tokenization.<br><br>
5. Build Vocabulary and Numericalize Data
- Build a vocabulary using `torchtext.vocab.build_vocab_from_iterator`.
  - Set a minimum frequency threshold (e.g., `min_freq=3`) to filter out rare tokens.
- Add special tokens (e.g., `'<unk>'` and `'<eos>'`) if they don't already exist in the vocabulary. <br><br>
6. Inspect the Results
- Verify that the tokenization was applied correctly by inspecting the tokenized dataset.

## Task 2.2: Describe the model architecture and the training process. (1 points)

Model Architecture
The model is an LSTM-based Language Model implemented in PyTorch. It consists of the following key components:
1. Embedding Layer
2. **LSTM Layer
3. Dropout Layer
4. Fully Connected Layer
5. Initialization

Training Process (Data Preparation and Training Loop)
1. Tokenization and Numericalization
2. Batching
3. Training Loop
- Initialization,Forward Pass, Loss Calculation, Backpropagation, Evaluation, Checkpointing

Metrics (Perplexity):
  - Used to evaluate the model's performance.
  - Defined as exp(loss).
  - Lower perplexity indicates better performance.

Training Output
During training, the following metrics are printed for each epoch:
- Train Perplexity: Perplexity on the training set.
- Valid Perplexity: Perplexity on the validation set.

## Task 3. Text Generation - Web Application Development (2 points)
Provide documentation on how the web application interfaces with the language model.

1. Frontend (HTML/CSS)
2. Backend (Flask)
3. User Workflow

Step 1: User Input
- The user visits the web application and sees a form with two fields:
- Text Prompt: The user enters a starting sentence or phrase (e.g., "Once upon a time").
- Max Word Limit: The user selects the maximum number of words to generate (options: 20, 50, 100).
- The user clicks the "Generate Story" button to submit the form.

Step 2: Backend Processing
- The Flask application (app.py) receives the form data via a POST request.
- The application extracts the following from the form:
- prompt: The text prompt entered by the user.
- seq_len: The maximum word limit selected by the user.
- The application calls the generate function from class_function.py with the following parameters:
- prompt: The user's text prompt.
- max_seq_len: The maximum word limit.
- temperature: Controls the randomness of the generated text (default: 1.0).
- model: The pre-trained LSTM language model.
- tokenizer: Converts text into tokens for the model.
- vocab: Maps tokens to indices and vice versa.
- device: Specifies whether to use CPU or GPU (automatically detected).
- seed: Ensures reproducibility (default: 0).

Step 3: Text Generation
- The generate function processes the input prompt:
- Tokenizes the prompt using the tokenizer.
- Converts tokens into indices using the vocab.
- The LSTM model generates text autoregressively:
- The model predicts the next word based on the input sequence.
- The process repeats until the maximum word limit is reached or an end-of-sequence token (<eos>) is generated.
- The generated tokens are converted back into text using the vocab.

Step 4: Display Results
- The generated text is returned to the Flask application.
- The application renders the index.html template with the generated text and displays it to the user.
