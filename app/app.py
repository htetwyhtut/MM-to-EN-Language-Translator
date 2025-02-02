# Import libraries
import torch
import utils
from flask import Flask, render_template, request

# Choose CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Initialize Additive Attention Model
model = utils.define_model()

# Best model is Additive Attention
save_path = f'../models/additiveAttention.pt'

#Load model
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

# Set the model to the correct device
model.to(device)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    # Home page
    if request.method == 'GET':
        return render_template ('index.html', prompt = '')
    
    # Page after user input
    if request.method == 'POST':
        # Get user input from form
        prompt = request.form.get('query')

        try:
            generation = utils.greedy_decode(model, prompt, max_len=50, device=device)
            generation = [token for token in generation if token != '<eos>']
            sentence = ' '.join(generation)
        except Exception as e:
            sentence = f"Error generating translation: {str(e)}"

        # Return the rendered HTML with the generated sentence
        return render_template('index.html', query = prompt, sentence = sentence)

if __name__ == '__main__':
    app.run(debug=True)