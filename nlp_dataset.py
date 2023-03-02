import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import glue_compute_metrics, glue_output_modes, glue_tasks_num_labels
from transformers import glue_processors, glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample

# Set the device to run the model on (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the languages you want to use for training
languages = ['en', 'fr', 'es', 'zh']

# Load the XNLI dataset and tokenizer for multiple languages
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
train_dataset = []
for lang in languages:
    processor = glue_processors["mnli"]
    train_examples = processor.get_train_examples('XNLI-MT-1.0/')
    train_features = glue_convert_examples_to_features(train_examples, tokenizer, max_length=128, task='xnli')
    train_dataset += train_features
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the BERT model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model for a fixed number of epochs

