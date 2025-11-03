#!/bin/bash

echo "🔄 Installing required HuggingFace model for sentence-transformers..."

# Run Python command to pre-download and cache the model
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model pre-downloaded and cached.')
"
