# End-to-End-Medical-Chatbot

# How to run?
### STEPS:

Clone the repository

``` bash
https://github.com/yadavpravin1452/End-to-End-Medical-Chatbot
```

### STEP 01 - Create a conda environment after opeaning the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02 - install the requirement

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone and Openai Credentail as follow:

```ini
PINECONE_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
OPENAI_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

```bash
# run the following command  to store  embeddings to Pinecone
python store_index.py
```

```bash
#Finally run the following command
python app.py
```

```bash
open up localhost:
```


### Techstack Used:
- Python
- Langchain
- Flask
- HuggingFace -> Llama
- Pinecone


