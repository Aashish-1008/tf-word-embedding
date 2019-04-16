# tf-word-embedding
This repo introduces word embeddings. It contains complete code to train word embeddings from scratch on a small dataset.

##Representing text as numbers
Machine learning models take vectors (arrays of numbers) as input. 
When working with text, the first thing we must do come up with a strategy to 
convert strings to numbers (or to "vectorize" the text) before feeding it 
to the model. In this section, we will look at three strategies for doing so.

### 1.  One-hot encodings 
This approach is inefficient. A one-hot encoded vector is sparse (meaning, most indicices are zero). 
Imagine we have 10,000 words in the vocabulary. 
To one-hot encode each word, we would create a vector where 99.99% of the 
elements are zero.
### 2. Encode each word with a unique number
### 3. Word embeddings
```
# on ubuntu
sudo apt install python3 

# on mac 
brew install python3 

# install ludwig
pip install ludwig
python -m spacy download en     
```

In this repo, I will try to build a simple text-classification using ludwig


## Using local python
You can run the code locally

```
JOB_DIR=jobDir
TRAIN_FILE=./data/train/*
EVAL_FILE=./data/eval/*
TRAIN_STEPS=2000

cd  tf-ludwig-google-cloud-ml-engine/

python3.6 -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS.
'''