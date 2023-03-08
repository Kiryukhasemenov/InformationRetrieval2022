# How to run the code

## Installation guide
there are two necessary elements needed to be downloaded: 
1. list of Czech stopwords, already available [here](stopwords_cs.txt);
2. language models of Morphodita, available [here](https://ufal.mff.cuni.cz/morphodita#language_models). We need:
    a. Czech MorfFlex+PDT for Czech to be put to `./morpho/` directory;
    b. English Morphium+WSJ to be put to the same place.
    
## Running the experiments

The code is run from the terminal by the command of the format:
```./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res -param value```

To run the ablations, use the parameters: 
1. `-preprocessing`: text preprocessing mode: out of {default, case_insensitive} values
2. `-tokenization`: tokenization mode: out of {default, nltk, morphodita, clear_stopwords, nltk_clear_punct, nltk_clear_punct_case_insensitive} values
3. `-idf`: IDF formula: out of {default, none, prob} values
4. `-tf`: TF formula: out of {default, boolean, augmented} values
5. `-query_construction`: Query Construction mode: out of {default, enhanced, enhanced_2} values.

The detailed description of the values is represented in the [report](report.pdf).