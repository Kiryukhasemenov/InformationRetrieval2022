# How to run the code

## Installation guide
there are two necessary elements needed to be downloaded: 
1. list of Czech stopwords, already available [here](stopwords_cs.txt);
2. language models of Morphodita, available [here](https://ufal.mff.cuni.cz/morphodita#language_models). We need:
    a. Czech MorfFlex+PDT for Czech to be put to `./morpho/` directory;
    b. English Morphium+WSJ to be put to the same place.
3. I could not run the code locally, thus all necessary code is shown [here](https://colab.research.google.com/drive/1-144PE0xm3MDLUCVpzFKzgvdUcughrZY?usp=sharing). 
    
## Running the experiments

The code is run from the terminal by the command of the format:
```./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res -param value```

To run the ablations, use the parameters: 
1. `-preprocessing`: text preprocessing mode: out of {default, case-sensitive} values;
2. `-tokenization`: tokenization mode: out of {default, nltk, morphodita, clear_stopwords} values;
3. `-weighting`: weighting model for indexation: out of {Tf, TF_IDF, BM25, PL2, Dirichlet_LM, LGD, Hiemstra} values;
4. `-query_construction`: Query Construction mode: out of {default, enhanced} values;
5. `-query_expansion`: Query Expansion mode: out of {default, Bo1} values.

The detailed description of the values is represented in the [report](report.pdf). 

The configurations of the runs 0, 1 and 2 are the following:

#### Run 0:
`python ./run.py -q IR_A2/topics-{train, test}_{cs, en}.xml -d IR_A2/documents_{cs, en}.lst -r run-0 -o run-0_{train, test}_{cs, en}.res -preprocessing case-sensitive`

#### Run 1:
`python ./run.py -q IR_A2/topics-{train, test}_{cs, en}.xml -d IR_A2/documents_{cs, en}.lst -r run-1 -o run-1_{train, test}_{cs, en}.res -weighting LGD`

#### Run 2:
`python ./run.py -q IR_A2/topics-{train, test}_{cs, en}.xml -d IR_A2/documents_{cs, en}.lst -r run-2 -o run-2_{train, test}_{cs, en}.res -weighting LGD -query_expansion Bo1`

