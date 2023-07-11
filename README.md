# Discontinuous Grammar as a Foreign Language
This repository includes the code of the sequence-to-sequence model for discontinuous constituent parsing described in paper [Discontinuous Grammar as a Foreign Language](https://arxiv.org/abs/2110.10431). In particular, it uses the in-order+SWAP linearization to deal with discontinuities and yields 95.47 F1 on the English Discontinuous Penn Treebank (DPTB). This implementation is based on the system by [Fernandez Astudillo et al. (2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) and reuses part of its [code](https://github.com/IBM/transition-amr-parser/tree/stack-transformer).


## Requirements
This implementation was tested on Python 3.6.9, PyTorch 1.1.0 and CUDA 9.0.176. Please run the following command to proceed with the installation:
``` 
    cd Disco-Seq2seq-Parser
    pip install -r requirements.txt
```

For the evaluation, script DISCODOP must be also installed following steps described in https://github.com/andreasvc/disco-dop.

## Data
To get shift-reduce linearizations from discontinuous constituent treebanks (for instance, the DPTB), please include train, dev and test splits in ``discbracket`` format in the ``disco_data`` folder and name them as ``train.discbracket``, ``dev.discbracket`` and ``test.discbracket``. Then use the following script:
``` 
    ./linearization/generate.sh DPTB
```

## Experiments
To train a model for the DPTB treebank, just execute the following script:
``` 
   ./scripts/stack-transformer/con_experiment.sh configs/ptb_roberta.large.sh
```

To test the trained model on the test split, please run the following command:
``` 
    ./scripts/stack-transformer/con_test-test.sh configs/test_roberta_large.sh DATA/dep-parsing/models/DPTB_RoBERTa-large_stnp6x6-seed44/checkpoint_top3-average.pt DATA/dep-parsing/models/DPTB_RoBERTa-large_stnp6x6-seed44/epoch-tests-test/dec-checkpoint-top3-average	
``` 


## Citation
```
@article{FERNANDEZGONZALEZ202343,
title = {Discontinuous grammar as a foreign language},
journal = {Neurocomputing},
volume = {524},
pages = {43-58},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2022.12.045},
url = {https://www.sciencedirect.com/science/article/pii/S092523122201551X},
author = {Daniel Fernández-González and Carlos Gómez-Rodríguez},
keywords = {Natural language processing, Computational linguistics, Parsing, Discontinuous constituent parsing, Neural network, Deep learning, Sequence-to-sequence model},
abstract = {In order to achieve deep natural language understanding, syntactic constituent parsing is a vital step, highly demanded by many artificial intelligence systems to process both text and speech. One of the most recent proposals is the use of standard sequence-to-sequence models to perform constituent parsing as a machine translation task, instead of applying task-specific parsers. While they show a competitive performance, these text-to-parse transducers are still lagging behind classic techniques in terms of accuracy, coverage and speed. To close the gap, we here extend the framework of sequence-to-sequence models for constituent parsing, not only by providing a more powerful neural architecture for improving their performance, but also by enlarging their coverage to handle the most complex syntactic phenomena: discontinuous structures. To that end, we design several novel linearizations that can fully produce discontinuities and, for the first time, we test a sequence-to-sequence model on the main discontinuous benchmarks, obtaining competitive results on par with task-specific discontinuous constituent parsers and achieving state-of-the-art scores on the (discontinuous) English Penn Treebank.}
}
```

## Acknowledgments

We acknowledge the European Research Council (ERC), which has funded this research under the European Union’s Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150) and the Horizon Europe research and innovation programme (SALSA, grant agreement No 101100615), ERDF/MICINN-AEI (SCANNER-UDC, PID2020-113230RB-C21), Xunta de Galicia (ED431C 2020/11), and Centro de Investigación de Galicia "CITIC", funded by Xunta de Galicia and the European Union (ERDF - Galicia 2014-2020 Program), by grant ED431G 2019/01.
                                                                                                                                   
