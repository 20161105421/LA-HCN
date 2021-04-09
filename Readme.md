# LA-HCN
The official Tensorflow implementation of LA-HCN: Label-based Attention for HierarchicalMulti-label Text Classification Neural Network. [LA-HCN](https://arxiv.org/abs/2009.10938).

The implementation of data preparsion part refers to the [CODE](https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification).
*** 

### Package Version

     numpy==1.18.2
     scikit-learn==0.23.2
     scipy==1.4.1
     tensorflow-estimator==2.2.0
     tensorflow-gpu==2.2.0
     tflearn==0.5.0
    
***
## Data Preparation

The experimental datasets used in paper are available at https://drive.google.com/file/d/1g3Ln0fzCIqUO7A1GTZpKntXSUaiJdqgZ/view?usp=sharing

### Data Format

The data format used in this implementation refers to the [HARNN](https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification) project:
```json
    {
        "id": 46, 
        "label_combine": [63, 72, 130, 2, 21, 30], 
        "label_local": [[2], [14, 23], [10, 19], [0]], 
        "content": ["forge", "christendom", "grand", "narrative", "history", "emergence", "europe", "following", "collapse", "roman", "empire", "approach", "first", "millennium", "christian", "europe", "seem", "likely", "candidate", "future", "greatness", "weak", "fractured", "hemmed", "hostile", "nation", "saw", "future", "beyond", "widely", "anticipated", "second", "coming", "christ", "world", "end", "people", "western", "europe", "suddenly", "found", "choice", "begin", "heroic", "task", "building", "jerusalem", "earth", "forge", "christendom", "tom", "holland", "masterfully", "describes"]
    }
```
- each data sample contains four parts:
    1. **id**: string. Data sample id.
    2. **label_local**: A list of sub-lists which contains Level-based labels. The lablels in each sub-list represents the labels in the corresponding label-level.
       
       (In this example, the labels of the input sample is "2", "14,23", "10,19" and "0" at the label level 1, 2, 3, 4 respectively.)
    3. **label_combine**: A list of labels which provides the overall labels.
       
       (In this example, the overall labels of the input sample is
            
            "2, 21, 30, 63, 72, 130" (which is associated with "2", "14,23", "10,19" and "0" respectively.)
    
       Sepecifically,
            
            "2"  = "2 +0",
            "21" = "14+7",
            "30" = "23+7",
            "63" = "10+7+46",
            "72" = "19+7+46",
            "130"= "0 +7+46+77"
       where the number of label types in label-level 1, 2, 3, 4 is "7, 46, 77, 16" respectively.
    3. **content**: The sample text content which is represented with splited words(necessary text pre-processing is applied).
    
***
## Excute

To run LA-HCN:

   ```
    python train_lahcn.py --dataname 'reuters_0' 
                          --training_data_file data/reuters/0/reuters_train_0.json 
                          --validation_data_file data/reuters/0/reuters_val_0.json 
                          --num_classes_list "4,55,42" 
                          --glove_file None
   ```
Important Hyper-arameter setting:

```
  --dataname                STR     prefix for storing current model    e.g. 'reuters_0'
  --training_data_file      STR     path to training data               e.g. 'data/reuters/0/reuters_train_0.json' 
  --validation_data_file    STR     path to validation data             e.g. 'data/reuters/0/reuters_val_0.json' 
  --num_classes_list        STR     Number of labels list               e.g. '4,55,42' 
  --glove_file GLOVE_FILE   STR     glove embeding file                 e.g. 'data/glove6b100dtxt/glove.6B.100d.txt' 
  --train_or_restore        STR     Train or Restore (default: Train)   e.g. 'Train'
  
  
  --learning_rate           FLOAT   Learning Rate (default: 0.001)      e.g. 0.001
  --batch_size BATCH_SIZE   FLOAT   Batch Size (default: 256)
  --num_epochs              FLOAT   Number of training epochs (default: 100)
  --pad_seq_len             INT     Padding Sequence length of data (default: 250)
  --embedding_dim           INT     Word emb dimension, if glove_file is provided, this value depends on glove emb. (default: 100)
  --lstm_hidden_size        INT     Hidden size for bi-lstm layer(default: 256)
  --attention_unit_size     INT     Attention unit size(default: 200)
  --fc_hidden_size          INT     Hidden size for fully connected layer (default: 512)
  --checkpoint_every        INT     Save model for every steps (default: 100)
  --num_checkpoints         INT     Number of checkpoints to store in all (default: 5)
  
  
  --evaluate_every          INT     Evaluate model on validation data after this many steps (default: 100)
  --top_num                 INT     Number of top K prediction on validation data (default: 5)
  --threshold               FLOAT   Threshold for prediction on validation data (default: 0.5)
```

*NOTE*. if glove_file is not available, a word2id dictionary will be generated.
***
## Output

- Output model is sotred in the folder: *runs/* 
***
## Evaluation

- An example of evaluation is provided in **test_lahcn.py**. To run it:
  
    ```
    python test_lahcn.py --dataname 'reuters_0' 
                         --test_data_file data/reuters/0/reuters_test_0.json
                         --num_classes_list "4,55,42" 
                         --glove_file None
    ```
  The result file can be found in folder *model/* 
***
## Citing

If you find LA-HCN is useful for your research, please consider citing the following paper:

    @misc{zhang2021lahcn,
      title={LA-HCN: Label-based Attention for Hierarchical Multi-label TextClassification Neural Network}, 
      author={Xinyi Zhang and Jiahao Xu and Charlie Soh and Lihui Chen},
      year={2021},
      eprint={2009.10938},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }

Please send any questions you might have about the codes and/or the algorithm to xinyi001@e.ntu.edu.sg.
