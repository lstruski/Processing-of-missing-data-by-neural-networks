# Processing of missing data by neural networks


This is the training code for our paper "*Processing of missing data by neural networks*".

In order to repeat the experiments from column "*our*":

1. in Figure 2:
    ```bash
        python ae.py
    ```
    This code will download MNIST database, run training with default (parameters) and test reconstruction on random
    test images. In order to use non-default parameters, modify initial lines in file "ae.py".

2. in Table 3 (except the first row):
    ```bash
        python rbfn.py --data_dir ./data_rbfn/bands/ --learning_rate 0.25 --batch_size 75 --training_epochs 500 --n_hidden_1 25 --n_distribution 3
    ```

   This command works for data set "bands", other dat sets attached to this code are as follows:
    * hepatitis
    * horse
    * kidney_disease
    * mammographics
    * pima
    * winconsin
