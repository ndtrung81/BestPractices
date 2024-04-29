The scripts in this folder are the non-jupyter version of the scripts under the notebooks folder.
All credits go to the main repo

- CVBF folder should be present in the same level with these python scripts.
- ML_figures folder same thing

Maybe need to pip install the following packages:

pandas
matplotlib
ydata-profiling
torch

On Midway3, need to load python/anaconda-2021.05, then activate the env.
 

```
module load python/anaconda-2021.05
source /project/rcc/trung/BestPractices-ML/ml-design/bin/activate
```

Step 1: Preprocessing the dataset

```
python preprocessing.py
```

Step 2: Splitting dataset into training, validating and testing sets

```
python splitting.py
```

Step 3: Training several classical models with scikit-learn

```
python modeling_classical_models.py
```

Step 4: Training neural networks with PyTorch

```
python modeling_neural_networks.py
```

Step 5: Visualizing the results

```
python visualizing_results.py
```
