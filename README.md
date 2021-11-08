
![137498084-1adb38cb-0491-44a9-8bf9-b8b326d50496](https://user-images.githubusercontent.com/92588313/137943056-142a3568-02a7-46a8-b1a8-67fc3631ae79.jpg)

# CEILS
Counterfactual Explanations as Interventions in Latent Space (CEILS) is a methodology to generate counterfactual explanations capturing by design the underlying causal relations from the data, and at the same time to provide feasible recommendations to reach the proposed profile.

> ### Authors & contributors:
> Riccardo Crupi, Alessandro Castelnovo, Beatriz San Miguel Gonzalez, Daniele Regoli

You can cite this work as:
```
@article{crupi2021counterfactual,
  title={Counterfactual Explanations as Interventions in Latent Space},
  author={Crupi, Riccardo and Castelnovo, Alessandro and Regoli, Daniele and Gonzalez, Beatriz San Miguel},
  journal={arXiv preprint arXiv:2106.07754},
  year={2021}
}
```


## Documentation

To know more about this research work, please refer to our [full paper](https://arxiv.org/abs/2106.07754).

Currently, CEILS has been published and/or presented in:
- 8th Causal Inference Workshop at UAI ([causalUAI2021](https://sites.google.com/uw.edu/causaluai2021/home))
  ([Video](https://www.youtube.com/watch?v=adTNX_Um47I)) by Riccardo Crupi</li>
- [Workshop on Explainable AI in Finance](https://sites.google.com/view/2021-workshop-explainable-ai/home) @ICAIF 2021 by Beatriz San Miguel</li>



 
## Installation
Create a new environment based on Python 3.9 or 3.6 and install the requirements.

Python 3.9:
```
pip install -r requirements.txt
```

Python 3.6:
```
pip install -r requirements_py36.txt
```

## CEILS Workflow
![image](https://user-images.githubusercontent.com/92302358/140288321-2ca4caf8-2e32-421c-916c-b466d6006663.png)

Prepare the dataset X as pandas.DataFrame, target variable Y as pandas.Series and causal graph G as networkx.DiGraph.
Define the feature constrains (immutable, higher, lower) in generating counterfactuals, e.g. constraints_features = {"immutable": ["native-country"], "higher": ["age"]}

### Create structural equations F: U->X, classifier C: X->Y (fitted inside the method) and the composition C_causal(u) = C(F(u))
F, C_causal = create_structural_eqs(X, Y, G)

### Create couterfactuals: CEILS from the latent space (U), [Alibi](https://github.com/SeldonIO/alibi) from feature space (X)
create_counterfactuals(X, Y, G, F, C_causal, constraints_features, numCF=20)

### Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features)

![image](https://user-images.githubusercontent.com/92302358/140289908-c827961d-f4b7-457d-9bd8-4e8f226fbf4f.png)

## Experiments

Currenly we have included 3 experiments based on public datasets and 2 experiments with synthetic data:
- [German credit dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [Sachs](https://www.bristol.ac.uk/Depts/Economics/Growth/sachs.htm)
- [Adult income dataset](https://archive.ics.uci.edu/ml/datasets/adult)

Experiments are under a specific folder in:
>\experiments_run

We recommend to check the `run_experiment.py` file to know the details and understand the whole CEILS workflow. 

Synthetic datasets experiments are the best way to have a first understanding of our solution
