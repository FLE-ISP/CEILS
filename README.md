# CEILS
![137498084-1adb38cb-0491-44a9-8bf9-b8b326d50496](https://user-images.githubusercontent.com/92588313/137943056-142a3568-02a7-46a8-b1a8-67fc3631ae79.jpg)
Counterfactual Explanations as Interventions in Latent Space (CEILS) is a methodology to generate counterfactual explanations capturing by design the underlying causal relations from the data, and at the same time to provide feasible recommendations to reach the proposed profile.
This work is based on this paper https://arxiv.org/pdf/2106.07754.pdf
A video presentation can be found here https://www.youtube.com/watch?v=adTNX_Um47I

To install the necessary libraries run into terminal: pip install -r requirements.txt

## Workflow
![image](https://user-images.githubusercontent.com/92302358/140288321-2ca4caf8-2e32-421c-916c-b466d6006663.png)

Prepare the dataset X as pandas.DataFrame, target variable Y as pandas.Series and causal graph G as networkx.DiGraph.
Define the feature constrains (immutable, higher, lower) in generating counterfactuals, e.g. constraints_features = {"immutable": ["native-country"], "higher": ["age"]}

### Create structural equations F: U->X, classifier C: X->Y (fitted inside the method) and the composition C_causal(u) = C(F(u))
F, C_causal = create_structural_eqs(X, Y, G)

### Create couterfactuals: CEILS from the latent space (U), Alibi from feature space (X)
create_counterfactuals(X, Y, G, F, C_causal, constraints_features, numCF=20)

### Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features)

![image](https://user-images.githubusercontent.com/92302358/140289908-c827961d-f4b7-457d-9bd8-4e8f226fbf4f.png)
