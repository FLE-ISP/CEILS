import networkx as nx
import matplotlib.pyplot as plt
from core.build_struct_eq import *
from core.counter_causal_generator import *
from core.metrics import *

# create folder to save models
os.makedirs('models', exist_ok=True)

def graph_adult_income():
    '''
    returns a graph for the adult income dataset. causal graph obtained from https://arxiv.org/pdf/1611.07438.pdf
    '''

    G = nx.DiGraph(directed=True)

    G.add_node("race")
    G.add_edges_from([("race", "income"), ("race", "occupation"), ("race", "marital-status"), ("race", "hours-per-week"), ("race", "education")])

    G.add_node("age")
    G.add_edges_from([("age", "income"), ("age", "occupation"), ("age", "marital-status"), ("age", "workclass"), ("age", "education"),
                      ("age", "hours-per-week"), ("age", "relationship")])

    G.add_node("native-country")
    G.add_edges_from([("native-country", "education"), ("native-country", "workclass"), ("native-country",  "hours-per-week"),
                      ("native-country", "marital-status"), ("native-country", "relationship"), ("native-country", "income") ])

    G.add_node("sex")
    G.add_edges_from([("sex", "education"), ("sex", "hours-per-week"), ("sex", "marital-status"), ("sex", "occupation"),
                      ("sex", "relationship"), ("sex", "income") ])

    G.add_node("education")
    G.add_edges_from([("education", "occupation"), ("education", "workclass"), ("education", "hours-per-week" ), ("education", "relationship"),
                      ("education", "income") ])

    G.add_node("hours-per-week")
    G.add_edges_from([("hours-per-week", "workclass"), ("hours-per-week", "marital-status" ), ("hours-per-week", "income")])

    G.add_node("workclass")
    G.add_edges_from([("workclass", "occupation"), ("workclass", "marital-status" ), ("workclass", "income")])

    G.add_node("marital-status")
    G.add_edges_from([("marital-status", "occupation"), ("marital-status", "relationship"), ("marital-status", "income")])

    G.add_node("occupation")
    G.add_edges_from([("occupation", "income")])

    G.add_node("relationship")
    G.add_edges_from([("relationship", "income")])

    G.add_node("income")
    return G


def load_adultdataset(nodes):
    '''
    read dataset and preprocessing for adult income dataset
    return data only for the nodes
    '''

    df = pd.read_csv("data/adult_income_dataset.csv")
    print(df.shape)
    # Binarize the target 0 = <= credit; 1 = >50K
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    # Finding the special characters in the data frame
    df.isin(['?']).sum(axis=0)

    # code will replace the special character to nan and then drop the columns
    df['native-country'] = df['native-country'].replace('?', np.nan)
    df['workclass'] = df['workclass'].replace('?', np.nan)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    # dropping the NaN rows now
    df.dropna(how='any', inplace=True)
    print(df.shape)


    # categorical variables
    catvars = ['workclass',  'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'native-country']
    # education order > https: // www.rdocumentation.org / packages / arules / versions / 1.6 - 6 / topics / Adult
    df['education'] = df['education'].map(
        {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
         'Prof-school': 9, 'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college':12, 'Bachelors': 13, 'Masters': 14,'Doctorate': 15}).astype(int)

    #create quickaccess list with numerical variables labels
    numvars = ['age', 'hours-per-week']

    #  dic categories Index(['A11', 'A12', 'A13', 'A14'], dtype='object')
    dict_categorical = {}
    for c in catvars:
        dict_categorical[c] = list(df[c].astype("category").cat.categories)
        df[c] = df[c].astype("category").cat.codes

    #  all features as float
    df = df.astype("float64")

    df["income"] = df["income"].astype("int32")
    # save codes
    with open('dict_adult.txt', 'w') as f:
        f.write(str(dict_categorical))

    return df[nodes]

### START EXPERIMENT
'''
# Downoload and save the dataset
s = requests.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv").text

df = pd.read_csv(io.StringIO(s), names=["age", "workclass", "fnlwgt", "education", "education-num",
                                         "marital-status", "occupation", "relationship", "race", "sex",
                                         "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"])

# save original dataset
df.to_csv("data/adult_income_dataset.csv")  # save as csv file
'''

#  generate graph
G = graph_adult_income()
nx.draw_circular(G, with_labels=True)
plt.show()
nodes = list(G.nodes)

# info about features
constraints_features = {"immutable": ["race", "native-country", "sex"], "higher": ["age", "education"]}
categ_features = ["sex", "race", "occupation", "marital-status", "education", "workclass", "relationship", "native-country"]

# load dataset
df = load_adultdataset(nodes)

Y = df["income"]
X = df.drop(["income"], axis=1)
# # Target encoding the categorical variables
# cols_target = ["race", "occupation", "marital-status", "education", "workclass", "relationship", "native-country"]
# X[cols_target] = X[cols_target].astype(float)
# enc = ce.target_encoder.TargetEncoder(cols=cols_target).fit(X, Y)
# X = enc.transform(X, Y)

###  Create structural equations (NNs saved in models folder), store residuals (saved in data folder)
struct_eq, nn_causal = create_structural_eqs(X, Y, G, n_nodes_se=100, n_nodes_M=180)

###  Create couterfactuals (saved in data folder)
create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features, numCF=20)

###  Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features)

print("------------------------- ONLY intersection")
calculate_metrics(X, Y, G, categ_features, constraints_features, intersection_only=True)