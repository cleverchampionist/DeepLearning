import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print("label_names:", label_names)
print("label0", labels[0])
print("feature_names:", feature_names[0])
print("feature", features[0])

train, test, train_labeles, test_labels = train_test_split(
    features, labels, test_size =0.40, random_state = 42
)
from sklearn.naive_bayes import GaussianNB
GNBclf = GaussianNB()
model = GNBclf.fit(train, train_labeles)
preds = GNBclf.predict(test)
print(preds)
