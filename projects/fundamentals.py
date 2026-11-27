"""Portfolio reproduction: nearest neighbors, penalized likelihood and split criteria."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def knn(train, labels, query, k):
    """Euclidean distance and majority vote; ties choose smallest class."""
    train, labels, query = map(np.asarray, (train, labels, query))
    if not 1 <= k <= len(train):
        raise ValueError('k must be between 1 and training-set size')
    idx = np.argsort(np.linalg.norm(train - query, axis=1), kind='stable')[:k]
    classes, counts = np.unique(labels[idx], return_counts=True)
    return classes[counts.argmax()]

def objective(beta, X, y, l1=.1, l2=.1):
    z = X @ beta
    return np.mean(y*z - np.logaddexp(0, z)) - l1*np.abs(beta[1:]).sum() - l2*(beta[1:]**2).sum()

def gradient(beta, X, y, l1=.1, l2=.1):
    g = X.T @ (y-expit(X@beta)) / len(y)
    g[1:] -= l1*np.sign(beta[1:]) + 2*l2*beta[1:]
    return g

def gini(y):
    if len(y)==0:
        return 0.
    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum((counts/len(y))**2)

def split_gain(x, y, threshold):
    left=x<=threshold
    return gini(y) - (left.mean()*gini(y[left]) + (~left).mean()*gini(y[~left]))

def knn_benchmark():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    scaler=StandardScaler().fit(Xtr)
    Xtr,Xte=scaler.transform(Xtr),scaler.transform(Xte)
    pred=np.array([knn(Xtr,ytr,x,5) for x in Xte])
    ref=KNeighborsClassifier(n_neighbors=5).fit(Xtr,ytr).predict(Xte)
    return {'test_samples':len(yte),'accuracy':float(accuracy_score(yte,pred)), 'balanced_accuracy':float(balanced_accuracy_score(yte,pred)), 'agreement_with_sklearn':float(np.mean(pred==ref))}

def dimension_experiment(seed=0, n=1000):
    rng=np.random.default_rng(seed)
    results=[]
    for d in [1,2,5,10,20,50,100]:
        X=rng.random((n,d)); D=cdist(X,X); np.fill_diagonal(D,np.inf)
        nn=D.min(axis=1)
        typical=np.median(D[np.triu_indices(n,1)])
        results.append({'dimension':d,'mean':float(nn.mean()),'std':float(nn.std()),'relative_to_median_pair_distance':float(nn.mean()/typical)})
    return results
