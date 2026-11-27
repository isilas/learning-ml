"""Two-layer JAX network with explicit backpropagation and stable binary loss."""
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

def initialize(seed=0, hidden=16):
    k1,k2=jax.random.split(jax.random.PRNGKey(seed))
    return {'W1':jax.random.normal(k1,(2,hidden))*jnp.sqrt(2/2), 'b1':jnp.zeros(hidden), 'W2':jax.random.normal(k2,(hidden,1))*jnp.sqrt(2/hidden), 'b2':jnp.zeros(1)}

def logits(p,x):
    return jax.nn.relu(x@p['W1']+p['b1'])@p['W2']+p['b2']

def loss(p,x,y):
    z=logits(p,x)
    return jnp.mean(jax.nn.softplus(z)-y*z)

def manual_grad(p,x,y):
    z1=x@p['W1']+p['b1']; a1=jax.nn.relu(z1)
    z2=a1@p['W2']+p['b2']; dz2=(jax.nn.sigmoid(z2)-y)/y.size
    dz1=(dz2@p['W2'].T)*(z1>0)
    return {'W1':x.T@dz1,'b1':dz1.sum(axis=0),'W2':a1.T@dz2,'b2':dz2.sum(axis=0)}

@jax.jit
def step(p,x,y):
    g=manual_grad(p,x,y)
    return jax.tree.map(lambda w,d:w-.1*d,p,g)

def experiment(seed=42, epochs=5000):
    rng=np.random.default_rng(seed)
    X=rng.uniform(-1,1,(2000,2)).astype(np.float32)
    y=(np.linalg.norm(X,axis=1)<.5).astype(np.float32)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=seed,stratify=y)
    x,t=jnp.asarray(Xtr),jnp.asarray(ytr[:,None]); p=initialize(seed)
    mg=manual_grad(p,x,t); ag=jax.grad(loss)(p,x,t)
    error=max(float(jnp.max(jnp.abs(mg[k]-ag[k]))) for k in p)
    history=[float(loss(p,x,t))]
    for i in range(epochs):
        p=step(p,x,t)
        if (i+1)%100==0: history.append(float(loss(p,x,t)))
    pred=np.asarray(logits(p,jnp.asarray(Xte))).ravel()>0
    metrics={'test_samples':len(yte),'epochs':epochs,'initial_train_loss':history[0],'final_train_loss':history[-1],'test_loss':float(loss(p,jnp.asarray(Xte),jnp.asarray(yte[:,None]))),'test_accuracy':float(accuracy_score(yte,pred)), 'test_balanced_accuracy':float(balanced_accuracy_score(yte,pred)), 'majority_baseline_accuracy':float(np.mean(yte==0)), 'max_gradient_error':error}
    return metrics,history,p,Xte,yte
