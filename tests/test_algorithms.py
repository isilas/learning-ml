import numpy as np
import jax
import jax.numpy as jnp
from projects.fundamentals import knn, gradient, objective, split_gain
from projects.neural_network import initialize, manual_grad, loss

def test_knn():
    x=np.array([[1,2],[2,3],[3,1],[6,5],[7,7]]); y=np.array([0,0,0,1,1]); q=np.array([5,5])
    assert knn(x,y,q,3)==1 and knn(x,y,q,5)==0

def test_logistic_gradient():
    X=np.column_stack([np.ones(5),np.arange(1,6)]); y=np.array([0,0,1,1,1]); b=np.array([-.7697,.5005]); h=1e-5
    numeric=np.array([(objective(b+h*v,X,y)-objective(b-h*v,X,y))/(2*h) for v in np.eye(2)])
    np.testing.assert_allclose(gradient(b,X,y),numeric,atol=1e-8)

def test_gini():
    assert split_gain(np.arange(4),np.array([0,0,1,1]),1.5)==.5

def test_manual_backprop():
    p=initialize(); x=jnp.array([[.1,.2],[-.4,.7],[.8,-.3]]); y=jnp.array([[1.],[0.],[0.]])
    expected=jax.grad(loss)(p,x,y)
    for key,value in manual_grad(p,x,y).items():
        np.testing.assert_allclose(value,expected[key],atol=1e-6)
