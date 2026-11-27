"""Generate portfolio results and figures from a fresh run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from projects.fundamentals import knn_benchmark, dimension_experiment, objective, gradient
from projects.neural_network import experiment, logits

out=Path(__file__).resolve().parent/'results'; out.mkdir(exist_ok=True)

def main():
    dimensions=dimension_experiment()
    metrics,history,p,X,y=experiment()
    beta=np.array([-.7697,.5005]); features=np.column_stack([np.ones(5),np.arange(1,6)]); labels=np.array([0,0,1,1,1])
    results={'knn':knn_benchmark(),'logistic':{'objective':float(objective(beta,features,labels)), 'gradient':gradient(beta,features,labels).tolist()},'dimensions':dimensions,'neural_network':metrics}
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4)); d=[r['dimension'] for r in dimensions]
    ax.errorbar(d,[r['mean'] for r in dimensions],yerr=[r['std'] for r in dimensions],marker='o',capsize=3)
    ax.set(xscale='log',xlabel='Dimensions',ylabel='Nearest-neighbor Euclidean distance',title='Data become sparse in high dimensions'); fig.tight_layout(); fig.savefig(out/'nearest-neighbors.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(len(history))*100,history); ax.set(xlabel='Epoch',ylabel='Binary cross-entropy',title='Manual backpropagation: training loss'); fig.tight_layout(); fig.savefig(out/'training-loss.svg'); plt.close(fig)
    xx,yy=np.meshgrid(np.linspace(-1,1,250),np.linspace(-1,1,250)); zz=np.asarray(logits(p,jnp.asarray(np.c_[xx.ravel(),yy.ravel()],dtype=jnp.float32))).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(5,5)); ax.contourf(xx,yy,zz>0,levels=[-.5,.5,1.5],alpha=.25,cmap='coolwarm'); ax.scatter(X[:,0],X[:,1],c=y,s=8,cmap='coolwarm'); ax.add_patch(plt.Circle((0,0),.5,fill=False,linestyle='--',color='black',label='True boundary')); ax.set(aspect='equal',xlabel='Feature 1',ylabel='Feature 2',title='Learned nonlinear boundary (held-out points)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'decision-boundary.svg'); plt.close(fig)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
