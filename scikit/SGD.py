import numpy as np
from sklearn import linear_model

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)

x = rng.randn(n_samples, n_features)
y = rng.randn(n_samples)
SGDReg = linear_model.SGDRegressor(
    max_iter=1000, penalty="elasticnet", loss = 'huber', tol = 1e-3, average=True, 
)
SGDReg.fit(x,y)