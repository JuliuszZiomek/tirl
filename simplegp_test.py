from barl.models.simple_gp import SimpleGp
from argparse import Namespace
import numpy as np

np.random.seed(0)
data = Namespace()
data.x = [np.random.randn(1,) for _ in range(10)]
data.y = [np.sin(dx) for dx in data.x]
params = {"n_dimx": 1}
gp = SimpleGp(data=data, params=params)

test_x = [np.random.randn(1,) for _ in range(10)]
mu, std = gp.get_post_mu_cov(test_x,full_cov=False)
std_removed = gp.get_post_std_w_point_removed(test_x, [1])

data.x = data.x[:1] + data.x[2:]
data.y = data.y[:1] + data.y[2:]
gp = SimpleGp(data=data, params=params)
_, std_manually_removed = gp.get_post_mu_cov(test_x,full_cov=False)

assert np.all(abs(std_removed - std_manually_removed) < 0.001 * abs(std_manually_removed))
print("Test passed succesfully.")