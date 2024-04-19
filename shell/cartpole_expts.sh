python run.py -m name=mpc_cartpole alg=mpc num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib

git add experiments/*
git commit -m "Added experiment results"
git push origin master

python run.py -m name=mpc_frozencartpole alg=mpc num_iters=300 eval_frequency=10 env=frozencartpole seed="range(5)" hydra/launcher=joblib

git add experiments/*
git commit -m "Added experiment results"
git push origin master