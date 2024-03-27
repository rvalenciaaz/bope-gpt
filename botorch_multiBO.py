import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound,qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)

SMOKE_TEST = os.environ.get("SMOKE_TEST")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dtype = torch.float64

MC_SAMPLES = 128 if not SMOKE_TEST else 16

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

BATCH_SIZE = 5

dim = 2
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device)

N=20
iteration_number=20
# Define the Sobol sampler for initial samples
sobol_engine = SobolEngine(dimension=2, scramble=False)  # 2 dimensions for your input space
train_x = draw_sobol_samples(bounds=bounds, n=1, q=N).squeeze(0)

refi=torch.tensor([0.,  0.], device='cpu', dtype=torch.float64)

train_y = #branin(train_x, negate=True).unsqueeze(-1) the 2 output function

#separate GPs, but it could a multioutput GP
modelsub1= SingleTaskGP(train_X=train_x, train_Y=train_y[:,0].reshape(init_samples,1))
modelsub2= SingleTaskGP(train_X=train_x, train_Y=train_y[:,1].reshape(init_samples,1))

mll1 = ExactMarginalLogLikelihood(modelsub1.likelihood, modelsub1)
mll2 = ExactMarginalLogLikelihood(modelsub2.likelihood, modelsub2)

fit_gpytorch_mll(mll1)
fit_gpytorch_mll(mll2)

premol=[mll1.model,mll2.model]

new_model=ModelListGP(*premol)

with torch.no_grad():
    train_y = new_model.posterior(train_x).mean

partitioning = FastNondominatedPartitioning(
    ref_point=refi,
    Y=train_y,
    )

best_list=[]
suma=init_samples
# Perform 20 BO iterations
for i in range(iteration_number):
    print("Iteration: "+str(i))
    acq_func = qExpectedHypervolumeImprovement(
        model=new_model,
        ref_point=refi,
        partitioning=partitioning,
        sampler=qehvi_sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds.cpu(),
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = candidates.detach()
    with torch.no_grad():
        new_y = model.posterior(new_x.cpu()).mean

    train_x=torch.vstack([train_x,new_x])
    train_y=torch.vstack([train_y,new_y])
    
    #suma=suma+BATCH_SIZE
    #del modelsub1,modelsub2,mll1,mll2, premol
    modelsub1= SingleTaskGP(train_X=train_x, train_Y=train_y[:,0].reshape(suma,1))
    modelsub2= SingleTaskGP(train_X=train_x, train_Y=train_y[:,1].reshape(suma,1))

    mll1 = ExactMarginalLogLikelihood(modelsub1.likelihood, modelsub1)
    mll2 = ExactMarginalLogLikelihood(modelsub2.likelihood, modelsub2)
    
    fit_gpytorch_mll(mll1)
    fit_gpytorch_mll(mll2)

    with torch.no_grad():
        train_y = new_model.posterior(train_x.cpu()).mean

    partitioning = FastNondominatedPartitioning(
        ref_point=refi,
        Y=train_y,
        )