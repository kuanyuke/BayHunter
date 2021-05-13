
import os
# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import os.path as op
import matplotlib
matplotlib.use('PDF')

from BayHunter import PlotFromStorage
from BayHunter import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer
from BayHunter import ModelMatrix
from BayHunter import SynthObs
from BayHunter.diagnostics import convergence_diagnostics
import logging



#
# console printout formatting
#
formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger()


#
# ------------------------------------------------------------  obs SYNTH DATA
#
# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)

# add noise to create observed data
# order of noise values (correlation, amplitude):
# noise = [corr1, sigma1, corr2, sigma2] for 2 targets
noise = [0.0, 0.012, 0.98, 0.005]

# Load observed data (synthetic test data)
xswsv, _yswsv = np.loadtxt('observed/st3_rdispph.dat').T
yswsv_err = SynthObs.compute_expnoise(_yswsv, corr=noise[0], sigma=noise[1])
yswsv = _yswsv + yswsv_err

## RF
xrf, _yrf = np.loadtxt('observed/st3_prf.dat').T
yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
yrf = _yrf + yrf_err

#
# -------------------------------------------  get reference model for BayWatch
#
# Create truemodel only if you wish to have reference values in plots
# and BayWatch. You ONLY need to assign the values in truemodel that you
# wish to have visible.

dep, vs = np.loadtxt('observed/st3_mod.dat', usecols=[0, 2], skiprows=1).T
pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
pvs = np.repeat(vs, 2)

truenoise = np.concatenate(([noise[0]], [np.std(yswsv_err)],   # target 1
                            [noise[2]], [np.std(yrf_err)]))  # target 2	

explike = SynthObs.compute_explike(yobss=[yswsv, yrf], ymods=[_yswsv, _yrf],
                                   noise=truenoise, gauss=[False, True],
                                   rcond=initparams['rcond'])
ra = 0
truemodel = {'model': (pdep, pvs,ra),
             'nlays': 3,
             'noise': truenoise,
             'explike': explike
             }
            
#
#  -----------------------------------------------------------  DEFINE TARGETS
#
# Only pass x and y observed data to the Targets object which is matching
# the data type. You can chose for SWD any combination of Rayleigh, Love, group
# and phase velocity. Default is the fundamendal mode, but this can be updated.
# For RF chose P or S. You can also use user defined targets or replace the
# forward modeling plugin wih your own module.

target1 = Targets.RayleighDispersionPhase(xswsv, yswsv, yerr=yswsv_err)
target2 = Targets.PReceiverFunction(xrf, yrf)
target2.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)

targets = Targets.JointTarget(targets=[target1, target2])
#
#
#  ---------------------------------------------------  Quick parameter update
#
# "priors" and "initparams" from config.ini are python dictionaries. You could
# also simply define the dictionaries directly in the script, if you don't want
# to use a config.ini file. Or update the dictionaries as follows, e.g. if you
# have station specific values, etc.
# See docs/bayhunter.pdf for explanation of parameters

#setup temperatures  used for parallel tempering
#There are many ways to define temperature and different way will affect the efficiency of convergence.
#Here I try to extract temperatur from a log-unform distribution between 1 and 10**0.3
#You can also try the other setting
#Each temperature will be assigned to a chain, so the number of temperatures define the number of chains in each group
##total number of chains = number of temperture * ngroups
from create_tmp import log_uniform
loguniform = log_uniform(a=0, b=0.3, base=10)
temp1 = loguniform.rvs(size=2)
temp0 = np.ones(3) #only when temperature == 1, the chains will be considered for the final models
temp = np.concatenate((temp0, temp1))


#freq_swap: frequency of temperture exchange
#init_swap: when to start swap temperture
#propdist = 0.015, 0.015, 0.015, 0.005, 0.005, 0.025 (last two are for radial anisotropy)

priors.update({'mohoest': (38, 4),  # optional, moho estimate (mean, std)
               'rfnoise_corr': 0.96,
               'swdnoise_corr': 0.
               })

initparams.update({'ngroups' : 1,
                   'iter_burnin': (2048* 4),
                   'iter_main': (2048 * 2),
                   'temperatures':temp,
                   'propdist': (0.025, 0.025, 0.015, 0.005, 0.005, 0.025, 0.025),
                   })


#
#  -------------------------------------------------------  MCMC BAY INVERSION
#
# Save configfile for baywatch. refmodel must not be defined.
utils.save_baywatch_config(targets, path='.', priors=priors,
                           initparams=initparams, refmodel=truemodel, plot_ra = False)
optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                           random_seed=None)

# default for the number of threads is the amount of cpus == number of chains per group + 1.
# parallel tempering takes 1 cpu
# nthreads = len(temp) +1 (for parellel tempering) + 1(for BayWatch)
# if baywatch is True, inversion data is continuously send out (dtsend)
# to be received by BayWatch (see below).
optimizer.mp_inversion(nthreads=7, baywatch=True, dtsend=1)


#
#
# #  ---------------------------------------------- Model resaving and plotting
path = initparams['savepath']
cfile = '%s_config.pkl' % initparams['station']
configfile = op.join(path, 'data', cfile)
obj = PlotFromStorage(configfile)
obj.save_final_distribution(maxmodels=100000, dev=0.05)
# Save a selection of important plots
obj.save_plots(nchains=6,refmodel=truemodel)
# step: every 100 step calculate convergence once, if convergence value psrf < 1.1
# then it converges
convergence_diagnostics(configfile, 'diagnostics.png', step = 100)
obj.merge_pdfs()

#
# If you are only interested on the mean posterior velocity model, type:
# file = op.join(initparams['savepath'], 'data/c_models.npy')
# models = np.load(file)
# singlemodels = ModelMatrix.get_singlemodels(models)
# vs, dep = singlemodels['mean']

#
# #  ---------------------------------------------- WATCH YOUR INVERSION
# if you want to use BayWatch, simply type "baywatch ." in the terminal in the
# folder you saved your baywatch configfile or type the full path instead
# of ".". Type "baywatch --help" for further options.

# if you give your public address as option (default is local address of PC),
# you can also use BayWatch via VPN from 'outside'.
# address = '139.?.?.?'  # here your complete address !!!
