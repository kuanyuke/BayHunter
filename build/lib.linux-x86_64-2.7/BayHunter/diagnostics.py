import os.path as op
import glob
import numpy as np
import scipy
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from BayHunter import utils
from BayHunter.Models import ModelMatrix, Model

class convergence_diagnostics(object):                  
        def __init__(self,configfile, filename, step = 100 ):
        	condict = self.read_config(configfile)
        	self.priors = condict['priors']
        	self.initparams = condict['initparams']
        	self.datapath = op.dirname(configfile)
        	self.figpath = self.datapath.replace('data', '')
        	self.targets = condict['targets']
        	self.ntargets = len(self.targets)
        	
        	self.get_coldchains()
        	self.init_filelists()
        	
        	self.init_tmp = self.initparams.get('temperatures')
        	self.burnin = self.initparams['iter_burnin']
        	self.maxlayers = int(self.priors['layers'][1]) + 1
        	
        	self.step = step
        	
        	chains_psrf = self.run_conv()
        	self.plot(chains_psrf,  filename)
        	
        def read_config(self, configfile):
        	return utils.read_config(configfile)
        
        	
        def get_coldchains(self):
        	#temperature        
        	init_tmp = self.initparams.get('temperatures')
        	self.init_tmp =np.array(init_tmp)

        	self.ncoldchains = np.count_nonzero(self.init_tmp == 1)
        	self.ngroups = self.initparams.get('ngroups')
        	self.pernchains=  len(self.init_tmp)
        	for i in range(self.ngroups):
                	std = np.array(range(self.ncoldchains))
                	if i == 0:
                        	self.coldchainsidx=std
                	else:
                        	stdnew =  std + self.pernchains * i
                        	self.coldchainsidx = np.concatenate((self.coldchainsidx,stdnew ))
		
        def init_filelists(self):
        	filetypes = ['models', 'noise']
        	filepattern = op.join(self.datapath, 'tmp???_p%d%s.npy')
        	files = []
        	size = []

        	for ftype in filetypes:
            		p1files = sorted(glob.glob(filepattern % (1, ftype)))
            		p2files = sorted(glob.glob(filepattern % (2, ftype)))
            		files.append([p1files, p2files])
            		size.append(len(p1files) + len(p2files))
	
        	if len(set(size)) == 1:
            		self.modfiles,  self.noisefiles = files
        	else:
            		logger.info('You are missing files. Please check ' +
                        '"%s" for completeness.' % self.datapath)
            		logger.info('(filetype, number): ' + str(zip(filetypes, size)))
            		
        def _return_c_p_t(self, filename):
        	"""Return chainindex, phase number, type of file from filename.
       	 	Only for single chain results.
       		"""
        	c, pt = op.basename(filename).split('.npy')[0].split('_')
        	cidx = int(c[3:])
        	phase, ftype = pt[:2], pt[2:]

        	return cidx, phase, ftype    
	        	
        def run_conv(self):
                modfiles = self.modfiles[0]
                noisefiles = self.noisefiles[0]

		modelsdata= []
		noisesdata = []
        	for i, modfile in enumerate(modfiles):
            		chainidx, _, _ = self._return_c_p_t(modfile)
            		if chainidx not in self.coldchainsidx:
            			continue
        		
        		model = np.load(modfile)
        		noise = np.load(noisefiles[i]).T
        		
        		modelsdata.append(model)
        		noisesdata.append(noise)
        	
        	self.nchains = self.ncoldchains*self.ngroups
        	S_CHAINS_T = np.zeros([self.nchains, (self.burnin //self.step)+1, 3+self.ntargets])

        	#obtain Scalar Model Indicator value of each chain
        	for j in range(self.nchains):
			p1chainmodels = modelsdata[j]
			p1chainnoise = noisesdata[j].T
			S_CHAINS_T[j] = self.ChainScalarModelIndicator( S_CHAINS_T[j], p1chainmodels,p1chainnoise)
			
		chains_psrf = self._get_conv_psrf(S_CHAINS_T)
		return chains_psrf
		
	def _get_conv_psrf(self,S_CHAINS ):
		"""
		potential scale reduction factor Gelman-Rubin	 		
		"""

        	thin = self.step
        	n2 = int(self.burnin/thin)
        	#Scalar Typeset
        	RvectorST = np.zeros(n2)
		for i in range(50,n2):
    			RvectorST[i] = np.nanmean(self.gelman_rubin(S_CHAINS[:,:i]))
    		return RvectorST
    				
    	
	def ChainScalarModelIndicator(self, chainindicators, chainmodels, chainnoise):
		k=0
		for i in range(0, len(chainmodels)-self.step,self.step ):
			model = chainmodels[i]
			noise = chainnoise[i]
    			n, vs, ra, z_vnoi = Model.split_modelparams(model)
    			parameters = [vs, ra, z_vnoi, noise]
    			
    			chainindicators[k]= self.ScalarModelIndicatorType(parameters)

    			k=k+1
		return 	chainindicators

	def ScalarModelIndicatorType(self, STATE):
    		sum1 = 0
    		sum2 = 0
    		sum3 = 0
    		for j in range(1,len(STATE[0])):
        		sum1 = sum1 + STATE[0][j]*2**j
        		sum2 = sum2 + STATE[1][j]*2**j
			sum3 = sum3 + STATE[2][j]*2**j
        	total = [sum1, sum2, sum3]
        	
        	for i in range(self.ntargets):
        		idx = i *2 +1 
			total.append(STATE[3][idx])
				
    		return total 
    	def plot(self, chains_psrf, filename):
    		x = range(-1*self.burnin, 0, self.step )
    		y =  chains_psrf
    		if len(x) != len(chains_psrf):
    			if  len(chains_psrf) < len(x): 
    				diff = len(x) - len(chains_psrf) 
    				x = x[diff:]
    			else:
    				diff = len(chains_psrf)- len(x) 
    				y = y[diff:]
    		fig, ax = plt.subplots(figsize=(7, 4.4))		
    		ax.plot(x, y)
    		y = np.ones(len(x))*1.1
    		ax.plot( x,y, '--')
    		ax.set_xlabel('Iterations')
        	ax.set_ylabel('PSRF')
    		ax.set_ylim(1, 2)
    		fig.savefig(op.join(self.figpath, filename))

	def gelman_rubin(self, x, return_var=False):		
    		if np.shape(x) < (2,):
   	 		raise ValueError(
            		'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    		try:
   	 		m, n = np.shape(x)
    		except ValueError:
        		return [self.gelman_rubin(np.transpose(y)) for y in np.transpose(x)]
        		
    		# Calculate between-chain variance
    		B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    		# Calculate within-chain variances
    		W = np.sum(
        		[(x[i] - xbar) ** 2 for i,
         		xbar in enumerate(np.mean(x,
                                   	1))]) / (m * (n - 1))
		
    		# (over) estimate of variance
    		s2 = W * (n - 1) / n + B_over_n
    
    		if return_var:
        		return s2

    		# Pooled posterior variance estimate
    		V = s2 + B_over_n / m

    		# Calculate PSRF
    		if W !=0:
    			R = V / W
		else:
			R = 0
    		return np.sqrt(R)
    		
