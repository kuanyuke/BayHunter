import copy
import numpy as np
import os.path as op
import multiprocessing as mp
from collections import Counter
from BayHunter import utils
import logging
import random

logger = logging.getLogger()


class convergence_diagnostics(object):                  
        def __init__(self, nchains=None,  iterations=None, maxlayers =None,
        		ntargets=None):
        	self.nchains = nchains
        	self.iterations= iterations *2
        	self.maxlayers = maxlayers
        	self.ntargets = ntargets
		self.dtype = np.float32
        	
        def run_conv(self, nthreads, chainidxs, iter_phase1, ncoldchains, sharedmodels = None, sharednoise=None, step = 10):
        	self.pernchains= nthreads
        	        	
        	ncoldchains = ncoldchains+1
        	self.step = step
        	self.iter_phase1  = int(iter_phase1)
        	S_CHAINS_T = np.zeros([ncoldchains,self.iter_phase1 //self.step,6])
            	
            	#models = np.frombuffer(sharedmodels, dtype=self.dtype).\
            	#reshape(self.nchains, (self.iterations * self.maxlayers * 3 ))

            	#noise = np.frombuffer(sharednoise, dtype=self.dtype).\
            	#reshape((self.nchains,  self.iterations * self.ntargets*2))

        	
        	#obtain Scalar Model Indicator value of each chain ?
        	for j, tmpchainidx in enumerate(chainidxs):
			if (tmpchainidx % self.pernchains)< ncoldchains:
				tmpchainmodels = sharedmodels[tmpchainidx].reshape(
            				self.iterations, self.maxlayers*3)
				p1chainmodels = tmpchainmodels[:self.iter_phase1, :]
				p1chainnoise = sharednoise[tmpchainidx].reshape(
            				self.iterations,  self.ntargets*2)
				S_CHAINS_T[j] = self.ChainScalarModelIndicator( S_CHAINS_T[j],
						 p1chainmodels,p1chainnoise)
		
		chains_psrf = self._get_conv_psrf(S_CHAINS_T)
		last_psrf = np.array(chains_psrf[-11:-1])
		if len(last_psrf[last_psrf <=1.1]) ==10:
			return False
		else:
			return True
		
	def _get_conv_psrf(self,S_CHAINS ):
		"""
		potential scale reduction factor Gelman-Rubin	 		
		"""
        	n2 = int(self.iter_phase1/self.step)
        	#Scalar Typeset
        	RvectorST = np.zeros(n2)
		for i in range(10,n2):
    			RvectorST[i-1] = np.nanmean(self.gelman_rubin(S_CHAINS[:,:i]))
    		return RvectorST
    	
	def ChainScalarModelIndicator(self, chainindicators, chainmodels,chainnoise):
		k=0
		for i in range(0, len(chainmodels)-self.step,self.step ):
			model = chainmodels[i]

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
        	
        	for i in range(len(STATE[3])/2):
        		idx = i *2 +1 
			total.append(STATE[3][idx])
    		return total
    		

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
    		R = V / W

    		return np.sqrt(R)
    		
class Parallel_tempering(object):
    """
    Computation methods for Parallel_tempering.
    """
    def __init__(self, initparams={}, maxlayers=None, ntargets=None,
    		sharedtmpmodels=None,sharedtmpmisfits=None, 
    		sharedtmplikes=None, sharedtmpnoise=None, sharedtmpvpvs=None,
        	sharedcurrentlikes=None,sharediiter=None, 
        	sharedgettmp=None, sharedtmpidx=None, sharedconvergence=None,random_seed=None):
                 
        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.initparams.update(initparams)

        self.rstate = np.random.RandomState(random_seed)
        self.dtype = np.float32

        
        # set parameters
        self.maxlayers = maxlayers
        self.ntargets = ntargets
        
        
        #iteration
	self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
	self.iterations = self.iter_phase1 + self.iter_phase2
        self.addiiter = 0.5 *self.iterations

	#temperature        
	self.init_tmp = self.initparams.get('temperatures')
        self.freq_swap = self.initparams.get('freq_swap')
        self.init_swap = self.initparams.get('init_swap')
        self.ncoldchains = np.count_nonzero(self.init_tmp == 1)
        
        #nchains
        self.ngroups = self.initparams.get('ngroups')
        self.nchains = self.ngroups * len(self.init_tmp)
        self.pernchains=  len(self.init_tmp)

        
        self._init_arrays( sharedtmpmodels, sharedtmpmisfits,
        		sharedtmplikes ,sharedtmpnoise, sharedtmpvpvs,
        		sharedcurrentlikes,  sharediiter, 
        		sharedgettmp, sharedtmpidx, sharedconvergence)
      
      	
        self.convdia = convergence_diagnostics(nchains=self.nchains, 
        	        iterations=self.iterations, 
        		maxlayers =self.maxlayers, ntargets=self.ntargets) #,
        		#sharedmodels = sharedtmpmodels, sharednoise = sharedtmpnoise)   
        		                     
    def _init_arrays(self, sharedtmpmodels, sharedtmpmisfits,
        		sharedtmplikes ,sharedtmpnoise, sharedtmpvpvs,
        		sharedcurrentlikes, sharediiter, sharedgettmp,
        		sharedconvergence, sharedtmpidx):
       """Initialize shared array
       All models / likes will be saved and load from this array in 
       temperature space.       
       """
       self.ntmpmodels= self.iterations #*2
       self.sharedtmpmodels = np.frombuffer(sharedtmpmodels, dtype=self.dtype).\
            reshape(self.nchains, (self.ntmpmodels * self.maxlayers * 3 ))
       self.sharedtmpmisfits = np.frombuffer(sharedtmpmisfits, dtype=self.dtype).\
            reshape(self.nchains, (self.ntmpmodels *(self.ntargets + 1)))
       self.sharedtmplikes =np.frombuffer(sharedtmplikes, dtype=self.dtype).\
            reshape((self.nchains,  self.ntmpmodels))
       self.sharedtmpnoise =np.frombuffer(sharedtmpnoise, dtype=self.dtype).\
            reshape((self.nchains,  self.ntmpmodels * self.ntargets*2))
       self.sharedtmpvpvs = np.frombuffer(sharedtmpvpvs, dtype=self.dtype).\
            reshape((self.nchains,  self.ntmpmodels))  
            
       """Initialize shared array
       The paramters to swap temperatures will be saved and load from this array       
       """       
       self.sharedcurrentlikes = np.frombuffer(sharedcurrentlikes, dtype=self.dtype)
       
       self.sharediiter  = np.frombuffer(sharediiter, dtype=self.dtype)
        
       self.sharedgettmp = np.frombuffer(sharedgettmp, dtype=self.dtype).\
            reshape(self.ntmpmodels, self.nchains)
       self.sharedgettmp.fill(np.nan)
       
       self.sharedtmpidx = np.frombuffer(sharedtmpidx, dtype=self.dtype)
       self.sharedconvergence = np.frombuffer(sharedconvergence, dtype=self.dtype)
       self.sharedconvergence.fill(np.nan)      

        
    def _init_tmpidx(self): 
    	"""
    	Give every chain in parameter space a idx of temperature. 
    	Index will later be used to find the chains where T=1.
    	"""
        chainidxs = np.arange(self.nchains)
        for i in chainidxs:
        	self.sharedtmpidx[i] = int(i)
        	
	logger.info('Initial Temp %s' % self.init_tmp)
	   		
    def _get_init_tmp(self, chainidx ):
    	idx = chainidx % self.pernchains
    	
    	return self.init_tmp[idx]
    	
    def get_alpha(self, like1, like2, T1, T2): 
     	A = (1./T1 - 1./T2)
     	B = (like2-like1)
     	return A * B
    
    def swap_chains(self, ):
        likes = np.frombuffer(self.sharedcurrentlikes , dtype=self.dtype)
	sharedtmpidx = np.frombuffer(self.sharedtmpidx, dtype=self.dtype) 
	tmps = self.current_tmp
	
        chainidxs = self.chainidxs

	latest_likes = np.zeros(self.nchains)
        gettmp = np.nan * np.ones(self.nchains)
        latest_tmps = np.nan * np.ones(self.nchains)  
        latest_tmpidx = np.nan * np.zeros(self.nchains)  
        
        for chainidx in chainidxs:
        	latest_likes[chainidx] = likes[chainidx]
        	tmpidx = chainidx % self.pernchains
        	latest_tmps[chainidx] = tmps[tmpidx]
        	latest_tmpidx[chainidx] = sharedtmpidx[chainidx]
        	gettmp = latest_tmps #in case exchange some num don't exchage will be 0  
        	
	
        nswap = len(np.unique(tmps)) - np.isnan(latest_likes).sum()

        """check how many chains have invalid modles and decide how many swaps
        """
        available_tmp = []
        for i, nanlike in enumerate(latest_likes):
        	if not np.isnan(nanlike):
        		available_tmp.append(latest_tmps[i]) 
        if nswap ==1 or len(np.unique(available_tmp))== 1 :        	
        	return chainidxs, gettmp 
        		
        nswap =  len(self.init_tmp) - np.isnan(latest_likes).sum()
        
        """Only models are valid and temperatures are different will be exchanged
        """ 
        
	for i in range(nswap):
		chainidx1 = np.random.choice(chainidxs)
			
		like1 = latest_likes[chainidx1]	
			
		while np.isnan(like1):
			chainidx1 = np.random.choice(chainidxs)
			like1 = latest_likes[chainidx1]
			
		T1 = latest_tmps[chainidx1]	
			
		chainidx2 = chainidx1
		like2 = latest_likes[chainidx2]
		T2 = latest_tmps[chainidx2]
        	while chainidx2 == chainidx1 or np.isnan(like2) or T2 == T1:
        		chainidx2 = np.random.choice(chainidxs)
        		like2 = latest_likes[chainidx2]	
        		T2 = latest_tmps[chainidx2]
			
        	u = np.log(self.rstate.uniform(0, 1))
        	alpha = self.get_alpha(like1, like2, T1, T2)
		alpha = min(np.log(1), alpha)

        	if u < alpha:
        		T1idx = latest_tmpidx[int(chainidx1)]
        		T2idx =	latest_tmpidx[int(chainidx2)]
        			
            		gettmp[chainidx1] = T2
            		latest_tmpidx[chainidx1] = T2idx
            		
           		gettmp[chainidx2] = T1
       	   		latest_tmpidx[chainidx2] = T1idx
       	   	
       	   		  
        for chainidx in chainidxs:
        	tmpidx = chainidx % self.pernchains
        	self.current_tmp[tmpidx] = gettmp[chainidx]
        	sharedtmpidx[chainidx] = latest_tmpidx[chainidx]
	return chainidxs, gettmp  

    def update_currenttmp(self, iiter, gettmp):
    	for i in range(self.nchains):

    		self.sharedgettmp[iiter][i]= gettmp[i]

    	
    def run_swap(self, iiter):
    	"""
    	Before swap temperatures, make sure every chain is in the same iteration.
    	"""
	sharediiter  = np.frombuffer(self.sharediiter , dtype=self.dtype)
	currentiiter = sharediiter
	count = Counter(currentiiter)

	while count[iiter] != self.pernchains:
		sharediiter  = np.frombuffer(self.sharediiter , dtype=self.dtype)
		currentiiter = sharediiter
		count = Counter(currentiiter)
	
	chainidxs, gettmp   = self.swap_chains()
	self.update_currenttmp(iiter, gettmp)
	return gettmp
    
    def get_conv(self, chainidx):
    	return  self.sharedconvergence[chainidx]

    	
    def run_conv(self, iiter):
    	sharediiter  = np.frombuffer(self.sharediiter , dtype=self.dtype)
	currentiiter = sharediiter
	count = Counter(currentiiter)
	
	while sum(swap) != self.pernchains and len(swap)==0:
		sharediiter  = np.frombuffer(self.sharediiter , dtype=self.dtype)
		currentiiter = sharediiter
		count = Counter(currentiiter)

		
	add_iiter = self.convdia.run_conv(self.pernchains, self.chainidxs, self.iter_phase1,
			self.ncoldchains, self.sharedtmpmodels ,  self.sharedtmpnoise)
	
		
    	if add_iiter and self.nconv < 2:
    		self.nconv += 1
    		self.iterations = self.iterations + self.addiiter
		self.iter_phase1 += self.addiiter
    		for chainidx in self.chainidxs:
    			self.sharedconvergence[chainidx]= self.nconv
    			
    	elif  add_iiter :
		for chainidx in self.chainidxs:
			self.sharedconvergence[chainidx]= 3
    		logger.debug("chains are not covereged:increase iterations")
    		
    	else:
    		for chainidx in self.chainidxs:
    			self.sharedconvergence[chainidx]=0

    def run_partmp(self, dtsend ):
	gettmp = np.nan * np.ones(self.nchains)
	chainidxs = np.arange(0, self.nchains,1)
	
	initnum = 1
	while initnum <= self.ngroups:
		self.current_tmp =  copy.deepcopy(self.init_tmp)
 	       	start = (initnum -1 )*self.pernchains
 	       	end = (initnum)*self.pernchains
        	self.chainidxs = chainidxs[start:end]
		
		self.nconv = 0
		
		iiter = 1	
		
		while iiter < self.iterations-1:	
			if iiter >= self.init_swap and iiter % self.freq_swap == 0:
				gettmp = self.run_swap(iiter)
			else:		
				self.update_currenttmp(iiter, gettmp)
				
			iiter+=1
		
		initnum += 1

		
    def get_swaptmp(self, tmpiiter, chainidx):
    	return self.sharedgettmp[tmpiiter][chainidx]
    	
    def save_finalmodels(self):
    	"""Save chainmodels as pkl file"""
    	savepath = op.join(self.initparams['savepath'], 'data')
    	names = ['models', 'likes', 'misfits', 'noise', 'vpvs']
    	#finaliiter = self.iter_phase1 + self.iter_phase2
    	finaliiter =  sum(~np.isnan(self.sharedtmplikes[0]))
    	
	chainidxs = np.arange(self.nchains)
 	
        
        for tmpchainidx in chainidxs:
        
		tmpchainmodels = self.sharedtmpmodels[tmpchainidx].reshape(
            		self.ntmpmodels, self.maxlayers*3)
		tmpchainmisfits = self.sharedtmpmisfits[tmpchainidx].reshape(
            		self.ntmpmodels, self.ntargets+1)
		tmpchainlikes = self.sharedtmplikes[tmpchainidx]
		tmpchainnoise = self.sharedtmpnoise[tmpchainidx].reshape(
            		self.ntmpmodels, self.ntargets*2)
		tmpchainvpvs = self.sharedtmpvpvs[tmpchainidx]
		
		
		# update chain values (eliminate nan rows)
		tmpchainmodels = tmpchainmodels[:finaliiter, :]
		tmpchainmisfits = tmpchainmisfits[:finaliiter, :]
		tmpchainlikes = tmpchainlikes[:finaliiter]
		tmpchainnoise = tmpchainnoise[:finaliiter, :]
		tmpchainvpvs = tmpchainvpvs[:finaliiter]
        
		p1models = tmpchainmodels[:self.iter_phase1, :]
		p1misfits = tmpchainmisfits[:self.iter_phase1, :]
		p1likes = tmpchainlikes[:self.iter_phase1]
		p1noise = tmpchainnoise[:self.iter_phase1, :]
		p1vpvs = tmpchainvpvs[:self.iter_phase1]
		
		p2models = tmpchainmodels[self.iter_phase1:, :]
		p2misfits = tmpchainmisfits[self.iter_phase1:, :]
		p2likes = tmpchainlikes[self.iter_phase1:]
		p2noise = tmpchainnoise[self.iter_phase1:, :]
		p2vpvs = tmpchainvpvs[self.iter_phase1:]
		
		
        	# phase 1 -- burnin
        	try:
            		for i, data in enumerate([p1models, p1likes,
                                     		p1misfits, p1noise,
                                     		p1vpvs]):
                		outfile = op.join(savepath, 'tmp%.3d_p1%s' % (tmpchainidx, names[i]))
                		np.save(outfile, data)
                	logger.info('> Saving %d models (burinin phase).' % len(data))
        	except:
            		logger.info('No burnin models accepted.')

        	# phase 2 -- main / posterior phase
        	try:
            		for i, data in enumerate([p2models, p2likes,
                                     		 p2misfits, p2noise,
                                     		 p2vpvs]):
                		outfile = op.join(savepath, 'tmp%.3d_p2%s' % (tmpchainidx, names[i]))
                		np.save(outfile, data)

           		logger.info('> Saving %d models (main phase).' % len(data))
        	except:
            		logger.info('No main phase models accepted.')  
            


    def append_tmpmodels(self,  tmpiiter, chainidx, currentmodel,  currentmisfits,
        		currentlikelihood, currentnoise, currentvpvs):
        		
    	chaintmpdict = np.frombuffer(self.sharedtmpidx, dtype=self.dtype) 
                	      
        tmpidx = int(chaintmpdict[chainidx])
	
        tmpchainmodels = self.sharedtmpmodels[tmpidx].reshape(
            self.ntmpmodels, self.maxlayers*3)
        tmpchainmisfits = self.sharedtmpmisfits[tmpidx].reshape(
            self.ntmpmodels, self.ntargets+1)
        tmpchainlikes = self.sharedtmplikes[tmpidx]
        tmpchainnoise = self.sharedtmpnoise[tmpidx].reshape(
            self.ntmpmodels, self.ntargets*2)
        tmpchainvpvs = self.sharedtmpvpvs[tmpidx]
        
        tmpchainmodels[tmpiiter, : currentmodel.size] = currentmodel
        tmpchainmisfits[tmpiiter, :] = currentmisfits
        tmpchainlikes[tmpiiter] = currentlikelihood
        tmpchainnoise[tmpiiter, :] = currentnoise
        tmpchainvpvs[tmpiiter] = currentvpvs


        
                
  
    	
    
