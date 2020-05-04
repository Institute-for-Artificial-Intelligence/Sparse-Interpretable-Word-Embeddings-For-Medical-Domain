import torch
from torch import nn
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.INFO)


class SPINEModel(torch.nn.Module):

	def __init__(self, params):
		super(SPINEModel, self).__init__()
		
		# params
		self.inp_dim = params['inp_dim']
		self.hdim = params['hdim']
		self.noise_level = params['noise_level']
		self.RL_coef = params['RL_coef']
		self.PSL_coef = params['PSL_coef']
		self.SimLoss_coef = params['simloss_coef']
		self.getReconstructionLoss = nn.MSELoss()
		self.getSIMLoss = nn.MSELoss()
		self.rho_star = 1.0 - params['sparsity']
		
		# autoencoder
		logging.info("Building model ")
		self.linear1 = nn.Linear(self.inp_dim, self.hdim)
		self.linear2 = nn.Linear(self.hdim, self.inp_dim)
		

	def forward(self, batch_x, batch_y):
		
		# forward
		batch_size = batch_x.data.shape[0]
		linear1_out = self.linear1(batch_x)
		h = linear1_out.clamp(min=0, max=1) # capped relu
		out = self.linear2(h)

		# different terms of the loss
		reconstruction_loss = self.RL_coef * self.getReconstructionLoss(out, batch_y) # reconstruction loss
		psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
		asl_loss = self._getASLLoss(h)    	# average sparsity loss
		sparsity_ratio = self._sparsity(h, batch_size)
		simLoss = self._getSIMLoss(h, batch_x, batch_size)
		total_loss = reconstruction_loss + psl_loss + asl_loss + simLoss
		
		return out, h, total_loss, [reconstruction_loss,psl_loss, asl_loss, simLoss], sparsity_ratio


	def _getPSLLoss(self, h, batch_size):
		return self.PSL_coef * torch.sum(h*(1-h)) / (batch_size * self.hdim)


	def _getASLLoss(self, h):
		temp = torch.mean(h, dim=0) - self.rho_star
		temp = temp.clamp(min=0)
		return torch.sum(temp * temp) / self.hdim

	def _sparsity(self, h, batch_size):
		return torch.sum(1-h[h==0])/ (batch_size * self.hdim)
		
			
	def _getSIMLoss(self, h, batch_x, batch_size):
		temp1 = batch_x / torch.norm(batch_x, p=2, dim=1).reshape(batch_size, 1)	
		temp2  = h / torch.norm(h, p=2, dim=1).reshape(batch_size, 1)
		temp1 = torch.mm(temp1, torch.t(temp1))
		temp2 = torch.mm(temp2, torch.t(temp2))
		simloss = self.getSIMLoss(temp1, temp2)
		return self.SimLoss_coef * simloss
		

		






