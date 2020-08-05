import sys
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.optim as optim
from torch.autograd import Function
import pypower as pf
from pypower.api import case9, runpf, ppoption, printpf
import matplotlib.pyplot as plt
mpc = case9()
#print(mpc)
mpc["branch"][:,0:2] -= 1
mpc["bus"][:,0] -= 1
mpc["gen"][:,0] -= 1
#variables:
#alpha: vector of contingencies
#Pset: vector of generator power
#L: vector of lines

def makeY(mpc):
	#use this function to make the Y matrix, use MakeYbus with the correct indices
	baseMVA, br, bus = mpc["baseMVA"], mpc["branch"], mpc["bus"]
	
	Y,_,_ = pf.makeYbus.makeYbus(baseMVA, bus, br)
	return Y

# This function finds the working and out indices. Changes the power of out lines to new reduced power. Doesn't change power for working lines.
def P_modifier(P, gen, alpha_gen):
	out_indices = []
	working_indices = []
	Pmax = torch.Tensor([gen[i,8] for i in range(len(gen))])


	#get working_indices
	for index in range(len(gen)):
		if (Pmax[index] - 200)*(1 - alpha_gen[index]) <= P[index]: #(REMAINING)
			out_indices.append(index)
		else:
			working_indices.append(index)

	minimum = torch.min(P, ((Pmax - 200) * (torch.ones(len(gen)) - alpha_gen)))
	#print(P[working_indices], minimum[working_indices])
	Plost = torch.sum(P - minimum)
	
	out_indices = torch.LongTensor(out_indices)
	working_indices = torch.LongTensor(working_indices)
	return minimum, Plost, out_indices, working_indices

#Finds new power for working lines. Returns all new powers.
def P_modifier2(Pmax, Plost, minimum, alpha_gen, out_indices, working_indices):
	#if len(working_indices) == 0 or len(out_indices == 0):
	#	return
	wi = working_indices
	gamma = (torch.ones((len(wi))) - (alpha_gen[wi]) * Pmax[wi]) / torch.sum(torch.ones((len(wi))) - (alpha_gen[wi]) * Pmax[wi])
	#Plost += 100  # COMMENT OUT THE LINE. (REMAINING)
	minimum[wi] += gamma * Plost
	return minimum

#Class to represent DCPF Solver. Forward function calculates Theta
class DCPFLayer(nn.Module):
	def __init__(self, B, gen, bus, br):
		super(DCPFLayer, self).__init__()
		self.B = B
		self.gen = gen
		self.bus = bus
		self.br  = br

	def forward(self, Pnew):
		mapping = {}
		mapping = {**mapping, **dict(zip(self.gen[:,0], Pnew))}
		non_slack_indices = []
		Pnet = torch.zeros((len(self.bus)))
		for index in range(len(self.bus)):
			if self.bus[index, 1] != 3:
				non_slack_indices.append(index)
			else:
				slack_index = index
			if self.bus[index, 0] in mapping:
				Pnet[index] = mapping[self.bus[index, 0]] - self.bus[index, 2]
			else:
				Pnet[index] = - self.bus[index, 2]

		non_slack_indices = torch.LongTensor(non_slack_indices)
		B_reduced = self.B[:, non_slack_indices]
		B_reduced = B_reduced[non_slack_indices,:]
		theta = torch.zeros(len(self.br))
		theta[non_slack_indices] = (Pnet[non_slack_indices]) @ torch.inverse(B_reduced)
		return theta, non_slack_indices

# Function to calculate flows using Theta. Returns new flows
def GetFlows(theta, B, br):
	L = torch.zeros(len(br))
	L = [B[br[index,0], br[index,1]] * (theta[br[index,0]] - theta[br[index, 1]]) for index in range(len(br))]
	return(torch.Tensor(L))

#calculates loss based on infeasibility and cost terms.
def getLoss(L, Lmax, Pnew, Pmin, Pmax, alpha_line, gencost):
	costs = (Pnew**2).sum()
	# CHECK COEFFICIENTS FROM GENCOST TO GET POWER GENERATION COST (REMAINING)
	loss = (Pnew**2).sum() + \
	(torch.clamp((Pnew - (Pmax - 200)), min = 0)**2).sum() + \
	(torch.clamp((Pmin - Pnew), min = 0)**2).sum() + \
	(torch.clamp((torch.abs(L) - Lmax * (1 - alpha_line)), min = 0)**2).sum() #torch.abs required? remove 200 from Pmax #(REMAINING)
	infeas = loss - costs
	return loss, costs, infeas

def training_attack(alpha, alpha_opt, L, Lmax, Pnew, Pmin, Pmax, alpha_lines, gencost):	
	alpha_opt.zero_grad()
	loss, _,_ = getLoss(L, Lmax, Pnew, Pmin, Pmax, alpha_lines, gencost)
	loss = -loss
	loss.backward(retain_graph = True)
	alpha_opt.step()
	alpha.data.clamp_(0,1)
	if alpha.sum() > 1: alpha.data = alpha.data/alpha.sum()
	return -loss, alpha

def training_defense(P, Pnew, P_opt, L, Lmax, Pmin, Pmax, alpha_lines, gencost) :
	loss, costs, infeas = getLoss( L, Lmax, Pnew, Pmin, Pmax, alpha_lines, gencost)
	print(P, "\n")
	P_opt.zero_grad()
	loss.backward(retain_graph =True)
	P_opt.step()
	print(P, "\n")
	return loss, P, costs, infeas


def main():
	Y = makeY(mpc)

	#get susceptance matrix
	B_np = -1 * Y.imag
	B = torch.Tensor(np.matrix(B_np.toarray()))	

	#Initialize the casefile and its data here
	bus = mpc["bus"]
	gen = mpc["gen"]
	br = torch.LongTensor(mpc["branch"])
	gencost = mpc["gencost"]

	Pmax = torch.Tensor([gen[i,8] for i in range(len(gen))])
	P = torch.Tensor([gen[i,1] for i in range(len(gen))])
	Pmin = torch.Tensor([gen[i,9] for i in range(len(gen))])
	Lmax = torch.Tensor([br[i,6] for i in range(len(br))])

	#define ALPHA and NORMALIZE it.
	alpha = torch.rand(len(br) + len(gen))
	if alpha.sum() > 1: alpha.data = alpha.data/alpha.sum()
	alpha.requires_grad = True
	P.requires_grad = True

	#define OPTIMIZERS
	alpha_opt = optim.SGD([alpha], lr = 1e-4)
	P_opt = optim.SGD([P], lr = 1e-4)


	# Instantiate the DCPF solver class
	DCPF = DCPFLayer(B, gen, bus, br)

	#TRAINING BELOW
	loss_list = []
	defense_loss_list = []
	cost_list = []
	infeas_list = []

	for outer_epoch in range(10):

		for epoch in range(9):
			#print("INNER EPOCH NO = " + str(epoch + 1))
			
			# Send the gen data to the P_modifier function which will give the minimum between P and Pmax after accounting for generator contingencies.
			minimum, Plost, out_indices, working_indices = P_modifier(P, gen, alpha[len(br):])

			# Send the output of P_modifier to P_modifier2 to get pickups.
			Pnew = P_modifier2(Pmax, Plost, minimum, alpha[len(br):], out_indices, working_indices)

			#forward function on DCPF
			theta, non_slack_indices = DCPF(Pnew)

			#Get the required parameters for the loss function, flows
			L = GetFlows(theta, B, br)

			#Get the loss after optimizing.
			loss, alpha = training_attack(alpha, alpha_opt, L, Lmax, Pnew, Pmin, Pmax, alpha[:len(br)], gencost)
			loss_list.append(loss)

		outer_epoch_input_loss, _ = training_attack(alpha, alpha_opt, L, Lmax, Pnew, Pmin, Pmax, alpha[:len(br)], gencost)
		loss_list.append(outer_epoch_input_loss)
		
		
		defense_loss, P, costs, infeas = training_defense(P, Pnew, P_opt, L, Lmax, Pmin, Pmax, alpha[:len(br)], gencost)
		print(loss_list[-1], defense_loss, "\n")
		defense_loss_list.append(defense_loss)
		cost_list.append(costs)
		infeas_list.append(infeas)

	print(working_indices, out_indices)
	#print("FINAL ATTACK LOSS")
	#print(loss_list[-1], "\n")
	# Defense loss
	#print("FINAL DEFENSE LOSS")
	#print(defense_loss, "\n")
	print("OPTIMAL P")
	print(Pnew,"\n")
	
	plt.plot(range(1,101),loss_list)
	plt.xlabel("EPOCH")
	plt.ylabel("LOSS")
	plt.show()

	plt.plot(range(1,11),defense_loss_list, "r--", range(1,11), cost_list, "g^")
	plt.show()

	plt.plot(range(1,11), infeas_list)
	plt.show()
main()
