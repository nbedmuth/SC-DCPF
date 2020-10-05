import sys
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.optim as optim
from torch.autograd import Function
import pypower as pf
from pypower.api import case9, case14, case30, runpf, ppoption, printpf
import matplotlib.pyplot as plt
import ipdb
import cvxpy as cp
mpc = case30()
mpc["branch"][:,0:2] -= 1
mpc["bus"][:,0] -= 1
mpc["gen"][:,0] -= 1
mpc["gen"][:,8] *= 1


#VARIABLES
#ALPHA: vector of contingencies
#PSET: vector of generator power
#L: Vector of line flows


#CONSTANTS
OUTER_RANGE = 100
INNER_RANGE = 20
LR_P = 1e-1
LR_ALPHA = 1e-1
SF = 5
BF = 10
USE_BREAKPOINTS = False
ALPHA_BREAKPOINTS = False

#MAKES ADMITTANCE MATRIX
def makeY(mpc):
	baseMVA, br, bus = mpc["baseMVA"], mpc["branch"], mpc["bus"]
	B_np = pf.makeB.makeB(baseMVA, bus, br, alg = 3)
	B = torch.Tensor(np.array(B_np[0].toarray()))
	Y,_,_ = pf.makeYbus.makeYbus(baseMVA, bus, br)
	return Y, B


#FUNCTION TO FIND WORKING AND OUT INDICES. REDUCED OUTAGE GENERATION.
def P_modifier(P, gen, alpha_gen):

	if USE_BREAKPOINTS: 
		ipdb.set_trace()
		print("current P: {}".format(P))
		print("Pmax: {}".format(Pmax))
		print("alpha generators: {}".format(alpha_gen))

	out_indices = []
	working_indices = []

	Pmax = torch.Tensor([gen[i,8] for i in range(len(gen))])
	for index in range(len(gen)):
		if (Pmax[index])*(1 - alpha_gen[index]) <= P[index]: #(REMAINING)
			out_indices.append(index)
		else:
			working_indices.append(index)

	minimum = torch.min(P, ((Pmax) * (torch.ones(len(gen)) - alpha_gen)))
	Plost = torch.sum(P - minimum)

	out_indices = torch.LongTensor(out_indices)
	working_indices = torch.LongTensor(working_indices)

	return minimum, Plost, out_indices, working_indices



#FINDS NET POWER FOR WORKING LINES. RETURNS PNEW
def P_modifier2(Pmax, Plost, minimum, alpha_gen, out_indices, working_indices):
	wi = working_indices
	gamma = Pmax[wi] / torch.sum(Pmax[wi])
	minimum[wi] += gamma * Plost
	return minimum


#DCPF SOLVER CLASS. FORWARD RETURNS THETA
class DCPFLayer(nn.Module):
	def __init__(self, mpc, B, gen, bus, br):
		super(DCPFLayer, self).__init__()
		self.B = B
		self.gen = gen
		self.bus = bus
		self.br  = br
		self.mpc = mpc
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
		if USE_BREAKPOINTS: 
			ipdb.set_trace()
			print("Pnet: {}".format(Pnet))
		
		non_slack_indices = torch.LongTensor(non_slack_indices)
		B_reduced = self.B[:, non_slack_indices]
		B_reduced = B_reduced[non_slack_indices,:]

		theta = torch.zeros(len(self.bus))
		theta[non_slack_indices] = torch.inverse(B_reduced) @ (Pnet[non_slack_indices])
		if USE_BREAKPOINTS: 
			ipdb.set_trace()
			print("theta : {}".format(theta))

		Pnet[slack_index] = self.B[slack_index,:] @ theta
		Pnew[slack_index] = Pnet[slack_index] + self.bus[slack_index, 2]
		return theta, non_slack_indices, Pnew




def GetFlows(theta, B, br, Lmax, alpha_lines):
	outage_lines = []
	L = torch.zeros(len(br))
	L = [-B[br[index,0], br[index,1]] * (theta[br[index,0]] - theta[br[index, 1]]) for index in range(len(br))]
	for index in range(len(br)):
		if Lmax[index]*(1 - alpha_lines[index]) < torch.abs(L[index]):
			outage_lines.append(index)

	return(torch.Tensor(L)), outage_lines


def getLoss(L, Lmax, Pnew, Pmin, Pmax, alpha_gen, alpha_line, gencost):
	#COSTS
	highest_power = int(gencost[0,3] - 1)
	costs = 0
	for i in range(highest_power + 1):
		costs += (torch.Tensor(gencost[:, 4 + i]) * (Pnew**(highest_power - i))).sum()	
	print("costs: {}".format(costs))


	#LOSS
	print("feasibility loss: {}".format((SF * torch.clamp((1/(Pmax - Pnew)) * alpha_gen, min = 0, max = 1000).sum()) + SF * (torch.clamp((1/(Lmax - torch.abs(L)) * alpha_line), min = 0, max = 1000)).sum()))
	loss = costs + \
	BF * torch.clamp(Pnew - Pmax * (1 - alpha_gen), min = 0).sum() + \
	BF * torch.clamp(Pmin - Pnew, min = 0).sum()  + \
	SF * torch.clamp((1/(Pmax - Pnew)) * alpha_gen, min = 0, max = 1000).sum() + \
	BF * (torch.clamp((torch.abs(L) - Lmax * (1 - alpha_line)), min = 0)).sum() + \
	SF * (torch.clamp((1/(Lmax - torch.abs(L)) * alpha_line), min = 0, max = 1000)).sum()
	print("total loss: {}".format(loss))

	#INFEAS
	infeas = BF * torch.clamp(Pnew - Pmax * (1 - alpha_gen), min = 0).sum() + \
	BF * torch.clamp(Pmin - Pnew, min = 0).sum() + \
	BF * (torch.clamp((torch.abs(L) - Lmax * (1 - alpha_line)), min = 0)).sum()
	print("infeas: {}".format(infeas))
	#SF * torch.clamp(Pnew - Pmin, min = 0).sum() + \
	#ipdb.set_trace()
	return loss, costs, infeas

def baseLoss(L, Lmax, Pnew, Pmin, Pmax, gencost):
	highest_power = int(gencost[0,3] - 1)
	costs = 0
	for i in range(highest_power + 1):
		costs += (torch.Tensor(gencost[:, 4 + i]) @ (Pnew**(highest_power - i))).sum()

	ipdb.set_trace()

	loss = costs + \
	BF * torch.clamp(Pnew - Pmax, min = 0).sum() + \
	BF * torch.clamp(torch.abs(L) - Lmax, min = 0).sum()

	return loss

def training_attack(epoch, alpha, alpha_opt, L, Lmax, Pnew, Pmin, Pmax,alpha_gen, alpha_lines, gencost):	
	alpha_prev = alpha.detach()
	alpha_opt.zero_grad()
	loss, _,_ = getLoss(L, Lmax, Pnew, Pmin, Pmax,alpha_gen, alpha_lines, gencost)
	(-loss).register_hook(lambda grad: print(grad))
	(-loss).backward(retain_graph =True)

	
	if ALPHA_BREAKPOINTS: 
		ipdb.set_trace()
		print("alpha before step: {}".format(alpha), "\n")
	alpha_opt.step()


	if ALPHA_BREAKPOINTS: 
		ipdb.set_trace()
		print("alpha_grad: {}".format(alpha.grad.data), "\n")
		print("alpha after step: {}".format(alpha), "\n")
		#ipdb.set_trace()


	#CVXPY
	if (epoch + 1) % 10 == 0:
		n = len(alpha)
		x = cp.Variable(n)
		objective_func = cp.Minimize(cp.norm((x - alpha.detach()), 2))
		constraints = [0 <= x, sum(x) <= 1]
		problem = cp.Problem(objective_func, constraints)
		result = problem.solve()
		alpha.data = torch.as_tensor(x.value).float()



		if ALPHA_BREAKPOINTS: 
			ipdb.set_trace()
			print("alpha after clamping: {}".format(alpha), "\n")
	return loss, alpha, alpha_prev



def training_defense(P, Pnew, P_opt, L, Lmax, Pmin, Pmax,alpha_gen, alpha_lines, gencost) :
	P_opt.zero_grad()
	print("inside training defense:", "\n")
	SF = 0
	alpha_loss, costs, infeas = getLoss(L, Lmax, Pnew, Pmin, Pmax, alpha_gen, alpha_lines, gencost)
	base_loss = baseLoss(L, Lmax, Pnew, Pmin, Pmax, gencost)
	loss2 = alpha_loss + base_loss


	print("defense loss before: alpha loss: {}, base_loss: {}".format(alpha_loss, base_loss))
	(loss2).register_hook(lambda grad: print(grad))
	loss2.backward()



	print("P_grad: {}".format(P.grad.data), "\n")
	print("P before: {}".format(P))
	P_opt.step()
	print("P after: {}".format(P))
	ipdb.set_trace()
	return loss2, P, costs, infeas
###########################################################################################################################################################

def main():

	Y, B = makeY(mpc)	
	#INITIALIZE CASEFILE DATA
	bus = mpc["bus"]
	gen = mpc["gen"]
	br = torch.LongTensor(mpc["branch"])
	gencost = mpc["gencost"]


	output = pf.rundcopf.rundcopf(mpc)
	P = torch.Tensor(output["gen"][:,1])
	Pmax = torch.Tensor([gen[i,8] for i in range(len(gen))])
	Pmin = torch.Tensor([gen[i,9] for i in range(len(gen))])
	P.requires_grad = True
	P_opt = optim.SGD([P], lr = LR_P)


	Lmax = torch.Tensor([br[i,6] for i in range(len(br))])


	#INSTANTIATE THE DCPF SOLVER CLASS OBJECT
	DCPF = DCPFLayer(mpc, B, gen, bus, br)

	#TRAINING BELOW
	defense_loss_list = []
	cost_list = []
	infeas_list = []
	overall_list = []
	plist = []
	pnewlist = []

#START OF LOOP		
#############################################################################################################################################################
	for outer_epoch in range(OUTER_RANGE):

		#DEFINE ALPHA and NORMALIZE IT.
		alpha = torch.rand(len(br) + len(gen))
		if alpha.sum() > 1: alpha.data = alpha.data / alpha.sum()
		alpha.requires_grad = True
		alpha_opt = optim.SGD([alpha], lr = LR_ALPHA)
		loss_list = []
		print("alpha initiliazation : {}".format(alpha))

#START OF INNER LOOP	
#############################################################################################################################################################
		for epoch in range(INNER_RANGE):
		
			minimum, Plost, out_indices, working_indices = P_modifier(P, gen, alpha[len(br):])	
			Pnew = P_modifier2(Pmax, Plost, minimum, alpha[len(br):], out_indices, working_indices)
			theta, non_slack_indices, Pnew = DCPF(Pnew)
			L, outage_lines = GetFlows(theta, B, br, Lmax, alpha[:len(br)])
			P_opt.zero_grad()
			loss, alpha, alpha_prev = training_attack(epoch, alpha, alpha_opt, L, Lmax, Pnew, Pmin, Pmax,alpha[len(br):], alpha[:len(br)], gencost)


#END OF INNER LOOP, FINAL ATTACK LOSS ADDED TO LOSS LIST		
#############################################################################################################################################################
		minimum, Plost, out_indices, working_indices = P_modifier(Pnew, gen, alpha[len(br):])
		Pnew = P_modifier2(Pmax, Plost, minimum, alpha[len(br):], out_indices, working_indices)
		theta, non_slack_indices, Pnew = DCPF(Pnew)
		L,_ = GetFlows(theta, B, br, Lmax, alpha[:len(br)])
		#loss_list.append(getLoss(L, Lmax, Pnew, Pmin, Pmax,alpha[len(br):], alpha[:len(br)], gencost)[0].data.item())
		#overall_list.append(loss_list)
		print("Start of Outer Epoch: {}".format(outer_epoch + 1))

#DEFENSE LOSS
#############################################################################################################################################################
		defense_loss, P, costs, infeas = training_defense(P, Pnew, P_opt, L, Lmax, Pmin, Pmax, alpha[len(br):], alpha[:len(br)], gencost)
		# Should this P go as input to the next outr epoch ?
		minimum, Plost, out_indices, working_indices = P_modifier(P, gen, alpha[len(br):])
		Pnew = P_modifier2(Pmax, Plost, minimum, alpha[len(br):], out_indices, working_indices)
		theta, non_slack_indices, Pdef = DCPF(Pnew)
		L,outage_lines = GetFlows(theta, B, br, Lmax, alpha[:len(br)])
		defense_loss, costs, _ = getLoss(L, Lmax, Pnew, Pmin, Pmax,alpha[len(br):], alpha[:len(br)], gencost)
		#P = Pnew # required or not ? 

		P_opt.zero_grad()
		alpha_opt.zero_grad()

		defense_loss_list.append(defense_loss.data.item())
		cost_list.append(costs.data.item())
		infeas_list.append(infeas.data.item())

		#print(working_indices, out_indices)
		print("outage lines after defense loss: {}".format(outage_lines))
		print("defense loss: {}".format(defense_loss))
		print("alpha after end of outer EPOCH: {}".format(alpha))

	#PLOTS
	#fig, axs = plt.subplots(5)
	#fig.suptitle("Maximization loss vs EPOCH")
	#for i in range(5):
	#	axs[i].plot(range(1, len(overall_list[2*i]) + 1), overall_list[2*i])
	#plt.show()

	#TOTAL OBJECTIVE FUNCTION & COSTS
	plt.plot(range(1,OUTER_RANGE + 1),defense_loss_list, "r--")
	plt.show()

	plt.plot(range(1,OUTER_RANGE + 1), cost_list, "g^")
	plt.show()

	#INFEASIBILITY COSTS
	plt.plot(range(1,OUTER_RANGE + 1), infeas_list)
	plt.show()


main()
