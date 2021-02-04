import sys
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.optim as optim
from torch.autograd import Function
import pypower as pf
from pypower.api import case9, case14, case30, runpf, ppoption, printpf
from pypower import idx_bus, idx_gen, idx_brch, idx_cost
import matplotlib.pyplot as plt
import ipdb
import cvxpy as cp

from model_classes import DCPFLayer

mpc = case30()
mpc["branch"][:,0:2] -= 1
mpc["bus"][:,0] -= 1
mpc["gen"][:,0] -= 1
mpc["gen"][:,8] *= 1


# VARIABLES
# ALPHA: vector of contingencies
# PSET: vector of generator power
# L: Vector of line flows

# CONSTANTS
OUTER_RANGE = 50
MAX_INNER_RANGE = 10000
LR_P = 1e-1
LR_ALPHA = 1e-1
SF = 5
BF = 10
USE_BREAKPOINTS = False
ALPHA_BREAKPOINTS = False

def main():

    ## Initialize casefile data
    B = makeY(mpc)  
    bus = mpc["bus"]
    gen = mpc["gen"]
    br = torch.LongTensor(mpc["branch"])
    gencost = mpc["gencost"]
    gencost[:, 4] = gencost[:, 4] * 2  # TODO: perhaps change back later

    output = pf.rundcopf.rundcopf(mpc)
    P = torch.Tensor(output["gen"][:, idx_gen.PG])
    Pmax = torch.Tensor(gen[:, idx_gen.PMAX])
    Pmin = torch.Tensor(gen[:, idx_gen.PMIN])
    P.requires_grad = True
    # P_opt = optim.SGD([P], lr = LR_P)

    Lmax = torch.Tensor(br[:, idx_brch.RATE_B].float())

    ## Initialize DC power flow solver
    DCPF = DCPFLayer(mpc, B, gen, bus, br)

    ## Training
    defense_losses = []
    attack_losses = []
    start_alphas = []
    end_alphas = []

    for outer_epoch in range(OUTER_RANGE):
        print("DEFENSE EPOCH {}".format(outer_epoch))
        # print('THIS IS P: {}'.format(P))

        ## Initialize and project alpha
        alpha = torch.rand(len(br) + len(gen), requires_grad=True)
        project_alpha_inplace(alpha)
        # alpha_opt = optim.SGD([alpha], lr = LR_ALPHA)
        start_alphas.append(alpha.detach().numpy())

        ## Construct worst-case contingency (attack)
        alpha_prev = alpha.detach().clone()
        inner_losses = []
        for epoch in range(MAX_INNER_RANGE):
            print("ATTACK EPOCH {}".format(epoch))
        
            # Zero gradients
            # alpha_opt.zero_grad()
            # P_opt.zero_grad()
            alpha.grad = None
            P.grad = None

            # Calculate generation and flows
            Pnew = apply_gen_cont_pickup(P, gen, alpha[-len(gen):])
            theta, Pnew_postflow = DCPF(Pnew)
            L, outage_lines = get_flows(theta, B, br, Lmax, alpha[:len(br)])

            # Calculate loss and update alpha
            loss, _ , _ = get_contingency_loss(L, Lmax, Pnew_postflow, Pmin, Pmax, alpha[-len(gen):], alpha[:len(br)], gencost)
            # (-loss).register_hook(lambda grad: print(grad))
            (-loss).backward()
            
            # Take l1-norm bounded step in alpha
            # alpha_opt.step()
            delta_norm = 0.1  # TODO: could be changed/tuned
            grad_norm = torch.sum(torch.abs(alpha.grad))
            if grad_norm > delta_norm:
                alpha.data -= alpha.grad*(delta_norm/grad_norm)
            else:
                alpha.data -= alpha.grad
            project_alpha_inplace(alpha)

            inner_losses.append(loss.data.item())

            # stop if alpha has converged
            #  TODO: probably should change stopping criteria to %age of loss change
            print(torch.norm(alpha - alpha_prev))
            print(loss)
            if torch.norm(alpha - alpha_prev) < 1e-5:
                break

            alpha_prev = alpha.detach().clone()

        attack_losses.append(inner_losses)
        end_alphas.append(alpha.detach().numpy())


        ## Take a defense step
        print("DEFENSE STEP FOR DEFENSE EPOCH {}".format(outer_epoch))
        # alpha_opt.zero_grad()
        # P_opt.zero_grad()
        alpha.grad = None
        P.grad = None

        # Calculate generation and flows
        Pnew = apply_gen_cont_pickup(P, gen, alpha[-len(gen):])
        theta, Pnew_postflow = DCPF(Pnew)
        L, outage_lines = get_flows(theta, B, br, Lmax, alpha[:len(br)])

        # Calculate loss and update alpha
        cont_loss, cont_costs, cont_infeas = get_contingency_loss(L, Lmax, Pnew, Pmin, Pmax, alpha[-len(gen):], alpha[:len(br)], gencost)
        base_loss, base_costs, base_infeas = get_base_loss(L, Lmax, P, Pmin, Pmax, gencost)
        
        loss = cont_loss + base_loss
        # loss.register_hook(lambda grad: print(grad))
        loss.backward()

        # Take l1 norm bounded step in P
        # P_opt.step()
        delta_norm = 5 # TODO: could be changed/tuned
        grad_norm = torch.sum(torch.abs(P.grad))
        if grad_norm > delta_norm:
            P.data -= P.grad*(delta_norm/grad_norm)
        else:
            P.data -= P.grad

        # Store loss values
        defense_losses.append(dict([('loss', loss.data.item()), 
            ('cont_loss', cont_loss.data.item()), ('cont_costs', cont_costs.data.item()), ('cont_infeas', cont_infeas.data.item()),
            ('base_loss', base_loss.data.item()), ('base_costs', base_costs.data.item()), ('base_infeas', base_infeas.data.item())]))

        print("outage lines after defense loss: {}".format(outage_lines))
        print("defense loss: {}".format(loss))
        print("alpha after end of outer EPOCH: {}".format(alpha))
    
    
    ## Plots

    # See first 6 alpha curves
    fig, axes = plt.subplots(2,3)
    for i, ax in enumerate(axes.flatten()):
        if i > OUTER_RANGE - 1:
            break
        ax.plot(range(len(attack_losses[i])), attack_losses[i])
    plt.show()

    # Total objective function
    plt.plot(range(len(defense_losses)), [x['loss'] for x in defense_losses], "r--")
    plt.title('Total loss')
    plt.show()

    # Total power costs
    plt.plot(range(len(defense_losses)), [x['cont_costs'] + x['base_costs'] for x in defense_losses], "g-")
    plt.title('Power costs')
    plt.show()

    # Total infeasibility costs
    plt.plot(range(len(defense_losses)), [x['cont_infeas'] + x['base_infeas'] for x in defense_losses])
    plt.title('Infeasibility costs')
    plt.show()


# MAKES ADMITTANCE MATRIX
def makeY(mpc):
    baseMVA, br, bus = mpc["baseMVA"], mpc["branch"], mpc["bus"]
    B_np = pf.makeB.makeB(baseMVA, bus, br, alg = 3)
    B = torch.Tensor(np.array(B_np[0].toarray()))
    # Y,_,_ = pf.makeYbus.makeYbus(baseMVA, bus, br)
    return B

# APPLY GENERATION CONTINGENCY AND PICKUPS
def apply_gen_cont_pickup(P, gen, alpha_gen):

    Pmax = torch.Tensor(gen[:, idx_gen.PMAX])
    Pmax_contingency = Pmax * (1-alpha_gen)

    # Calculate outages
    has_outage = (Pmax_contingency <= P)
    # out_indices = torch.where(has_outage)[0]
    working_indices = torch.where(~has_outage)[0]

    Pnew = torch.min(P, Pmax_contingency)
    Plost_total = torch.sum(P - Pnew)

    # Calculate pickup
    gamma = Pmax[working_indices] / torch.sum(Pmax[working_indices])
    Pnew[working_indices] += gamma * Plost_total

    return Pnew

def get_flows(theta, B, br, Lmax, alpha_lines):
    # line flows: B_ft * (\theta_f - \theta_t)
    # TODO: should this indeed be negated?
    L = -B[br[:, idx_brch.F_BUS], br[:, idx_brch.T_BUS]] * \
        (theta[br[:, idx_brch.F_BUS]] - theta[br[:, idx_brch.T_BUS]])

    # determine which lines have outages
    has_outage = Lmax*(1-alpha_lines) < torch.abs(L)
    outage_lines = torch.where(has_outage)[0]

    return L, outage_lines

def get_contingency_loss(L, Lmax, Pnew, Pmin, Pmax, alpha_gen, alpha_line, gencost):
    Pmax_contingency = Pmax * (1-alpha_gen)
    Lmax_contingency = Lmax * (1-alpha_line)
    return get_base_loss(L, Lmax_contingency, Pnew, Pmin, Pmax_contingency, gencost)

def get_base_loss(L, Lmax, P, Pmin, Pmax, gencost):
    costs = get_power_cost(P, gencost)
    infeas = ((P - (Pmin + Pmax)/2)**2).sum() + ((L - Lmax)**2).sum()
    loss = costs + infeas
    return loss, costs, infeas

def get_power_cost(P, gencost):
    assert((gencost[:, idx_cost.MODEL] == 2).all())  # polynomial cost
    num_exp = int(gencost[:, idx_cost.NCOST].max())

    costs = 0
    for i in range(num_exp):
        costs += torch.Tensor(gencost[:, idx_cost.COST + i]) * P**(num_exp-i-1)

    return costs.sum()

def project_alpha_inplace(alpha):
    # print(alpha)
    n = len(alpha)
    x = cp.Variable(n)
    objective_func = cp.Minimize(cp.norm((x - alpha.detach()), 2))
    constraints = [0 <= x, sum(x) <= 1]
    problem = cp.Problem(objective_func, constraints)
    result = problem.solve()
    alpha.data = torch.as_tensor(x.value).float()


if __name__ == '__main__':
    main()
