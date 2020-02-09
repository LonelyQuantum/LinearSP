# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from gurobipy import *
import numpy as np


# +
#Function to print results of an optimization problem
def printResult(m):
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('\nMinimal cost %g \n' % m.objVal)

##L-shaped algorithm
#remaining things to address
# 1. m2 == 0, n2 == 0, n1 == 0
#
def LShapedAlgo(c, q, A, b, W, T, h, p):
    n1 = len(c)
    m1 = len(b)
    n2 = len(q)
    m2 = W.shape[0]
    K = len(p)
    
    #Initialization
    optimal = False
    r = 0
    s = 0
    nu = 0
    
    #Create master problem
    master = Model('master problem')
    master.setParam('OutPutFlag', False)
    x = master.addMVar(n1, lb=0, name='x')
    master.setObjective(c @ x, GRB.MINIMIZE)
    if m1 > 0:
        master.addConstr(A @ x == b)
    theta_val = float("-inf")
    
    #L-shaped iterations
    while(not optimal):
        infease = False
        nu += 1
        master.optimize()
        if s >= 1:
            theta_val = theta.x
        print('\nmaster problem %d solved' % nu)
        print('x = ')
        print(master.x)
        print('theta = ')
        print(theta_val)

        x_cur = x.x
        #Feasibility cuts
        for k in range(K):
            hk = h[k]
            Tk = T[k]
            fcut = Model('feasibilityCut')
            fcut.setParam('OutPutFlag', False)
            y = fcut.addMVar(n2, lb=0, name='y')
            vp = fcut.addMVar(m2, lb=0, name='vplus')
            vm = fcut.addMVar(m2, lb=0, name='vminus')
            e = np.ones(m2)
            fcut.setObjective(e @ vp + e @ vm)
            fcut.addConstr(W @ y + vp - vm == hk - np.dot(Tk, x_cur))
            fcut.optimize()
            if fcut.objVal > 0:
                r += 1
                pi = np.array(fcut.pi)
                D = np.dot(pi,T)
                d = np.dot(pi,hk)
                print('feasibility cut %d created with\n D =' % r)
                print(D)
                print('d=%f'%d)
                master.addConstr(D @ x >= d)
                infease = True
                break
        if infease:
            continue

        ##Optimality cuts
        E = np.zeros(n1)
        e = 0
        for k in range(K):
            hk = h[k]
            Tk = T[k]
            ocut = Model('optimalityCut')
            ocut.setParam('OutPutFlag', False)
            y = ocut.addMVar(n2,lb=0, name='y')
            ocut.setObjective(q @ y, GRB.MINIMIZE)
            ocut.addConstr(W @ y == hk - np.dot(Tk,x_cur))
            ocut.optimize()            
            pi = np.array(ocut.pi) 
            E += p[k]*np.dot(pi,Tk)
            e += p[k]*np.dot(pi,hk)
        if theta_val >= e - np.dot(E,x_cur):
            optimal = True
            print('theta: %f >= e-E\'X = %f ' % (theta_val, e - np.dot(E,x_cur)))
            print('optimality reached')
            printResult(master)
        else:
            s += 1
            if s == 1:
                theta = master.addMVar(1,lb=-GRB.INFINITY, name='theta')
                master.setObjective(c @ x + theta, GRB.MINIMIZE)
            master.addConstr(E @ x + theta >= e)
            print('optimality cut %d created with \n E = ' % s)
            print(E)
            print('e = %f' % e)
