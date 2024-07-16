# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:55:51 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
from time import time ; from time import sleep ; from datetime import timedelta

def kappa(k,head="None"):
    if head == "None":
        if k=="div,sc":
            return 0.025
        if k=="sc,pc":
            return 0.025
        if k=="div,pc":
            return 0.02
        if k=="pc,ent":
            return 0.075
        if k=="pc,gc":
            return 0.075
        if k=="ex,ent":
            return 0.075
        if k=="ex,gc":
            return 0.075
    elif head == "bar":
        if k=="div,sc":
            return 0.113
        if k=="div,pc":
            return 0.113
        if k=="ex,ent":
            return 0.113
        if k=="ex,gc":
            return 0.133

def K(k,head="None"):
    if head == "None":
        if k=="div,sc":
            return 0.06
        if k=="sc,pc":
            return 0.06
        if k=="div,pc":
            return 0.2
        if k=="pc,ent":
            return 0.2
        if k=="pc,gc":
            return 0.2
        if k=="ex,ent":
            return 0.95
        if k=="ex,gc":
            return 0.95
    elif head == "bar":
        if k=="div,sc":
            return 1
        if k=="div,pc":
            return 0.774
        if k=="ex,ent":
            return 0.377
        if k=="ex,gc":
            return 0.377

def R(k,X,head="None"): #R_tilde, R_bar_tilde
    R = []
    for x in X:
        if x <= K(k,head)-kappa(k,head):
            R.append(0)
        elif K(k,head)-kappa(k,head) < x < K(k,head)+kappa(k,head):
            alpha = -x**3 + 3*K(k,head)*x**2 - (3*K(k,head)**2-3*kappa(k,head)**2)*x + K(k,head)**3+2*kappa(k,head)**3-3*K(k,head)*kappa(k,head)**2
            R.append(alpha/(4*kappa(k,head)**3))
        elif K(k,head)+kappa(k,head) <= x:
            R.append(1)
        else:
            print("Erreur",x)
    return np.array(R)

q_div_sc = 1112.109
q_div_pc = 2718.488
q_ex_ent = 4201.3
q_ex_gc = 4201.3
q_sc_pc = 2471.353
q_pc_ent = 1853.515
q_pc_gc = 611.66

def f_sc(x,rho_tot,rho_sc):
    return rho_sc*q_div_sc*(1-R("div,sc",x))*(1-R("div,sc",rho_tot,"bar")) - rho_sc*q_sc_pc*R("sc,pc",x)

def f_ent(x,rho_tot,rho_pc,rho_ent):
    return rho_pc*q_pc_gc*R("pc,gc",x) - rho_ent*q_ex_ent*R("ex,ent",x)*R("ex,ent",rho_tot,"bar")

def f_gc(x,rho_tot,rho_pc,rho_gc):
    return rho_pc*q_pc_gc*R("pc,gc",x) - rho_gc*q_ex_gc*R("ex,gc",x)*R("ex,gc",rho_tot,"bar")

def f_pc(x,rho_tot,rho_pc,rho_sc):
    return rho_pc*q_div_pc*(1-R("div,pc",x))*(1-R("div,pc",rho_tot,"bar")) - rho_pc*q_pc_ent*R("pc,ent",x) - rho_pc*q_pc_gc*R("pc,gc",x) + rho_sc*q_sc_pc*R("sc,pc",x)

def phi(X):
    # r0 = 34
    # eps = 4
    # xmax = 400
    r0 = 25
    eps = 2.5
    xmax = 200
    phi = []
    for x in X:
        if x<=(r0-eps)/xmax :
            toto = (np.sqrt((xmax*x+eps)/r0*(2-(xmax*x+eps)/r0))-np.sqrt(eps/r0*(2-eps/r0)))/(1-np.sqrt(eps/r0*(2-eps/r0)))
            phi.append(toto)
        elif (r0-eps)/xmax<x<1-(r0-eps)/xmax:
            toto = 1
            phi.append(toto)
        elif 1-(r0-eps)/xmax<=x:
            toto = (np.sqrt((xmax*(1-x)+eps)/r0*(2-(xmax*(1-x)+eps)/r0))-np.sqrt(eps/r0*(2-eps/r0)))/(1-np.sqrt(eps/r0*(2-eps/r0)))
            phi.append(toto)
        else :
            print("Erreur phi")
    return np.array(phi)

def f(rho,rho_almost_tot,dx,phi_vect): #scheme function; rho_almost_tot = rho_tot-rho
    y = - 1/dx**2*( np.maximum(rho[1:-1]+rho_almost_tot[1:-1]-rho[2:]-rho_almost_tot[2:],0)*rho[1:-1]*phi_vect[1:-1] \
                  + np.minimum(rho[1:-1]+rho_almost_tot[1:-1]-rho[2:]-rho_almost_tot[2:],0)*rho[2:]*phi_vect[2:] \
                  - np.maximum(rho[:-2]+rho_almost_tot[:-2]-rho[1:-1]-rho_almost_tot[1:-1],0)*rho[:-2]*phi_vect[:-2] \
                  - np.minimum(rho[:-2]+rho_almost_tot[:-2]-rho[1:-1]-rho_almost_tot[1:-1],0)*rho[1:-1]*phi_vect[1:-1] )

    return y

def solve_crypt(sc_0,pc_0,dcs,gc_0,ent_0,nt,dt,dx,Xe,f_sc,f_pc,f_gc,f_ent,phi):
    tps = time()

    sc_mat = np.zeros((1,len(sc_0)))
    sc_mat[0,:] = sc_0
    sc_old = sc_0

    pc_mat = np.zeros((1,len(pc_0)))
    pc_mat[0,:] = pc_0
    pc_old = pc_0

    gc_mat = np.zeros((1,len(gc_0)))
    gc_mat[0,:] = gc_0
    gc_old = gc_0

    ent_mat = np.zeros((1,len(ent_0)))
    ent_mat[0,:] = ent_0
    ent_old = ent_0

    mem = 100 ; mem = min(nt,mem) #number of time step kept in memory
    list_t = [0]

    for n in range(1,nt):

        sc_new = np.zeros_like(sc_old)
        pc_new = np.zeros_like(pc_old)
        gc_new = np.zeros_like(gc_old)
        ent_new = np.zeros_like(ent_old)

        sc_new[1:-1] = sc_old[1:-1] + dt*f(sc_old,pc_old+gc_old+ent_old+dcs,dx,phi(Xe)) + dt*f_sc(Xe[1:-1],sc_old[1:-1]+pc_old[1:-1]+gc_old[1:-1]+ent_old[1:-1]+dcs[1:-1],sc_old[1:-1])
        pc_new[1:-1] = pc_old[1:-1] + dt*f(pc_old,sc_old+gc_old+ent_old+dcs,dx,phi(Xe)) + dt*f_pc(Xe[1:-1],sc_old[1:-1]+pc_old[1:-1]+gc_old[1:-1]+ent_old[1:-1]+dcs[1:-1],pc_old[1:-1],sc_old[1:-1])
        gc_new[1:-1] = gc_old[1:-1] + dt*f(gc_old,sc_old+pc_old+ent_old+dcs,dx,phi(Xe)) + dt*f_gc(Xe[1:-1],sc_old[1:-1]+pc_old[1:-1]+gc_old[1:-1]+ent_old[1:-1]+dcs[1:-1],pc_old[1:-1],gc_old[1:-1])
        ent_new[1:-1] = ent_old[1:-1] + dt*f(ent_old,sc_old+pc_old+gc_old+dcs,dx,phi(Xe)) + dt*f_ent(Xe[1:-1],sc_old[1:-1]+pc_old[1:-1]+gc_old[1:-1]+ent_old[1:-1]+dcs[1:-1],pc_old[1:-1],ent_old[1:-1])

        sc_new[0] = sc_new[1] ; sc_new[-1] = sc_new[-2]
        pc_new[0] = pc_new[1] ; pc_new[-1] = pc_new[-2]
        gc_new[0] = gc_new[1] ; gc_new[-1] = gc_new[-2]
        ent_new[0] = ent_new[1] ; ent_new[-1] = ent_new[-2]

        sc_old = sc_new
        pc_old = pc_new
        gc_old = gc_new
        ent_old = ent_new

        if (n%(nt//mem)==0) or (n==nt-1): #keep in memory some of the iterations
            print(f"\r{min(int(len(list_t)/mem*100),100)}% ",end="",flush=True) #print of the advancement in %
            sleep(0.001)
            sc_mat = np.vstack([sc_mat,sc_new])
            pc_mat = np.vstack([pc_mat,pc_new])
            gc_mat = np.vstack([gc_mat,gc_new])
            ent_mat = np.vstack([ent_mat,ent_new])
            list_t.append(n)
    print("","Upwind :",np.round(timedelta(seconds=time()-tps).total_seconds(),3),"seconds")

    return sc_mat,pc_mat,gc_mat,ent_mat,list_t

def init_sc(x):
    # return np.sqrt(np.maximum(0,1-5000*(0.05-x)**2))
    return np.sqrt(np.maximum(0,1-300*(0.0-x)**2))*0.35

def init_pc(x):
    return 0.0*x

def init_gc(x):
    return 0.0*x

def init_ent(x):
    return 0.0*x

def init_dcs(X): #dcs as in the article
    # D = 8
    # e = -0.06667
    # d = 20
    # N_dcs = 36
    # xe = 6
    # xd = 3
    # xmax = 400
    D = 12.3
    e = -0.06667
    d = 2.25
    N_dcs = 12
    xe = 6
    xd = 3
    xmax = 200
    # toto = (2*D*e*d*N_dcs)/(K("div,sc","bar")*(2*e*d*(xe-xd)+e-d))
    toto = (2*D*e*d*N_dcs)/(K("div,sc","bar")*(2*e*d*(xe-xd)+e-d))*0.045
    dcs = []
    shift = 0.1 #shift of dcs
    for x in X:
        if (d*xd-1)/(d*xmax)<x-shift<xd/xmax :
            dcs.append(toto*(1+d*(xmax*(x-shift)-xd)))
        elif xd/xmax<=x-shift<=xe/xmax:
            dcs.append(toto)
        elif xe/xmax<x-shift<(e*xe-1)/(e*xmax):
            dcs.append(toto*(1+e*(xmax*(x-shift)-xe)))
        else :
            dcs.append(0)
    return np.array(dcs)

#%% Mesh and initialisation

long = 1.0 ; coef_cfl = 0.4 ; tmax = 0.1 ; dx = 1/100 #choose the values of the cfl, tmax and dx you want
dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1) #dt following the cfl and such as the inverse is an integer
nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True) #mesh in space
nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True) #mesh in time

sc_0 = init_sc(Xe) ; pc_0 = init_pc(Xe) ; dcs = init_dcs(Xe) ; gc_0 = init_gc(Xe) ; ent_0 = init_ent(Xe) #initialisation

cfl = dt/dx**2 ; print(f"{nx+1} mesh cells, {nt-1} iterations in time, dt/dx**2 = {np.round(cfl,4)}")

#Solve
sc_mat,pc_mat,gc_mat,ent_mat,t = solve_crypt(sc_0,pc_0,dcs,gc_0,ent_0,nt,dt,dx,Xe,f_sc,f_pc,f_gc,f_ent,phi)

#%% Plot

# Initial conditions sc_0,pc_0,dcs ; graph phi
# plt.figure() ; plt.plot(Xe,sc_0,'b--',label=r"$\rho_{sc}^0$") ; plt.plot(Xe,pc_0,'r--',label=r"$\rho_{pc}^0$") ; plt.plot(Xe,dcs,'m-',label=r"$\rho_{dcs}$") ;plt.xlim(0,1.1) ; plt.ylim(0,1.3) ; plt.xlabel("x") ; plt.legend() ; plt.title("Conditions initiales "+r"$\rho_{sc}^0,\rho_{pc}^0,\rho_{dcs},$"+f" N={nx+1}")
# plt.figure() ; plt.plot(Xe,phi(Xe),label="$\phi$") ; plt.legend()

def plot(sc_mat,pc_mat,gc_mat,ent_mat,dcs,sc_0,pc_0,t):
    plt.figure()
    for n in range(len(t)):
        plt.cla()

        plt.plot(Xe,sc_mat[n,:],'b.-',label=r"$\rho_{sc}$")
        # plt.plot(Xe,sc_mat[n,:]+dcs,'b.-',label=r"$\rho_{sc}(+\rho_{dcs})$")
        if np.max(sc_0)>0:
            plt.plot(Xe,sc_0,'b--',label=r"$\rho_{sc}(t=0)$")

        # plt.plot(Xe,pc_mat[n,:],'r.-',label=r"$\rho_{pc}$")
        plt.plot(Xe,pc_mat[n,:]+dcs,'r.-',label=r"$\rho_{pc}(+\rho_{dcs})$")
        if np.max(pc_0)>0:
            plt.plot(Xe,pc_0,'r--',label=r"$\rho_{pc}(t=0)$")

        # plt.plot(Xe,gc_mat[n,:],'.-',label=r"$\rho_{gc}$")
        plt.plot(Xe,gc_mat[n,:]+dcs,'.-',label=r"$\rho_{gc}(+\rho_{dcs})$")
        if np.max(gc_0)>0:
            plt.plot(Xe,gc_0,'--',label=r"$\rho_{gc}(t=0)$")

        # plt.plot(Xe,ent_mat[n,:],'.-',label=r"$\rho_{ent}$")
        plt.plot(Xe,ent_mat[n,:]+dcs,'.-',label=r"$\rho_{ent}(+\rho_{dcs})$")
        # plt.plot(Xe,ent_mat[n,:]+gc_mat[n,:],'.-',label=r"$\rho_{ent}(+\rho_{gc})$")
        plt.plot(Xe,ent_mat[n,:]+gc_mat[n,:]+dcs,'.-',label=r"$\rho_{ent}+\rho_{gc}(+\rho_{dcs})$")
        if np.max(ent_0)>0:
            plt.plot(Xe,ent_0,'--',label=r"$\rho_{ent}(t=0)$")

        plt.plot(Xe,dcs,'m-',label=r"$\rho_{dcs}$")
        plt.plot(Xe,sc_mat[n,:]+pc_mat[n,:]+gc_mat[n,:]+ent_mat[n,:]+dcs,'k--',label="total")

        # plt.ylim(-0.03,1.3)
        plt.ylim(0.0,1.3)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Upwind, N={nx+1}, t={round(t[n]*dt,4)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,3)}")
        plt.ylabel("Densities")
        plt.pause(0.01+0.5/(n**2+1))

plot(sc_mat,pc_mat,gc_mat,ent_mat,dcs,sc_0,pc_0,t)
