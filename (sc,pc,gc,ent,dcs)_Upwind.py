# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:55:51 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
from time import time ; from time import sleep ; from datetime import timedelta

def kappa(i):
    if i==1:
        return 0.0125
    if i==2:
        return 6
    if i==3:
        return 0.1
    if i==4:
        return 6
    if i==5:
        return 0.0375

def K(i):
    if i==1:
        return 0.06
    if i==2:
        return 53
    if i==3:
        return 0.275
    if i==4:
        return 41
    if i==5:
        return 0.95

def R(i,X):
    R = []
    for x in X:
        if x<=K(i)-kappa(i):
            R.append(0)
        elif K(i)-kappa(i)<x<K(i)+kappa(i):
            alpha = -x**3 + 3*K(i)*x**2 - (3*K(i)**2-3*kappa(i)**2)*x + K(i)**3+2*kappa(i)**3-3*K(i)*kappa(i)**2
            R.append(alpha/(4*kappa(i)**3))
        elif K(i)+kappa(i)<=x:
            R.append(1)
        else:
            print("Erreur",x)
    return np.array(R)

def f1(x,utot):
    q1 = 4822.152 ; q2 = 6429.536
    return q1*(1-R(1,x))*(1-R(2,K(2)*(utot))) - q2*R(1,x)

def f2(x,utot):
    q3 = 7072.489 ; q4 = 10930.211
    return q3*(1-R(3,x))*(1-R(4,K(2)*(utot))) - q4*R(5,x)*R(4,K(2)*(utot))

q2 = 6429.536

def phi(X):
    r0 = 34
    eps = 4
    xmax = 400
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

def f(p,pp,dx,phi_vect): #pp = ptot-p
    y = - 1/dx**2*( np.maximum(p[1:-1]+pp[1:-1]-p[2:]-pp[2:],0)*p[1:-1]*phi_vect[1:-1] \
                  + np.minimum(p[1:-1]+pp[1:-1]-p[2:]-pp[2:],0)*p[2:]*phi_vect[2:] \
                  - np.maximum(p[:-2]+pp[:-2]-p[1:-1]-pp[1:-1],0)*p[:-2]*phi_vect[:-2] \
                  - np.minimum(p[:-2]+pp[:-2]-p[1:-1]-pp[1:-1],0)*p[1:-1]*phi_vect[1:-1] )

    return y

def solve_crypt(sc_0,pc_0,dcs,gc_0,ent_0,nt,dt,dx,Xe,f1,f2,phi):
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

    mem = 100 ; mem = min(nt,mem)
    list_t = [0]

    for n in range(1,nt):

        sc_new = np.zeros_like(sc_old)
        pc_new = np.zeros_like(pc_old)
        gc_new = np.zeros_like(gc_old)
        ent_new = np.zeros_like(ent_old)

        sc_new[1:-1] = sc_old[1:-1] + dt*f(sc_old,pc_old+dcs,dx,phi(Xe)) + dt*sc_old[1:-1]*f1(Xe[1:-1],sc_old[1:-1]+pc_old[1:-1]+dcs[1:-1])
        pc_new[1:-1] = pc_old[1:-1] + dt*f(pc_old,sc_old+dcs,dx,phi(Xe)) + dt*sc_old[1:-1]*q2*R(1,Xe[1:-1]) + dt*pc_old[1:-1]*f2(Xe[1:-1]+sc_old[1:-1]+pc_old[1:-1],dcs[1:-1])
        # gc_new[1:-1] = gc_old[1:-1] + dt*f(gc_old,sc_old+pc_old+dcs,dx,phi(Xe)) +
        # ent_new[1:-1] = ent_old[1:-1] + dt*f(ent_old,sc_old+pc_old+dcs,dx,phi(Xe)) +

        #f(...,...+gc+ent) #!!!

        sc_new[0] = sc_new[1] ; sc_new[-1] = sc_new[-2]
        pc_new[0] = pc_new[1] ; pc_new[-1] = pc_new[-2]
        # gc_new[0] = gc_new[1] ; gc_new[-1] = gc_new[-2]
        # ent_new[0] = ent_new[1] ; ent_new[-1] = ent_new[-2]

        sc_old = sc_new
        pc_old = pc_new
        gc_old = gc_new
        ent_old = ent_new

        if (n%(nt//mem)==0) or (n==nt-1): #keep in memeory some of the iterations
            print(f"\r{int(len(list_t)/mem*100)}% ",end="",flush=True) #print of the advancement in %
            sleep(0.001)
            sc_mat = np.vstack([sc_mat,sc_new])
            pc_mat = np.vstack([pc_mat,pc_new])
            gc_mat = np.vstack([gc_mat,gc_new])
            ent_mat = np.vstack([ent_mat,ent_new])
            list_t.append(n)
    print("","Upwind :",np.round(timedelta(seconds=time()-tps).total_seconds(),3),"seconds")

    return sc_mat,pc_mat,gc_mat,ent_mat,list_t

def init_sc(x):
    # return np.sqrt(np.maximum(0,1-64*(1/4-x)**2))
    # return np.sqrt(np.maximum(0,1-5000*(0.05-x)**2))
    return np.sqrt(np.maximum(0,1-150*(0.0-x)**2))*0.35

def init_pc(x):
    # return np.sqrt(np.maximum(0,1-64*(3/4-x)**2))*0.6
    return 0.0*x

def init_gc(x):
    return 0.0*x

def init_ent(x):
    return 0.0*x

def init_dcs(X): #dcs as in the article
    D = 8
    e = -0.06667
    d = 20
    N_dcs = 36
    xe = 6
    xd = 3
    xmax = 400
    toto = (2*D*e*d*N_dcs)/(K(2)*(2*e*d*(xe-xd)+e-d))
    dcs = []
    decal = 0.1
    for x in X:
        if (d*xd-1)/(d*xmax)<x-decal<xd/xmax :
            dcs.append(toto*(1+d*(xmax*(x-decal)-xd)))
        elif xd/xmax<=x-decal<=xe/xmax:
            dcs.append(toto)
        elif xe/xmax<x-decal<(e*xe-1)/(e*xmax):
            dcs.append(toto*(1+e*(xmax*(x-decal)-xe)))
        else :
            dcs.append(0)
    return np.array(dcs)

#%% Mesh and initialisation

long = 1.0 ; coef_cfl = 0.4 ; tmax = 0.01 ; dx = 1/250 #choose the values of the cfl, tmax and dx you want
dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1) #dt following the cfl and such as the inverse is an integer
nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

sc_0 = init_sc(Xe) ; pc_0 = init_pc(Xe) ; dcs = init_dcs(Xe) ; gc_0 = init_gc(Xe) ; ent_0 = init_ent(Xe) #initialisation

cfl = dt/dx**2 ; print(f"{nx+1} mesh cells, {nt-1} iterations in time, dt/dx**2 = {np.round(cfl,4)}")

#Solve
sc_mat,pc_mat,gc_mat,ent_mat,t = solve_crypt(sc_0,pc_0,dcs,gc_0,ent_0,nt,dt,dx,Xe,f1,f2,phi) #solve

#%% Graphs source terms, phi

# plt.figure()
# plt.plot(Xe,f1(Xe,300+dcs),'-',label="$f_1$")
# plt.plot(Xe,f2(Xe,300+dcs),'-',label="$f_2$")
# plt.legend()

# plt.figure()
# plt.plot(Xe,R(1,Xe),label="R1")
# # plt.plot(Xe*K(2),R(2,Xe*K(2)),label="R2")
# plt.plot(Xe,R(3,Xe),label="R3")
# # plt.plot(Xe*K(2),R(4,Xe*K(2)),label="R4")
# plt.plot(Xe,R(5,Xe),label="R5")
# plt.legend()

# plt.figure()
# plt.plot(Xe,phi(Xe),label="$\phi$")
# plt.legend()

#%% Plot

# Initial conditions sc_0,pc_0,dcs
# plt.figure() ; plt.plot(Xe,sc_0,'b--',label=r"$\rho_{sc}^0$") ; plt.plot(Xe,pc_0,'r--',label=r"$\rho_{pc}^0$") ; plt.plot(Xe,dcs,'m-',label=r"$\rho_{dcs}$") ;plt.xlim(0,1.1) ; plt.ylim(0,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title("Conditions initiales "+r"$\rho_{sc}^0,\rho_{pc}^0,\rho_{dcs},$"+f" N={nx+1}")

mass_sc = dx*np.sum(sc_0) ; mass_pc = dx*np.sum(pc_0) ; mass_dcs = dx*np.sum(dcs) ; xshock = mass_sc/(mass_sc+mass_pc)

def plot(sc_mat,pc_mat,dcs,sc_0,pc_0,t):
    plt.figure()
    for n in range(np.shape(sc_mat)[0]):
        plt.cla()

        plt.plot(Xe,sc_mat[n,:],'b.-',label=r"$\rho_{sc}$")
        # plt.plot(Xe,sc_mat[n,:]+dcs,'b.-',label=r"$\rho_{sc}(+\rho_{dcs})$")
        plt.plot(Xe,sc_0,'b--',label=r"$\rho_{sc}(t=0)$")
        # plt.plot(Xe,pc_mat[n,:],'r.-',label=r"$\rho_{pc}$")
        plt.plot(Xe,pc_mat[n,:]+dcs,'r.-',label=r"$\rho_{pc}(+\rho_{dcs})$")
        plt.plot(Xe,pc_0,'r--',label=r"$\rho_{pc}(t=0)$")
        plt.plot(Xe,dcs,'m-',label=r"$\rho_{dcs}$")
        plt.plot(Xe,sc_mat[n,:]+pc_mat[n,:]+dcs,'k--',label="total")

        plt.ylim(-0.03,1.4)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Upwind, N={nx+1}, t={round(t[n]*dt,4)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,3)}, mass(u0)={np.round(mass_sc+mass_pc,4)}")
        plt.ylabel("Densities")
        plt.pause(0.01+0.5/(n**2+1))

plot(sc_mat,pc_mat,dcs,sc_0,pc_0,t)

