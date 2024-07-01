# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:29:06 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
from time import time ; from time import sleep ; from datetime import timedelta

#%%

def init_u(x): #double bosse
    a1 = 1/4 ; a2 = 3/4
    # a1 = 1/6 ; a2 = 5/6
    return np.sqrt(np.maximum(0,1-64*(a1-x)**2)) + np.sqrt(np.maximum(0,1-64*(a2-x)**2))
    # return np.sin(np.pi*x)

# def init_u(x): #bosse sinus
    # return np.sin(np.pi*x)

# def init_u(x): #double bosse smooth
#     if 0.1<x and 0.3>x  :
#         return np.exp(-1/( 1-( (x-0.2) /0.1)**2) )
#     if 0.7<x and 0.9>x:
#         return np.exp(-1/( 1-( (x-0.8)/0.1)**2) )
#     else :
#         return 0

# def init_u(x): #trapèze
#     if 0.1<x and 0.15>x  :
#         return (x-0.1)*(1/(0.15-0.1))
#     if 0.15<x and 0.25>x  :
#         return 1
#     if 0.25<x and 0.3>x:
#         return (x-0.3)*(1/(0.25-0.3))
#     else :
#         return 0

def init_w(x): #indicatrice
    b = 3/4
    # b = 5/6
    c = b - 0.25/2 ; d = b + 0.25/2
    return np.where(c<x,1.0,0)*np.where(x<d,1.0,0)
    # b = 0.8
    # c = b - 0.1 ; d = b + 0.1
    # return np.where(c<x,1.0,0)*np.where(x<d,1.0,0)


def solve_uw_Lagr(Xe0,u0,w0,long,cfl,tmax):
    u = u0 #init u, matrice des des u_j^n pour des temps n choisis
    uold = u0 #init uold
    
    w = w0 #init w, matrice des des w_j^n pour des temps n choisis
    
    X = Xc0 #init de la matrice des maillage en espace pour des temps n choisis
    Xe = Xe0 #init du maillage initiale en espace
    Xc = (Xe[:-1]+Xe[1:])/2 ; Xc = np.hstack([0,Xc,1]) #init maillage dual
    
    h = Xe[1:] - Xe[:-1] #init h_j
    
    mem = 1/100 #pourcentage temps qu'on garde en mémoire
    # mem = 1
    list_t_n = [0]
    list_dt = []
    
    temps = 0
    n = 0
    nitermax = 1e+4

    while (temps < tmax) and (n < nitermax) :
    
        dt = cfl*np.min(h)**2 #dt respectant cfl à chaque iter
        # dt = 0.001
    
        sigma = (uold[:-1]-uold[1:])/(Xc[1:]-Xc[:-1])

        unew = np.zeros_like(uold)
        unew[1:-1] = uold[1:-1]/(1 + dt/h*(sigma[1:]-sigma[:-1]))
        # print(1 + dt/h*(sigma[1:]-sigma[:-1]))
    
        unew[0] = unew[1] ; unew[-1] = unew[-2]  #conditions de bords Neumann homogene
        uold = unew
    
        Xe = Xe + dt*sigma #actualisation maillage (on fait évoluer les volumes de controles)
        h = Xe[1:] - Xe[:-1] #actualisation des h_j
        Xc = (Xe[:-1]+Xe[1:])/2 ; Xc = np.hstack([0,Xc,1]) #actualisation maillage centres des mailles
    
        if n%(int(1/mem))==0: #garde en mémoire une partie des itérations
            print(f"\ritérations : {n}, temps : {np.round(temps,8)}/{tmax}, dt={dt}, h_min = {np.min(h)}",end="",flush=True) #print avancement
            u = np.vstack([u,unew])
            w = np.vstack([w,w0])
            X = np.vstack([X,Xc])
            list_t_n.append(temps)

        temps += dt
        n += 1
        list_dt.append(dt)

    print("") ; print(f"Nombre d'itérations : {n}")

    return u,w,X,list_t_n,list_dt  #list_t_n : liste temps gardés en mémoire


#%% plot

long = 1.0 ; cfl = 0.495 ; tmax = 1.0 ; dx = 1/100

N = round(long/dx-1) ; Xe0,dx = np.linspace(0,long,N+2,retstep=True) #maillage initial

Xc0 = (Xe0[:-1]+Xe0[1:])/2 ; Xc0 = np.hstack([0,Xc0,1]) #maillage dual

u0 = init_u(Xc0)
# u0 = np.array([init_u(x) for x in Xc0])
w0 = init_w(Xc0)

#Initial conditions
# plt.figure() ; plt.plot(Xc0,u0,'b--',label="u0") ; plt.plot(Xc0,w0,'r--',label="w0") ; plt.ylim(-0.03,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales u0,w0, N+1={N+1}")


print(f"N+1={N+1}, cfl = {cfl}")
# norm1 = dx*np.linalg.norm(u0,ord=1)

u,w,X,t,list_dt = solve_uw_Lagr(Xe0,u0,w0,long,cfl,tmax)


def plot_uw(X,u,w,t):
    plt.figure()
    perc = 0.1 #pourcentage des itérations à afficher
    for n in range(int(np.shape(u)[0]*perc)):
        plt.cla()
        plt.plot(X[n],u[n,:],'c.-',label="u")
        plt.plot(X[n],w[n,:],'m.-',label="w")
        plt.plot(X[0],u0,'c--',label="u0")
        plt.plot(X[0],w0,'m--',label="w0")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions schéma Lagrangien, N+1={N+1}, t={round(t[n],8)}/{tmax}")
        plt.xlabel(f"cfl={cfl}, max(u0)={np.round(np.max(np.abs(u0)),4)}")
        plt.pause(0.05+0.1/(n**2+1))


plot_uw(X,u,w,t)

#%% Evolution of dt

plt.figure()
plt.plot(list_dt,"k.-")
plt.title("dt à chaque itérations")
plt.xlabel("itérations")
plt.show()


plt.figure()
plt.semilogy(list_dt,"k.-")
plt.title("dt à chaque itérations, échelle semi-log")
plt.xlabel("itérations")
plt.show()

#%% Reconstruction

u2 = w*u ; u1 = u-u2
u2_0 = w0*u0 ; u1_0 = u0-u2_0

# Initial conditions u1_0 et u2_0
# plt.figure() ; plt.plot(Xe,u1_0,'b--',label="u0") ; plt.plot(Xe,u2_0,'r--',label="w0") ; plt.plot(Xe,v,'m--',label="v") ; plt.ylim(0,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales u1_0,u2_0,v, N={nx+1}")

def plot_uu(X,u1,u2,v,u1_0,u2_0,t):
    plt.figure()
    perc = 0.1
    for n in range(int(np.shape(u1)[0]*perc)):
        plt.cla()
        plt.plot(X[n],u1[n,:],'b-',label="u1=u-w*u")
        plt.plot(X[n],u2[n,:],'r-',label="u2=w*u")
        # plt.plot(X[n],v,'m-',label="v")
        # plt.plot(X[n],u1[n,:]+u2[n,:]+v,'k-',label="u1+u2+v")
        # plt.plot(X[n],[norm1 for i in Xe0],'k--',label="norm1(u0)")
        # plt.plot(X[n],u0,'-',label="u0")
        # plt.plot(X[n],w0,'-',label="w0")
        plt.plot(X[0],u1_0,'b--',label="u1(t=0)")
        plt.plot(X[0],u2_0,'r--',label="u2(t=0)")
        # plt.plot(X[n],masse_u1[n]*np.ones(np.shape(w)[1]),'c--',label="masse u1")
        # plt.plot(X[n],masse_u2[n]*np.ones(np.shape(w)[1]),'m--',label="masse u2")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions u1,u2,v, N+1={N+1}, t={round(t[n],8)}")
        plt.xlabel(f"dt/dx**2={cfl}, max(u0)={np.round(np.max(np.abs(u0)),4)}")
        plt.pause(0.01+0.1/(n**2+1))

# plot_uu(X,u1,u2,v,u1_0,u2_0,t)


