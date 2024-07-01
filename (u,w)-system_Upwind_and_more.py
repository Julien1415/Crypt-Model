# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:01:19 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
import scipy.sparse as sp ; import scipy.sparse.linalg as sps
from time import time ; from datetime import timedelta

#%% Equations croisées

def mobi_vect_u(u,sign,methode_u): #mobilité en vectorisé (vecteur des u_{i+1/2} entre 1 et N), sign==plus : u_{i+1/2} ; sign==moins : u_{i-1/2}
    if methode_u=="Upwind" or methode_u=="IMEX-Upwind":
        if sign=="+1/2":
            return np.maximum(u[1:-1],u[2:])
        elif sign=="-1/2":
            return np.maximum(u[1:-1],u[0:-2])

    elif methode_u=="Centré" or methode_u=="IMEX-Centré":
        if sign=="+1/2":
            return (u[1:-1]+u[2:])/2
        elif sign=="-1/2":
            return (u[1:-1]+u[0:-2])/2

    elif methode_u=="LagrangeRemap":
        ubig = np.hstack([u[0],u,u[-1]])
        if sign=="j":
            return ubig[2:-2]/(1-dt/dx**2*(ubig[1:-3]-2*ubig[2:-2]+ubig[3:-1]))
        elif sign=="j+1":
            return ubig[3:-1]/(1-dt/dx**2*(ubig[2:-2]-2*ubig[3:-1]+ubig[4:]))
        elif sign=="j-1":
            return ubig[1:-3]/(1-dt/dx**2*(ubig[:-4]-2*ubig[1:-3]+ubig[2:-2]))

    else :
        print("Erreur nom methode mobi u")

def mobi_vect_w(u,v,dx,sign,methode_w): #mobilité en vectorisé, sign==plus : v_{i+1/2}_- ; sign==moins : v_{i-1/2}_+
    if methode_w=="Upwind" or methode_w=="IMEX-Upwind":
        if sign=="+1/2":
            return np.minimum(-(u[2:]-u[1:-1])/dx,0)
        elif sign=="-1/2":
            return np.maximum(-(u[1:-1]-u[:-2])/dx,0)

    elif methode_w=="Caractéristiques":
        sigma = (u[:-1]+v[:-1]-u[1:]-v[1:]) #?
        if sign=="+": #v_{j+1/2}_+
            return np.maximum(sigma,0)
        elif sign=="-": #v_{j+1/2}_-
            return np.minimum(sigma,0)

    else :
        print("Erreur nom methode mobi w")

def scheme_u(u,v,dt,dx,nx,methode_u): #u le vecteur au temps n, F la fonction telle que u_n+1 = F(u_n) ; f le second membre sol manu
    d = dt/dx**2
    if methode_u=="Upwind":
        unew = np.zeros_like(u)
        unew[1:-1] = d*np.maximum(0,u[:-2]+v[:-2]-u[1:-1]-v[1:-1])*u[:-2] \
                   + (1 + d*np.minimum(0,u[:-2]+v[:-2]-u[1:-1]-v[1:-1]) - d*np.maximum(0,u[1:-1]+v[1:-1]-u[2:]-v[2:]))*u[1:-1] \
                   - d*np.minimum(0,u[1:-1]+v[1:-1]-u[2:]-v[2:])*u[2:]

        unew[0] = unew[1] ; unew[-1] = unew[-2]  #conditions de bords Neumann homogene

    elif methode_u=="LagrangeRemap":
        unew = np.zeros_like(u)

        unew[1:-1] = u[1:-1] - dt/dx**2*( mobi_vect_u(u,"j","LagrangeRemap")*np.maximum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
                                        + mobi_vect_u(u,"j+1","LagrangeRemap")*np.minimum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
                                        - mobi_vect_u(u,"j-1","LagrangeRemap")*np.maximum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) \
                                        - mobi_vect_u(u,"j","LagrangeRemap")*np.minimum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) )

        unew[0] = unew[1] ; unew[-1] = unew[-2]

    elif methode_u=="Centré": #!!!
        unew = np.zeros_like(u)

        unew[1:-1] = u[1:-1] - d*( (u[1:-1]+v[1:-1]+u[2:]+v[2:])/2*(u[1:-1]+v[1:-1]-u[2:]-v[2:]) \
                                  - (u[:-2]+v[:-2]+u[1:-1]+v[1:-1])/2*(u[:-2]+v[:-2]-u[1:-1]-v[1:-1]) )

        unew[0] = unew[1] ; unew[-1] = unew[-2]

    # elif methode_u=="IMEX-Upwind" or methode_u=="IMEX-Centré": #cas implicite-explicite #!!!
    #     sous_diag = np.zeros(nx+1)
    #     sous_diag[:-1] = (-d)*mobi_vect_u(u,"-1/2",methode_u)
    #     sous_diag[-1] = -1
    #     diag = np.ones(nx+2)
    #     diag[1:-1] = 1 + d*mobi_vect_u(u,"+1/2",methode_u) + d*mobi_vect_u(u,"-1/2",methode_u)
    #     sur_diag = np.zeros(nx+1)
    #     sur_diag[1:] = (-d)*mobi_vect_u(u,"+1/2",methode_u)
    #     sur_diag[0] = -1

    #     A = sp.diags([sous_diag,diag,sur_diag],[-1,0,1],format='csr')
    #     B = np.zeros_like(u) ; B[1:-1] = u[1:-1]

    #     #terme v
    #     if methode_u=="IMEX-Upwind":
    #         B[1:-1] += d*np.maximum(v[:-2]-v[1:-1],0)*u[:-2] \
    #                    - d*( np.maximum(v[1:-1]-v[2:],0) - np.minimum(v[:-2]-v[1:-1],0) )*u[1:-1] \
    #                    - d*np.minimum(v[1:-1]-v[2:],0)*u[2:]

    #     elif methode_u=="IMEX-Centré":
    #         B[1:-1] += d*(v[:-2]-v[1:-1])/2*u[:-2] \
    #                    - d*( (v[1:-1]-v[2:])/2 - (v[:-2]-v[1:-1])/2 )*u[1:-1] \
    #                    - d*(v[1:-1]-v[2:])/2*u[2:]

    #     unew = sps.spsolve(A,B)

    else :
        print("Erreur nom méthode")

    return unew

def scheme_w(w,u,v,dt,dx,nx,methode_w): #u le vecteur u au temps n
    d = dt/dx**2
    if methode_w=="Upwind":
        wnew = np.zeros_like(w)
        wnew[0] = w[0] ; wnew[-1] = w[-1]  #conditions de bords

        wnew[1:-1] = d*np.maximum(0,u[:-2]+v[:-2]-u[1:-1]-v[1:-1])*w[:-2] \
                   + (1 + d*np.minimum(0,u[1:-1]+v[1:-1]-u[2:]-v[2:]) - d*np.maximum(0,u[:-2]+v[:-2]-u[1:-1]-v[1:-1]))*w[1:-1] \
                   - d*np.minimum(0,u[1:-1]+v[1:-1]-u[2:]-v[2:])*w[2:]

    # elif methode_w=="IMEX-Upwind": #cas implicite-explicite
    #     sous_diag = np.zeros(nx+1)
    #     sous_diag[:-1] = -d*mobi_vect_w(u,dx,"-1/2",methode_w)
    #     diag = np.ones(nx+2)
    #     diag[1:-1] = 1 - d*mobi_vect_w(u,dx,"+1/2",methode_w) + d*mobi_vect_w(u,dx,"-1/2",methode_w)
    #     sur_diag = np.zeros(nx+1)
    #     sur_diag[1:] = d*mobi_vect_w(u,dx,"+1/2",methode_w)

    #     A = sp.diags([sous_diag,diag,sur_diag],[-1,0,1],format='csr')
    #     B = w

    #     #terme v
    #     B[1:-1] += -d*(mobi_vect_w(u,dx,"+1/2",methode_w)*(v[2:]-v[1:-1]) + mobi_vect_w(u,dx,"-1/2",methode_w)*(v[1:-1]-v[:-2]))

    #     wnew = sps.spsolve(A,B)

    elif methode_w=="Caractéristiques":
        wnew = (d*mobi_vect_w(u,v,dx,"+",methode_w))*w[:-2] \
              + (1 - d*mobi_vect_w(u,v,dx,"+",methode_w) + d*mobi_vect_w(u,v,dx,"-",methode_w))*w[1:-1] \
              + (-d*mobi_vect_w(u,v,dx,"-",methode_w))*w[2:]

        wnew = np.hstack([w[0],wnew,w[-1]])

    elif methode_w=="LagrangeRemap":
        wnew = np.zeros_like(w)

        wnew[1:-1] = w[1:-1] + dt/dx**2*( np.maximum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0)*(w[:-2]-w[1:-1]) \
                                        + np.minimum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0)*(w[1:-1]-w[2:]) )

        wnew[0] = w[0] ; wnew[-1] = w[-1]

    else :
        print("Erreur nom méthode")

    return wnew

def solve_uw(u0,w0,v,Xt,nt,dt,Xe,nx,dx,methode_u,methode_w):
    tps = time()
    
    u = np.zeros((1,len(u0)))
    u[0,:] = u0
    uold = u0

    w = np.zeros((1,len(w0)))
    w[0,:] = w0
    if methode_w=="Caractéristiques":
        Xee = (Xe[:-1]+Xe[1:])/2
        Xee = np.hstack([0,Xee,1])
        w0 = init_w(Xee) #w0 sur les mailles demi
    wold = w0

    mem = min(nt,100) #nombre de temps que l'on veux garder en mémoire, mem = nt pour tout garder
    list_t_n = [0]
    
    for n in range(1,nt):
        
        unew = scheme_u(uold,v,dt,dx,nx,methode_u)
        wnew = scheme_w(wold,uold,v,dt,dx,nx,methode_w)
        # wnew = scheme_w(wold,unew,v,dt,dx,nx,methode_w) #sigma^{n+1} pour Lagrange

        uold = unew
        wold = wnew

        if n%(nt//mem)==0 or (n==nt-1): #garde en mémoire une partie des itérations
            print(f"\r{len(list_t_n)}/{mem} ",end="",flush=True) #print avancement
            if methode_w=="Caractéristiques":
                wnew = (wnew[2:-1] + wnew[1:-2])/2 #reconstruction au centre des mailles
                wnew = np.hstack([wnew[0],wnew,wnew[-1]])
            u = np.vstack([u,unew])
            w = np.vstack([w,wnew])
            list_t_n.append(n)
    print("",f"{methode_u}-{methode_w}(u,w) :",np.round(timedelta(seconds=time()-tps).total_seconds(),3),"secondes")

    return u,w,list_t_n  #list_t_n : liste des indices des temps gardés en mémoire

def init_u(x): #double bosse
    a1 = 1/4 ; a2 = 3/4
    # a1 = 1/6 ; a2 = 5/6
    # return np.sqrt(np.maximum(0,1-64*(a1-x)**2)) + np.sqrt(np.maximum(0,1-64*(a2-x)**2))*0.6
    return np.sqrt(np.maximum(0,1-64*(a1-x)**2)) + np.sqrt(np.maximum(0,1-64*(a2-x)**2))

def init_w(x): #indicatrice
    a2 = 3/4
    # a2 = 5/6
    wL = a2 - 0.25/2 ; wD = a2 + 0.25/2
    return np.where(wL<x,1.0,0) * np.where(x<wD,1.0,0)

def init_v(x): #bosse/creneau
    a = 0.05
    a1 = 0.5 - a ; a2 = 0.5 + a
    # a = 3/6
    # return np.sqrt(np.maximum(0,1-10*64*(0.5-x)**2))*0.3
    # return (np.where(a1<x,1.0,0) * np.where(x<a2,1.0,0))*1.0
    # return (np.where(a1<x,1.0,0) * np.where(x<a2,1.0,0))*0.45
    return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))*(1+1/2*np.sin(2*np.pi*(x-1/2+a)/(2*a)))*0.265
    # return 0.0*x

#%% Paramètres et graphes

long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.25 ; dx = 1/200
dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1) #dt following the cfl and such as the inverse is an integer
nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

u0 = init_u(Xe)
w0 = init_w(Xe)
v = init_v(Xe)

methode_u = "Upwind"
# methode_u = "Centré"
# methode_u = "IMEX-Upwind"
# methode_u = "IMEX-Centré"
methode_u = "LagrangeRemap"

methode_w = "Upwind"
# methode_w = "IMEX-Upwind"
# methode_w = "Caractéristiques"
methode_w = "LagrangeRemap"

print(f"N={nx+1}, nt={nt}")
cfl = dt/dx**2 ; print(f"dt/dx**2 = {np.round(cfl,4)}")

#Initial conditions
plt.figure() ; plt.plot(Xe,u0,'c--',label="$u^0$") ; plt.plot(Xe,w0,'m--',label="$w^0$") ; plt.ylim(-0.03,1.1) ; plt.xlim(0,0.99) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u^0,w^0,$ N={nx+1}")
plt.figure() ; plt.plot(Xe,u0,'c--',label="$u^0$") ; plt.plot(Xe,w0,'m--',label="$w^0$") ; plt.plot(Xe,v,'b--',label="$v$") ; plt.ylim(-0.03,1.1) ; plt.xlim(0,0.99) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u^0,w^0,v,$ N={nx+1}")

#Solve
u,w,t = solve_uw(u0,w0,v,Xt,nt,dt,Xe,nx,dx,methode_u,methode_w)

u2 = w*u ; u1 = u-u2
u2_0 = w0*u0 ; u1_0 = u0-u2_0

m1 = dx*np.sum(u1_0) ; m2 = dx*np.sum(u2_0)
xshock = m1/(m1+m2)

def plot_uw(u,w,v,t):
    plt.figure()
    for n in range(np.shape(w)[0]):
        plt.cla()
        plt.plot(Xe,u[n,:],'c.-',label="$u$")
        plt.plot(Xe,u0,'c--',label="$u^0$")
        plt.plot(Xe,w[n,:],'m.-',label="$w$")
        plt.plot(Xe,w0,'m--',label="$w^0$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'b-',label="$v$")
            plt.plot(Xe,m1/0.45*Xe**0,'k--',label="Temps long")
        else:
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u_0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions $u,w$ ($u$:{methode_u}, $w$:{methode_w}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(dx*np.sum(u1_0*1/0.5),5)}")
        plt.pause(0.05+0.1/(n**2+1))

# plot_uw(u,w,v,t)

#%% Reconstruction

# Initial conditions u1_0 et u2_0
plt.figure() ; plt.plot(Xe,u1_0,'b--',label="$u_1^0$") ; plt.plot(Xe,u2_0,'r--',label="$u_2^0$") ; plt.ylim(-0.03,1.1) ; plt.xlim(0,0.99) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u_1^0,u_2^0$, N={nx+1}")
plt.figure() ; plt.plot(Xe,u1_0,'b--',label="$u_1^0$") ; plt.plot(Xe,u2_0,'r--',label="$u_2^0$") ; plt.plot(Xe,v,'m--',label="$v$") ; plt.ylim(-0.03,1.1) ; plt.xlim(0,0.99) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u_1^0,u_2^0,v$, N={nx+1}")

# masse_u1 = np.array([dx*np.linalg.norm(u1[n,:],ord=1) for n in range(np.shape(w)[0])])
# masse_u2 = np.array([dx*np.linalg.norm(u2[n,:],ord=1) for n in range(np.shape(w)[0])])

def plot_u1u2(u1,u2,v,u1_0,u2_0,t,methode_u,methode_w):
    plt.figure()
    for n in range(np.shape(u1)[0]):
        plt.cla()
        plt.plot(Xe,u1[n,:],'b-',label="$u_1=u-wu$")
        plt.plot(Xe,u1_0,'b--',label="$u_1^0$")
        plt.plot(Xe,u2[n,:],'r-',label="$u_2=wu$")
        plt.plot(Xe,u2_0,'r--',label="$u_2^0$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'m-',label="$v$")
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="u1+u2+v")
            plt.plot(Xe,m1/0.45*Xe**0,'k--',label="Temps long")
        else:
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u_0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions $u_1,u_2,v$ ($u$:{methode_u}, $w$:{methode_w}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(dx*np.sum(u1_0*1/0.5),5)}")
        plt.pause(0.01+0.1/(n**2+1))

# plot_u1u2(u1,u2,v,u1_0,u2_0,t,methode_u,methode_w)

#%% TVD

# TV_u = [np.sum(np.abs(u[n,:-1]-u[n,1:])) for n in range(len(t))]
# TV_w = [np.sum(np.abs(w[n,:-1]-w[n,1:])) for n in range(len(t))]

# plt.figure()
# plt.plot(Xt[t],TV_u,'k.-',label="TV(u)")
# plt.ylim(-0.03,4.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $u$ pour le schéma {methode_u}, N={nx+1}")

# plt.figure()
# plt.plot(Xt[t],TV_w,'k.-',label="TV(w)")
# plt.ylim(0,4.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $w$ pour le schéma {methode_w}, N={nx+1}")

#%% Entropy

# def entropie(u,t,dx): #u matrice des u_i^n
#     u[u==0] = 1e-15 #convention 0*log(0) (remplace 0 par 1e-15)
#     H = np.array([dx*np.sum(u[n,:]*(np.log(u[n,:])-1)+1) for n in range(np.shape(u)[0])])
#     plt.figure()
#     plt.plot(Xt[t],H,'k.-')
#     plt.ylim(0,1)
#     plt.xlabel("t")
#     plt.title(f"Entropie pour le schéma {methode_u} sur $u$, N={nx+1}")

# entropie(u,t,dx)

#%% Convergence order

# # methode_u = "Upwind"
# # methode_u = "Centré"
# # methode_u = "IMEX-Upwind"
# # methode_u = "IMEX-Centré"
# methode_u = "LagrangeRemap"

# # methode_w = "Upwind"
# # methode_w = "IMEX-Upwind"
# # methode_w = "Caractéristiques"
# methode_w = "LagrangeRemap"

# nb_point = 5
# long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.25 ; dx_ref = 1/(50*2**nb_point)

# dt_ref = coef_cfl*dx_ref**2 ; dt_ref = 1/(int(1/dt_ref)+1)
# nx = round(long/dx_ref-1) ; Xe_ref,dx_ref = np.linspace(0,long,nx+2,retstep=True)
# nt = round(tmax/dt_ref+1) ; Xt_ref,dt_ref = np.linspace(0,tmax,nt,retstep=True)

# print(f"dx:1/{int(1/dx_ref)}")
# u_ref,w_ref,t_ref = solve_uw(init_u(Xe_ref), init_w(Xe_ref), init_v(Xe_ref), Xt, nt, dt, Xe, nx, dx, methode_u, methode_w)

# # u2 = w_ref*u_ref ; u1 = u_ref-u2 #remplace u,w par u1,u2 pour avoir les ordres sur u1,u2
# # u_ref = u1 ; w_ref = u2

# list_dx = [1/(50*2**i) for i in range(nb_point)]
# erreur_u1 = [] ; erreur_u2 = []

# for dx in list_dx:
#     dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1)
#     nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
#     nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

#     print(f"dx:1/{int(1/dx)}")
#     u,w,t = solve_uw(init_u(Xe), init_w(Xe), init_v(Xe), Xt, nt, dt, Xe, nx, dx, methode_u, methode_w)
#     u_fin = u[-1,:] ; w_fin = w[-1,:]

#     # u2 = w_fin*u_fin ; u1 = u_fin-u2
#     # u_fin = u1 ; w_fin = u2

#     # Projection de la solution de reference
#     u_proj = u_ref[-1,:] ; w_proj = w_ref[-1,:]
#     if len(u_fin)%2==1:
#         u_fin = u_fin[:-1] ; w_fin = w_fin[:-1]
#         u_proj = u_proj[:-1] ; w_proj = w_proj[:-1]

#     r = int(np.log2(dx/dx_ref))
#     for i in range(r): #remove half of values
#         u_proj = np.reshape(u_proj,(2,int(len(u_proj)/2))) ; u_proj = u_proj[0,:]
#         w_proj = np.reshape(w_proj,(2,int(len(w_proj)/2))) ; w_proj = w_proj[0,:]

#     err1 = dx*np.linalg.norm(u_fin - u_proj,ord=2)
#     err2 = dx*np.linalg.norm(w_fin - w_proj,ord=2)

#     erreur_u1.append(err1)
#     erreur_u2.append(err2)

# x = np.array(list_dx)[:] ; y1 = np.array(erreur_u1)[:] ; y2 = np.array(erreur_u2)[:]

# ordre_u = np.round(np.polyfit(np.log10(x), np.log10(y1), 1)[0],2)
# ordre_w = np.round(np.polyfit(np.log10(x), np.log10(y2), 1)[0],2)

# plt.figure()
# plt.loglog(x,y1,'b.-',label=f"Résidu sur $u$, ~O({ordre_u})")
# # plt.loglog(x,y1[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y1[0]*(x/x[0])**1,'--',label='O(1)')
# plt.loglog(x,y1[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_u}, t={tmax}")

# plt.figure()
# plt.loglog(x,y2,'r.-',label=f"Résidu sur $w$, ~O({ordre_w})")
# plt.loglog(x,y2[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y2[0]*(x/x[0])**1,'--',label='O(1)')
# # plt.loglog(x,y2[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_w}, t={tmax}")

