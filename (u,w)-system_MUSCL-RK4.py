# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:25:54 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
from time import time ; from datetime import timedelta

#%% Equations croisées

def minmod(r,cfl):
    return np.maximum(0,np.minimum(1,r))

def superbee(r,cfl):
    return np.maximum(0,np.maximum(np.minimum(2*r,1),np.minimum(r,2)))

def osher(r,cfl):
    return np.maximum(0,np.minimum(1.5,r))

def ultrabee(r,cfl):
    return np.maximum(0,np.maximum(np.minimum(2*r/cfl,1),np.minimum(r,2/(1-cfl))))

def mobi_MUSCL(p,phi,sign):
    pbig = np.hstack([p[0],p,p[-1]]) #prolongement pour le stencil

    if sign=="+1/2,L":
        d = (pbig[3:-1]-pbig[2:-2])
        d[d<1e-15] = 1e-15 #remplace les valeurs <1e-15 par 1e-15
        r = (pbig[2:-2]-pbig[1:-3])/d
        return pbig[2:-2] + 0.5*phi(r,cfl)*(pbig[3:-1]-pbig[2:-2])

    elif sign=="+1/2,R":
        d = (pbig[4:]-pbig[3:-1])
        d[d<1e-15] = 1e-15
        r = (pbig[3:-1]-pbig[2:-2])/d
        return pbig[3:-1] - 0.5*phi(r,cfl)*(pbig[4:]-pbig[3:-1])

    elif sign=="-1/2,L":
        d = (pbig[2:-2]-pbig[1:-3])
        d[d<1e-15] = 1e-15
        r = (pbig[1:-3]-pbig[0:-4])/d
        return pbig[1:-3] - 0.5*phi(r,cfl)*(pbig[2:-2]-pbig[1:-3])

    elif sign=="-1/2,R":
        d = (pbig[3:-1]-pbig[2:-2])
        d[d<1e-15] = 1e-15
        r = (pbig[2:-2]-pbig[1:-3])/d
        return pbig[2:-2] - 0.5*phi(r,cfl)*(pbig[3:-1]-pbig[2:-2])

def f(y,u,v,dx):
    ynew = - 1/dx**2*( mobi_MUSCL(y,phi,"+1/2,L")*np.maximum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
                     + mobi_MUSCL(y,phi,"+1/2,R")*np.minimum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
                     - mobi_MUSCL(y,phi,"-1/2,L")*np.maximum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) \
                     - mobi_MUSCL(y,phi,"-1/2,R")*np.minimum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) )

    # print(np.sum( mobi_MUSCL(y,phi,"+1/2,L")*np.maximum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
    #                  + mobi_MUSCL(y,phi,"+1/2,R")*np.minimum(u[1:-1]+v[1:-1]-u[2:]-v[2:],0) \
    #                  - mobi_MUSCL(y,phi,"-1/2,L")*np.maximum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) \
    #                  - mobi_MUSCL(y,phi,"-1/2,R")*np.minimum(u[:-2]+v[:-2]-u[1:-1]-v[1:-1],0) ))

    return ynew

def solve_uw_MUSCL(u0,w0,v,phi,nt,dt,dx,methode_temps):
    tps = time()

    u = np.zeros((1,len(u0)))
    u[0,:] = u0
    uold = u0

    ksi = np.zeros((1,len(w0)))
    ksi[0,:] = np.ones(len(w0))
    ksiold = np.ones(len(w0))

    eta = np.zeros((1,len(w0)))
    eta[0,:] = ksi[0,:]*w0
    etaold = ksi[0,:]*w0

    mem = 100 ; mem = min(nt,mem) #nombre de temps que l'on veux garder en mémoire, mem = nt pour tout garder
    list_t = [0] #liste des indices des temps gardés en mémoire

    for n in range(1,nt):
    # for n in range(10):

        unew = np.zeros_like(uold)
        ksinew = np.zeros_like(ksiold)
        etanew = np.zeros_like(etaold)


        if methode_temps=="Euler":

            unew[1:-1] = uold[1:-1] + dt*f(uold,uold,v,dx)
            ksinew[1:-1] = ksiold[1:-1] + dt*f(ksiold,uold,v,dx)
            etanew[1:-1] = etaold[1:-1] + dt*f(etaold,uold,v,dx)

        elif methode_temps=="RK2":

            ytilde = uold[1:-1] + dt/2*f(uold,uold,v,dx) ; ytilde = np.hstack([ytilde[0],ytilde,ytilde[-1]])
            unew[1:-1] = uold[1:-1] + dt*f(ytilde,uold,v,dx)

            ytilde = ksiold[1:-1] + dt/2*f(ksiold,uold,v,dx) ; ytilde = np.hstack([ytilde[0],ytilde,ytilde[-1]])
            ksinew[1:-1] = ksiold[1:-1] + dt*f(ytilde,uold,v,dx)

            ytilde = etaold[1:-1] + dt/2*f(etaold,uold,v,dx) ; ytilde = np.hstack([ytilde[0],ytilde,ytilde[-1]])
            etanew[1:-1] = etaold[1:-1] + dt*f(ytilde,uold,v,dx)

        elif methode_temps=="RK4":

            k1 = f(uold,uold,v,dx) ; k1 = np.hstack([k1[0],k1,k1[-1]])
            k2 = f(uold+dt/2*k1,uold+dt/2*k1,v,dx) ; k2 = np.hstack([k2[0],k2,k2[-1]])
            k3 = f(uold+dt/2*k2,uold+dt/2*k2,v,dx) ; k3 = np.hstack([k3[0],k3,k3[-1]])
            k4 = f(uold+dt*k3,uold+dt*k3,v,dx) ; k4 = np.hstack([k4[0],k4,k4[-1]])
            unew[1:-1] = uold[1:-1] + dt/6*(k1[1:-1]+2*k2[1:-1]+2*k3[1:-1]+k4[1:-1])

            k1 = f(ksiold,uold,v,dx) ; k1 = np.hstack([k1[0],k1,k1[-1]])
            k2 = f(ksiold+dt/2*k1,uold,v,dx) ; k2 = np.hstack([k2[0],k2,k2[-1]])
            k3 = f(ksiold+dt/2*k2,uold,v,dx) ; k3 = np.hstack([k3[0],k3,k3[-1]])
            k4 = f(ksiold+dt*k3,uold,v,dx) ; k4 = np.hstack([k4[0],k4,k4[-1]])
            ksinew[1:-1] = ksiold[1:-1] + dt/6*(k1[1:-1]+2*k2[1:-1]+2*k3[1:-1]+k4[1:-1])

            k1 = f(etaold,uold,v,dx) ; k1 = np.hstack([k1[0],k1,k1[-1]])
            k2 = f(etaold+dt/2*k1,uold,v,dx) ; k2 = np.hstack([k2[0],k2,k2[-1]])
            k3 = f(etaold+dt/2*k2,uold,v,dx) ; k3 = np.hstack([k3[0],k3,k3[-1]])
            k4 = f(etaold+dt*k3,uold,v,dx) ; k4 = np.hstack([k4[0],k4,k4[-1]])
            etanew[1:-1] = etaold[1:-1] + dt/6*(k1[1:-1]+2*k2[1:-1]+2*k3[1:-1]+k4[1:-1])

        unew[0] = unew[1] ; unew[-1] = unew[-2]
        ksinew[0] = ksiold[0] ; ksinew[-1] = ksiold[-1]
        etanew[0] = etaold[0] ; etanew[-1] = etaold[-1]

        uold = unew
        ksiold = ksinew
        etaold = etanew

        if n%(nt//mem)==0 or (n==nt-1): #garde en mémoire une partie des itérations
            print(f"\r{len(list_t)}/{mem} ",end="",flush=True) #print avancement
            u = np.vstack([u,unew])
            ksi = np.vstack([ksi,ksinew])
            eta = np.vstack([eta,etanew])
            list_t.append(n)
    print("",f"MUSCL({limiteur}) :",np.round(timedelta(seconds=time()-tps).total_seconds(),3),"secondes")

    return u,ksi,eta,list_t

def init_u(x): #double bosse
    a1 = 1/4 ; a2 = 3/4
    # a1 = 1/6 ; a2 = 5/6
    return np.sqrt(np.maximum(0,1-64*(a1-x)**2)) + np.sqrt(np.maximum(0,1-64*(a2-x)**2))
    # return np.sqrt(np.maximum(0,1-64*(a1-x)**2)) + np.sqrt(np.maximum(0,1-64*(a2-x)**2))*0.4
    # return np.sqrt(np.maximum(0,1-64*(a2-x)**2))

def init_w(x): #indicatrice
    a2 = 3/4
    # a2 = 5/6
    wL = a2 - 0.25/2 ; wD = a2 + 0.25/2
    return np.where(wL<x,1.0,0) * np.where(x<wD,1.0,0)

def init_v(x): #bosse/creneau/v=0
    a = 1/2
    a1 = a - 0.05 ; a2 = a + 0.05
    # return np.sqrt(np.maximum(0,1-10*64*(a-x)**2))*0.3
    # return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))*1.0
    return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))*0.35
    # return 0.0*x

#%% Paramètres et graphes

long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.25 ; dx = 1/200
dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1) #dt following the cfl and such as the inverse is an integer
nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

u0 = init_u(Xe) ; w0 = init_w(Xe) ; v = init_v(Xe)

# methode_temps = "Euler"
methode_temps = "RK2"
# methode_temps = "RK4"

phi, limiteur = minmod, "minmod"
# phi, limiteur = superbee, "superbee"
# phi, limiteur = osher, "osher"
# phi, limiteur = ultrabee, "ultrabee"

print(f"N={nx+1}, nt={nt}")
cfl = dt/dx**2 ; print(f"dt/dx**2 = {np.round(cfl,4)}")

#Initial conditions
# plt.figure() ; plt.plot(Xe,u0,'c--',label="$u^0$") ; plt.plot(Xe,w0,'m--',label="$w^0$") ; plt.plot(Xe,v,'b--',label="$v$") ; plt.ylim(0,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u^0,w^0,v$, N={nx+1}")

#Flux limiters
# plt.figure()
# plt.plot(np.linspace(0,3,200),minmod(np.linspace(0,3,200),cfl),'',label="minmod")
# plt.plot(np.linspace(0,3,200),superbee(np.linspace(0,3,200),cfl),'',label="superbee")
# # plt.plot(np.linspace(0,3,200),osher(np.linspace(0,3,200),cfl),'',label="osher")
# plt.plot(np.linspace(0,3,200),ultrabee(np.linspace(0,3,200),cfl),'',label=f"ultrabee(cfl={np.round(cfl,4)})")
# plt.fill_between(np.linspace(0,3,200),minmod(np.linspace(0,3,200),cfl),superbee(np.linspace(0,3,200),cfl),color='gray',alpha=0.3,label="region TVD")
# plt.xlim(0,3)
# plt.ylim(0,3)
# plt.xlabel("x")
# plt.grid()
# plt.legend()
# plt.title("Limiteurs de flux")

# plt.figure()
# for cfl_test in [0.1,0.3,0.5,0.7,0.9]:
#     plt.plot(np.linspace(0,7,200),ultrabee(np.linspace(0,7,200),cfl_test),'',label=f"ultrabee(cfl={cfl_test})")
# plt.xlim(0,7)
# plt.ylim(0,7)
# plt.xlabel("x")
# plt.grid()
# plt.legend()
# plt.title("Limiteur de flux Ultrabee en fonction de la cfl")

#Solve
u,ksi,eta,t = solve_uw_MUSCL(u0,w0,v,phi,nt,dt,dx,methode_temps)

w = eta/ksi
u2 = w*u ; u1 = u-u2
u2_0 = w0*u0 ; u1_0 = u0-u2_0

m1 = dx*np.sum(u1_0[1:-1]) ; m2 = dx*np.sum(u2_0[1:-1])
xshock = m1/(m1+m2)

m1_inf = dx*np.sum(u1[-1,:]) ; m2_inf = dx*np.sum(u2[-1,:])
xshock_inf = m1_inf/(m1_inf+m2_inf)

#%% Plot u,ksi,eta

def plot_uksieta(u,ksi,eta,v,t):
    plt.figure()
    for n in range(np.shape(u)[0]):
        plt.cla()
        plt.plot(Xe,u[n,:],'c-',label="$u$")
        plt.plot(Xe,u0,'c--',label="u0")
        plt.plot(Xe,w0,'m--',label="w0")
        plt.plot(Xe,ksi[n,:],'-',label="$\\xi$")
        plt.plot(Xe,eta[n,:],'-',label="$\eta=w*\\xi$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'b-',label="v")
            plt.plot(Xe,m1/0.45*(Xe*0+1),'k--',label="Temps long")
        else:
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u^0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.5)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions schéma {methode_temps}-MUSCL({limiteur}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(dx*np.sum(m1+m2),5)}")
        plt.pause(0.05+0.1/(n**2+1))

# plot_uksieta(u,ksi,eta,v,t)

#%% Reconstruction u,w

def plot_uw(u,w,v,t):
    plt.figure()
    for n in range(np.shape(w)[0]):
        plt.cla()
        plt.plot(Xe,u[n,:],'c.-',label="$u$")
        plt.plot(Xe,u0,'c--',label="$u_0$")
        plt.plot(Xe,w[n,:],'m.-',label="$w$")
        plt.plot(Xe,w0,'m--',label="$w_0$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'b-',label="$v$")
            plt.plot(Xe,dx*np.sum(u1_0*1/0.45)*(Xe*0.0+1),'k--',label="Temps long")
        else:
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u^0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend(loc='upper left')
        plt.title(f"$u$,$w$ : schéma {methode_temps}-MUSCL({limiteur}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(dx*np.sum(m1+m2),5)}")
        plt.pause(0.05+0.1/(n**2+1))

# plot_uw(u,w,v,t)

def plot_uksiw(u,ksi,w,v,t):
    plt.figure()
    for n in range(np.shape(w)[0]):
        plt.cla()
        plt.plot(Xe,u[n,:],'c.-',label="$u$")
        plt.plot(Xe,u0,'c--',label="$u_0$")
        plt.plot(Xe,w[n,:],'m.-',label="$w$")
        plt.plot(Xe,w0,'m--',label="$w_0$")
        plt.plot(Xe,ksi[n,:],'-',label="$\\xi$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'b-',label="$v$")
            plt.plot(Xe,dx*np.sum(u1_0*1/0.45)*Xe**0,'k--',label="Temps long")
        else:
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u^0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend(loc='upper left')
        plt.title(f"$u$,$\\xi$,$w$ : schéma {methode_temps}-MUSCL({limiteur}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(dx*np.sum(m1+m2),5)}")
        plt.pause(0.05+0.1/(n**2+1))

# plot_uksiw(u,ksi,w,v,t)

#%% Reconstruction u1,u2

# Initial conditions u1_0 et u2_0
# plt.figure() ; plt.plot(Xe,u1_0,'b--',label="$u_1^0$") ; plt.plot(Xe,u2_0,'r--',label="$u_2^0$") ; plt.plot(Xe,v,'m--',label="$v$") ; plt.ylim(0,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u_1^0,u_2^0,v$, N={nx+1}")

masse_u1 = np.array([dx*np.linalg.norm(u1[n,:],ord=1) for n in range(np.shape(w)[0])])
masse_u2 = np.array([dx*np.linalg.norm(u2[n,:],ord=1) for n in range(np.shape(w)[0])])

def plot_u1u2(u1,u2,v,u1_0,u2_0,t):
    plt.figure()
    for n in range(np.shape(u1)[0]):
        plt.cla()
        # plt.plot(Xe,u1[n,:],'b-',label="$u_1$")
        plt.plot(Xe,u1[n,:]+v,'b.-',label="$u_1+v$")
        plt.plot(Xe,u1_0,'b--',label="$u_1^0$")
        # plt.plot(Xe,u2[n,:],'r-',label="$u_2$")
        plt.plot(Xe,u2[n,:]+v,'r.-',label="$u_2+v$")
        plt.plot(Xe,u2_0,'r--',label="$u_2^0$")
        if np.max(v)>0.0:
            plt.plot(Xe,v,'m-',label="$v$")
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="$u1+u2+v$")
            plt.plot(Xe,dx*np.sum(u1_0*1/0.45)*Xe**0,'k--',label="Temps long")
        else:
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="$u1+u2$")
            plt.plot(Xe,m1/xshock*Xe**0,'k--',label="masse $u^0$")
            plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"$u_1,u_2$ : schéma {methode_temps}-MUSCL({limiteur}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(m1+m2,5)}")
        plt.pause(0.01+0.1/(n**2+1))

plot_u1u2(u1,u2,v,u1_0,u2_0,t)

#%% TVD

# plt.figure()
# plt.plot(Xt[t],[np.sum(np.abs(u[n,:-1]-u[n,1:])) for n in range(len(t))],'k.-',label="TV(u)")
# plt.ylim(-0.03,4.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $u$ pour {methode_temps}-MUSCL({limiteur}), N={nx+1}")

# plt.figure()
# plt.plot(Xt[t],[np.sum(np.abs(w[n,:-1]-w[n,1:])) for n in range(len(t))],'k.-',label="TV(w)")
# plt.ylim(0,4.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $w$ pour {methode_temps}-MUSCL({limiteur}), N={nx+1}")

#%% Convergence order

# methode_temps = "Euler"
# # methode_temps = "RK2"
# # methode_temps = "RK4"

# phi, limiteur = minmod, "minmod"
# # phi, limiteur = superbee, "superbee"
# # phi, limiteur = osher, "osher"
# # phi, limiteur = ultrabee, "ultrabee"

# nb_point = 4
# long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.25 ; dx_ref = 1/(50*2**nb_point)

# dt_ref = coef_cfl*dx_ref**2 ; dt_ref = 1/(int(1/dt_ref)+1)
# nx = round(long/dx_ref-1) ; Xe_ref,dx_ref = np.linspace(0,long,nx+2,retstep=True)
# nt = round(tmax/dt_ref+1) ; Xt_ref,dt_ref = np.linspace(0,tmax,nt,retstep=True)

# print(f"dx:1/{int(1/dx_ref)}")
# u_ref,ksi_ref,eta_ref,t_ref = solve_uw_MUSCL(init_u(Xe_ref), init_w(Xe_ref), init_v(Xe_ref), phi, nt, dt, dx, methode_temps)
# w_ref = eta_ref/ksi_ref

# list_dx = [1/(50*2**i) for i in range(nb_point)]
# erreur_u = [] ; erreur_w = []

# for dx in list_dx:
#     dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1)
#     nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
#     nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

#     print(f"dx:1/{int(1/dx)}")
#     u,ksi,eta,t = solve_uw_MUSCL(init_u(Xe), init_w(Xe), init_v(Xe), phi, nt, dt, dx, methode_temps)
#     w = eta/ksi
#     u_fin = u[-1,:] ; w_fin = w[-1,:]

#     # Projection de la solution de reference
#     u_proj = u_ref[-1,:] ; w_proj = w_ref[-1,:]
#     if len(u_fin)%2==1:
#         u_fin = u_fin[:-1] ; w_fin = w_fin[:-1]
#         u_proj = u_proj[:-1] ; w_proj = w_proj[:-1]

#     r = int(np.log2(dx/dx_ref))
#     for i in range(r): #remove half of values
#         u_proj = np.reshape(u_proj,(2,int(len(u_proj)/2))) ; u_proj = u_proj[0,:]
#         w_proj = np.reshape(w_proj,(2,int(len(w_proj)/2))) ; w_proj = w_proj[0,:]

#     erru = dx*np.linalg.norm(u_fin - u_proj,ord=2)
#     errw = dx*np.linalg.norm(w_fin - w_proj,ord=2)

#     erreur_u.append(erru)
#     erreur_w.append(errw)

# x = np.array(list_dx)[:] ; y1 = np.array(erreur_u)[:] ; y2 = np.array(erreur_w)[:]

# ordre_u = np.round(np.polyfit(np.log10(x), np.log10(y1), 1)[0],2)
# ordre_w = np.round(np.polyfit(np.log10(x), np.log10(y2), 1)[0],2)

# plt.figure()
# plt.loglog(x,y1,'b.-',label=f"Résidu sur $u$, ~O({ordre_u})")
# plt.loglog(x,y1[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y1[0]*(x/x[0])**1,'--',label='O(1)')
# # plt.loglog(x,y1[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_temps}-MUSCL({limiteur}), t={tmax}")

# plt.figure()
# plt.loglog(x,y2,'r.-',label=f"Résidu sur $w$, ~O({ordre_w})")
# plt.loglog(x,y2[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y2[0]*(x/x[0])**1,'--',label='O(1)')
# # plt.loglog(x,y2[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_temps}-MUSCL({limiteur}), t={tmax}")

