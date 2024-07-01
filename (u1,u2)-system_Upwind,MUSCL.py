# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:42:58 2024

@author: Julie
"""

import numpy as np ; import matplotlib.pyplot as plt
from time import time ; from datetime import timedelta

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
        d = (pbig[3:-1]-pbig[2:-2]) #dénominateur de r
        d[d<1e-15] = 1e-5 #remplace les valeurs <1e-15 par 1e-15
        # d[np.abs(d)<1e-15] = 1e-15
        # d[d==0.0] = 1e-10
        r = (pbig[2:-2]-pbig[1:-3])/d
        # print(d,r)
        return pbig[2:-2] + 0.5*phi(r,cfl)*(pbig[3:-1]-pbig[2:-2])

    elif sign=="+1/2,R":
        d = (pbig[4:]-pbig[3:-1])
        d[d<1e-15] = 1e-5
        # d[np.abs(d)<1e-15] = 1e-15
        # d[d==0.0] = 1e-10
        r = (pbig[3:-1]-pbig[2:-2])/d
        # print(d,r)
        return pbig[3:-1] - 0.5*phi(r,cfl)*(pbig[4:]-pbig[3:-1])

    elif sign=="-1/2,L":
        d = (pbig[2:-2]-pbig[1:-3])
        d[d<1e-15] = 1e-5
        # d[np.abs(d)<1e-15] = 1e-15
        # d[d==0.0] = 1e-10
        r = (pbig[1:-3]-pbig[0:-4])/d
        # print(d,r)
        return pbig[1:-3] - 0.5*phi(r,cfl)*(pbig[2:-2]-pbig[1:-3])

    elif sign=="-1/2,R":
        d = (pbig[3:-1]-pbig[2:-2])
        d[d<1e-15] = 1e-5
        # d[np.abs(d)<1e-15] = 1e-15
        # d[d==0.0] = 1e-10
        r = (pbig[2:-2]-pbig[1:-3])/d
        # print(d,r)
        return pbig[2:-2] - 0.5*phi(r,cfl)*(pbig[3:-1]-pbig[2:-2])

def f(uk,ukk,dx,phi,methode_espace):
    if methode_espace=="MUSCL":
        y = - 1/dx**2*( mobi_MUSCL(uk,phi,"+1/2,L")*np.maximum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0) \
                      + mobi_MUSCL(uk,phi,"+1/2,R")*np.minimum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0) \
                      - mobi_MUSCL(uk,phi,"-1/2,L")*np.maximum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0) \
                      - mobi_MUSCL(uk,phi,"-1/2,R")*np.minimum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0) )

        # a = mobi_MUSCL(uk,phi,"+1/2,L")*np.maximum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0) \
          # + mobi_MUSCL(uk,phi,"+1/2,R")*np.minimum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0) \
          # - mobi_MUSCL(uk,phi,"-1/2,L")*np.maximum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0) \
          # - mobi_MUSCL(uk,phi,"-1/2,R")*np.minimum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0)
        # print(np.sum(a))
        # print(a[-1])

    elif methode_espace=="Upwind":
        y = - 1/dx**2*( np.maximum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0)*uk[1:-1] \
                      + np.minimum(uk[1:-1]+ukk[1:-1]-uk[2:]-ukk[2:],0)*uk[2:] \
                      - np.maximum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0)*uk[:-2] \
                      - np.minimum(uk[:-2]+ukk[:-2]-uk[1:-1]-ukk[1:-1],0)*uk[1:-1] )

    return y

def solve_u1u2_MUSCL(u1_0,u2_0,v,phi,Xt,nt,dt,dx,methode_espace,methode_temps):
    tps = time()

    u1 = np.zeros((1,len(u1_0)))
    u1[0,:] = u1_0
    u1old = u1_0

    u2 = np.zeros((1,len(u2_0)))
    u2[0,:] = u2_0
    u2old = u2_0

    mem = 100 ; mem = min(nt,mem)
    list_t = [0]

    for n in range(1,nt):
    # for n in range(3):

        u1new = np.zeros_like(u1old)
        u2new = np.zeros_like(u2old)

        if methode_temps=="Euler":

            u1new[1:-1] = u1old[1:-1] + dt*f(u1old,u2old+v,dx,phi,methode_espace)
            u2new[1:-1] = u2old[1:-1] + dt*f(u2old,u1old+v,dx,phi,methode_espace)

        elif methode_temps=="RK2":

            ytilde = u1old[1:-1] + dt/2*f(u1old,u2old+v,dx,phi,methode_espace) ; ytilde = np.hstack([ytilde[0],ytilde,ytilde[-1]])
            u1new[1:-1] = u1old[1:-1] + dt*f(ytilde,u2old+v,dx,phi,methode_espace)

            ytilde = u2old[1:-1] + dt/2*f(u2old,u1old+v,dx,phi,methode_espace) ; ytilde = np.hstack([ytilde[0],ytilde,ytilde[-1]])
            u2new[1:-1] = u2old[1:-1] + dt*f(ytilde,u1old+v,dx,phi,methode_espace)

        elif methode_temps=="RK4":

            k1 = f(u1old,u2old+v,dx,phi,methode_espace) ; k1 = np.hstack([k1[0],k1,k1[-1]])
            k2 = f(u1old+dt/2*k1,u2old+v,dx,phi,methode_espace) ; k2 = np.hstack([k2[0],k2,k2[-1]])
            k3 = f(u1old+dt/2*k2,u2old+v,dx,phi,methode_espace) ; k3 = np.hstack([k3[0],k3,k3[-1]])
            k4 = f(u1old+dt*k3,u2old+v,dx,phi,methode_espace) ; k4 = np.hstack([k4[0],k4,k4[-1]])
            u1new[1:-1] = u1old[1:-1] + dt/6*(k1[1:-1]+2*k2[1:-1]+2*k3[1:-1]+k4[1:-1])
    
            k1 = f(u2old,u1old+v,dx,phi,methode_espace) ; k1 = np.hstack([k1[0],k1,k1[-1]])
            k2 = f(u2old+dt/2*k1,u1old+v,dx,phi,methode_espace) ; k2 = np.hstack([k2[0],k2,k2[-1]])
            k3 = f(u2old+dt/2*k2,u1old+v,dx,phi,methode_espace) ; k3 = np.hstack([k3[0],k3,k3[-1]])
            k4 = f(u2old+dt*k3,u1old+v,dx,phi,methode_espace) ; k4 = np.hstack([k4[0],k4,k4[-1]])
            u2new[1:-1] = u2old[1:-1] + dt/6*(k1[1:-1]+2*k2[1:-1]+2*k3[1:-1]+k4[1:-1])

        #Conditions aux bords
        u1new[0] = u1new[1] ; u1new[-1] = u1new[-2]
        u2new[0] = u2new[1] ; u2new[-1] = u2new[-2]

        # u1new[0] = u1new[1] + u2new[1] - u2old[0] ; u1new[-1] = u1new[-2] + u2new[-2] - u2old[-1]
        # u2new[0] = u2new[1] + u1new[1] - u1old[0] ; u2new[-1] = u2new[-2] + u1new[-2] - u2old[-1]

        u1old = u1new
        u2old = u2new

        if (n%(nt//mem)==0) or (n==nt-1): #garde en mémoire une partie des itérations
            print(f"\r{int(len(list_t)/mem*100)}% ",end="",flush=True) #print avancement
            u1 = np.vstack([u1,u1new])
            u2 = np.vstack([u2,u2new])
            list_t.append(n)
    print("",f"{methode_temps}-{methode_espace}(u1,u2) :",np.round(timedelta(seconds=time()-tps).total_seconds(),3),"secondes")

    return u1,u2,list_t

def init_u1(x):
    return np.sqrt(np.maximum(0,1-64*(0.25-x)**2))
    # return np.sqrt(np.maximum(0,1-64*(0.2-x)**2)) + np.sqrt(np.maximum(0,1-64*(0.8-x)**2))
    # return np.sqrt(np.maximum(0,1-64*(0.4-x)**2))

def init_u2(x):
    return np.sqrt(np.maximum(0,1-64*(0.75-x)**2))
    # return np.sqrt(np.maximum(0,1-64*(0.75-x)**2))*0.8
    # return np.sqrt(np.maximum(0,1-64*(0.6-x)**2))

a = 0.05

def init_v(x): #bosse/creneau
    a1 = 1/2 - a ; a2 = 1/2 + a
    # return np.sqrt(np.maximum(0,1-10*64*(0.5-x)**2))*0.3
    # return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))
    return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))*0.2
    # return (np.where(a1<x,1.0,0)*np.where(x<a2,1.0,0))*(1+1/2*np.sin(2*np.pi*(x-1/2+a)/(2*a)))*0.265
    # return 0.0*x

#%% Choose your fighter

long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.015 ; dx = 1/200
dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1) #dt following the cfl and such as the inverse is an integer
nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)

u1_0 = init_u1(Xe) ; u2_0 = init_u2(Xe) ; u0 = u1_0 + u2_0 ; v = init_v(Xe)

methode_temps = "Euler"
# methode_temps = "RK2"
# methode_temps = "RK4"

methode_espace = "Upwind"
# methode_espace = "MUSCL"
# methode_espace = "LagrangeRemap"

phi = 0
# phi, limiteur = minmod, "minmod"
# phi, limiteur = superbee, "superbee"
# phi, limiteur = osher, "osher"
# phi, limiteur = ultrabee, "ultrabee"

cfl = dt/dx**2 ; print(f"N={nx+1}, nt={nt}, dt/dx**2 = {np.round(cfl,4)}")

#Solve
u1,u2,t = solve_u1u2_MUSCL(u1_0,u2_0,v,phi,Xt,nt,dt,dx,methode_espace,methode_temps)

m1_inf = dx*np.sum(u1[-1,:]) ; m2_inf = dx*np.sum(u2[-1,:]) ; print(m1_inf+m2_inf)
m1 = dx*np.sum(u1_0) ; m2 = dx*np.sum(u2_0) ; mv = dx*np.sum(v) ; xshock = m1/(m1+m2)

#%% Plot u1,u2

# Initial conditions u1_0,u2_0,v
# plt.figure() ; plt.plot(Xe,u1_0,'b--',label="u0") ; plt.plot(Xe,u2_0,'r--',label="w0") ; plt.plot(Xe,v,'m-',label="v") ; plt.ylim(0,1.1) ; plt.xlabel("x") ; plt.legend() ; plt.title(f"Conditions initiales $u_1^0,u_2^0,v$, N={nx+1}")

#Flux limiters
# plt.figure() ; plt.plot(np.linspace(0,3,100),minmod(np.linspace(0,3,100)),'b--',label="minmod") ; plt.plot(np.linspace(0,3,100),superbee(np.linspace(0,3,100)),'r--',label="superbee")  ; plt.xlim(0,3) ; plt.ylim(0,3) ; plt.xlabel("x") ; plt.grid() ; plt.legend() ; plt.title("Limiteur de flux")

def plot_u1u2(u1,u2,v,u1_0,u2_0,t,Xe,methode_temps,methode_espace):
    plt.figure()
    for n in range(np.shape(u1)[0]):
        plt.cla()
        if mv>0.0 and m1/(1/2-a)>np.max(v): #v non nul et u1>v
            plt.plot(Xe,u1[n,:]+v,'b.-',label="$u_1+v$")
            plt.plot(Xe,u1_0,'b--',label="$u_1(t=0)$")
            plt.plot(Xe,u2[n,:]+v,'r.-',label="$u_2+v$")
            plt.plot(Xe,u2_0,'r--',label="$u_2(t=0)$")
            plt.plot(Xe,v,'m-',label="$v$")
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="$u1+u2+v$")
            plt.plot(Xe,(m1+m2+mv)*Xe**0,'k--',label="Temps long")
        elif mv>0.0 and m1/(1/2-a)<np.max(v): #v non nul et u1<v
            plt.plot(Xe,u1[n,:],'b.-',label="$u_1$")
            plt.plot(Xe,u1_0,'b--',label="$u_1(t=0)$")
            plt.plot(Xe,u2[n,:],'r.-',label="$u_2$")
            plt.plot(Xe,u2_0,'r--',label="$u_2(t=0)$")
            plt.plot(Xe,v,'m-',label="$v$")
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="$u1+u2+v$")
            plt.plot(Xe,m1/(1/2-a)*Xe**0,'k--',label="Temps long")
        else: #v=0
            plt.plot(Xe,u1[n,:],'b-',label="$u_1$")
            plt.plot(Xe,u1_0,'b--',label="$u_1(t=0)$")
            plt.plot(Xe,u2[n,:],'r-',label="$u_2$")
            plt.plot(Xe,u2_0,'r--',label="$u_2(t=0)$")
            # plt.plot(Xe,u1[n,:]+u2[n,:]+v,'k-',label="$u1+u2$")
            # plt.plot(Xe,m1/xshock*(Xe*0+1),'k--',label="masse $u_0$")
            plt.plot(Xe,(m1+m2)*Xe**0,'k--',label="masse $u_0$")
            # plt.vlines(xshock,0,2,'k',linestyle='dashdot',label="$x_{shock}$")
            plt.vlines(xshock,0,m1/xshock,'k',linestyle='dashed')
            plt.text(xshock,-0.03,"$x_{shock}$")
        plt.ylim(-0.03,1.1)
        plt.xlim(0,0.99)
        plt.legend()
        plt.title(f"Solutions $u_1,u_2$ en {methode_temps}-{methode_espace}, N={nx+1}, t={round(t[n]*dt,3)}")
        # plt.title(f"Solutions u1,u2 en {methode_temps}-MUSCL({limiteur}), N={nx+1}, t={round(t[n]*dt,3)}")
        plt.xlabel(f"dt/dx**2={np.round(cfl,4)}, max(u0)={np.round(np.max(np.abs(u0)),4)}, norm1(u0)={np.round(m1+m2,5)}")
        plt.pause(0.01+0.1/(n**2+1))

plot_u1u2(u1,u2,v,u1_0,u2_0,t,Xe,methode_temps,methode_espace)

#%% TVD

# plt.figure()
# plt.plot(Xt[t],[np.sum(np.abs(u1[n,:-1]-u1[n,1:])) for n in range(len(t))],'k.-',label="$TV(u_1)$")
# plt.ylim(0,2.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $u_1$ pour {methode_temps}-MUSCL({limiteur}), N={nx+1}")

# plt.figure()
# plt.plot(Xt[t],[np.sum(np.abs(u2[n,:-1]-u2[n,1:])) for n in range(len(t))],'k.-',label="$TV(u_2)$")
# plt.ylim(0,2.1)
# plt.xlabel("t")
# plt.legend()
# plt.title(f"Semi-norme TVD de $u_2$ pour {methode_temps}-MUSCL({limiteur}), N={nx+1}")

#%% Convergence order

# methode_temps = "Euler"
# # methode_temps = "RK2"
# methode_espace = "Upwind"
# # methode_espace = "MUSCL"
# # phi, limiteur = minmod, "minmod"

# nb_point = 3
# long = 1.0 ; coef_cfl = 0.495 ; tmax = 0.25 ; dx_ref = 1/(50*2**nb_point)

# dt_ref = coef_cfl*dx_ref**2 ; dt_ref = 1/(int(1/dt_ref)+1)
# nx = round(long/dx_ref-1) ; Xe_ref,dx_ref = np.linspace(0,long,nx+2,retstep=True)
# nt = round(tmax/dt_ref+1) ; Xt_ref,dt_ref = np.linspace(0,tmax,nt,retstep=True)
# u1_0 = init_u1(Xe_ref) ; u2_0 = init_u2(Xe_ref) ; u0 = u1_0 + u2_0 ; v = init_v(Xe_ref)

# print(f"dx:1/{int(1/dx_ref)}")
# u1_ref,u2_ref,t_ref = solve_u1u2_MUSCL(u1_0,u2_0,v,phi,Xt_ref,nt,dt_ref,dx_ref,methode_espace,methode_temps)

# list_dx = [1/(50*2**i) for i in range(nb_point)]
# erreur_u1 = [] ; erreur_u2 = []

# for dx in list_dx:
#     dt = coef_cfl*dx**2 ; dt = 1/(int(1/dt)+1)
#     nx = round(long/dx-1) ; Xe,dx = np.linspace(0,long,nx+2,retstep=True)
#     nt = round(tmax/dt+1) ; Xt,dt = np.linspace(0,tmax,nt,retstep=True)
#     u1_0 = init_u1(Xe) ; u2_0 = init_u2(Xe) ; u0 = u1_0 + u2_0 ; v = init_v(Xe)

#     print(f"dx:1/{int(1/dx)}")
#     u1,u2,t = solve_u1u2_MUSCL(u1_0,u2_0,v,phi,Xt,nt,dt,dx,methode_espace,methode_temps)
#     u1_fin = u1[-1,:] ; u2_fin = u2[-1,:]

#     # Projection (moyennes sur deux mailles de la solution de reference)
#     r = int(np.log2(dx/dx_ref))
#     u1_proj = u1_ref[-1,:] ; u2_proj = u2_ref[-1,:]

#     if len(u1_fin)%2==1:
#         u1_fin = u1_fin[:-1] ; u2_fin = u2_fin[:-1]
#         u1_proj = u1_proj[:-1] ; u2_proj = u2_proj[:-1]

#     for i in range(r):
#         u1_proj = np.reshape(u1_proj,(2,int(len(u1_proj)/2)))
#         # u1_proj = np.mean(u1_proj,axis=0)
#         u1_proj = u1_proj[0,:]
#         u2_proj = np.reshape(u2_proj,(2,int(len(u2_proj)/2)))
#         # u2_proj = np.mean(u2_proj,axis=0)
#         u2_proj = u2_proj[0,:]

#     err1 = dx*np.linalg.norm(u1_fin - u1_proj,ord=2)
#     err2 = dx*np.linalg.norm(u2_fin - u2_proj,ord=2)

#     erreur_u1.append(err1)
#     erreur_u2.append(err2)

# #%%
# x = np.array(list_dx) ; y1 = np.array(erreur_u1) ; y2 = np.array(erreur_u2)

# ordre_u1 = np.round(np.polyfit(np.log10(x), np.log10(y1), 1)[0],2)
# ordre_u2 = np.round(np.polyfit(np.log10(x), np.log10(y2), 1)[0],2)

# plt.figure()
# plt.loglog(x,y1,'b.-',label=f"Résidu sur $u_1$, ~O({ordre_u1})")
# # plt.loglog(x,y1[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y1[0]*(x/x[0])**1,'--',label='O(1)')
# plt.loglog(x,y1[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_temps}-{methode_espace}, t={tmax}")

# plt.figure()
# plt.loglog(x,y2,'r.-',label=f"Résidu sur $u_2$, ~O({ordre_u2})")
# plt.loglog(x,y2[0]*(x/x[0])**0,'--',label='O(0)')
# plt.loglog(x,y2[0]*(x/x[0])**1,'--',label='O(1)')
# # plt.loglog(x,y2[0]*(x/x[0])**2,'--',label='O(2)')
# plt.xlabel(f"$N+1\in${[int(1/dx) for dx in x]}")
# plt.legend(loc=4)
# plt.title(f"Courbes d'erreurs pour {methode_temps}-{methode_espace}, t={tmax}")

