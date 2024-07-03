# Crypt-model
### Implementation of a PDE model of intestinal crypt.   

$(u,w)$-system : 

$$
\begin{align}
    &\partial_t u = \partial_x\big( u\partial_x u \big),\\
    &\partial_t w = \partial_x u\,\partial_x w.
\end{align}
$$

$(u_1,u_2)$-system :  

$$
\begin{align}
    &\partial_t u_1  = \partial_x \Big(u_1 \partial_x \big(u_1 +u_2 \big)\Big),\\
    &\partial_t u_2  = \partial_x \Big(u_2 \partial_x \big(u_1 +u_2 \big)\Big).
\end{align}
$$


## Small description of each file (each of these code files is independent of one another)
- *(u,w)-system_Upwind_and_more.py* :
Upwind, Central, Characteristics, IMEX-Upwind, IMEX-Central and LagrangeRemap schemes for the $(u,w)$-system, with $v$. TVD, entropy, convergence order.  

- *(u,w)-system_MUSCL-RK4.py* :
Euler, RK2 or RK4 in time and MUSCL (minmod, superbee, osher or ultrabee flux limiter) in space for the $(u,w)$-system (written in $(u,\xi,\eta)$), with $v$. TVD, convergence order.  

- *(u,w)-system_Lagrangian.py* :
Lagrangian scheme for the $(u,w)$-system.  

- *(u1,u2)-system_Upwind,MUSCL.py* :
Euler, RK2, RK4 in time and Upwind, MUSCL (minmod, superbee, osher, ultrabee), or LagrangeRemap in space for the $(u_1,u_2)$-system, with $v$. TVD, convergence order.

- *(sc,pc,gc,ent,dcs)_Upwind.py* :
Upwind scheme for the complete model, with source terms and $\phi$.  

## Instructions and general structure of the code  
Choose the scheme you want with the "methode_u", "methode_w" or "methode_temps", "methode_espace" variables, choose the initial condition profils you want, and choose what you to plot (all by commenting/decommenting code lines) then run the code.  

### Each file is composed of several parts :  
- Firt part : Definition of the functions used for the numerical scheme, the definition of the initial conditions, and some parameters of the model.
- Second part : where you choose the schemes you want + the values of $dx,tmax$ and $cfl$, then solve.
- Third part : Plot of the solutions, reconstructions of the solutions $u_1,u_2$ for the $(u,w)$-system...
- Last part : TVD, convergence orders, entropy...
