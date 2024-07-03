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
Upwind, Central, Characteristics, IMEX-Upwind, IMEX-Central and LagrangeRemap schemes for the $(u,w)$-system, with $v$.  

- *(u,w)-system_MUSCL-RK4.py* :
Euler, RK2 or RK4 in time and MUSCL (minmod, superbee, osher or ultrabee flux limiter) in space for the $(u,w)$-system (written in $(u,\xi,\eta)$), with $v$.  

- *(u,w)-system_Lagrangian.py* :
Lagrangian scheme for the $(u,w)$-system.  

- *(u1,u2)-system_Upwind,MUSCL.py* :
Euler, RK2, RK4 in time and Upwind, MUSCL (minmod, superbee, osher, ultrabee), or LagrangeRemap in space for the $(u_1,u_2)$-system, with $v$.

- *(sc,pc,gc,ent,dcs)_Upwind.py* :
Upwind scheme for the complete model, with source terms and $\phi$.  

## Instructions  
Choose the scheme you want with the "methode_u", "methode_w" or "methode_temps", "methode_espace" variables, choose the initial condition profils you want, then run the code.  
