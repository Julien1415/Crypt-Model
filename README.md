# Crypt-model
Implementation of a PDE model of intestinal crypt.  
Each of these code files is independent of one another.  

-- Small description of each file :  
- (u,w)-system_Upwind_and_more : Upwind,Central,Characteristics,IMEX-Upwind, IMEX-Central and LagrangeRemap scheme for the $(u,w)$-system, with $v$.  

- (u,w)-system_MUSCL-RK4 : Euler, RK2 or RK4 in time and MUSCL with minmod, superbee, osher or ultrabee flux limiter in space for the $(u,w)$-system (written in $(u,\xi,\eta)$), with $v$.  

- (u,w)-system_Lagrangian : Lagrangian scheme for the $(u,w)$-system.  

- (u1,u2)-system_Upwind,MUSCL : Euler, RK2, RK4 in time and Upwind, MUSCL (minmod, superbee, osher, ultrabee), or LagrangeRemap in space for the $(u_1,u_2)$-system, with $v$.

- (sc,pc,gc,ent,dcs)_Upwind : Upwind scheme for the complete model, with source terms and $\phi$.  

-- Instructions :  Choose the scheme you want with the "methode_u", "methode_w" or "methode_temps", "methode_espace" variables, choose the initial condition profils you want, then run the code.  
