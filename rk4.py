def step(f, v, t, dt, H, w):
    """Take one RK4 step. Return updated solution and time.
    f: Right-hand-side function: dv/dt = f(v)
    v: current solution
    t: current time
    dt: time step
    H: hamiltonian
    """

    # Compute rates k1-k4
    k1 = dt*f(H,v, w)
    k2 = dt*f(H,v + 0.5*k1, w)
    k3 = dt*f(H,v + 0.5*k2, w)
    k4 = dt*f(H,v + k3, w) 
    # Update solution and time
    # print(v)
    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4) 
    # print(v)
    # x = input("step [enter]")
    t = t + dt
    return v, t

