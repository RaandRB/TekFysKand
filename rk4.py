def step(f, v, t, dt, H):
    """Take one RK4 step. Return updated solution and time.
    f: Right-hand-side function: dv/dt = f(v)
    v: current solution
    t: current time
    dt: time step
    H: hamiltonian
    """

    # Compute rates k1-k4
    k1 = dt*f(H,v)
    k2 = dt*f(H,v + 0.5*k1)
    k3 = dt*f(H,v + 0.5*k2)
    k4 = dt*f(H,v + k3) 
    # Update solution and time
    # print(v)
    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4) 
    # print(v)
    # x = input("step [enter]")
    t = t + dt
    return v, t

