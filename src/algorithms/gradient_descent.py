def estimate_gradient(f,d):
    pass

def GD(x0,direction,alpha,ITERATIONS):
    i=0
    x_i = x0
    while i < ITERATIONS:
        direction = estimate_gradient(fun,data)
        x_i = x0+alpha*direction
    return x_i