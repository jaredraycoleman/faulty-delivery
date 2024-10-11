import math
import matplotlib.pyplot as plt
import argparse

def dist(x,y):
    return math.sqrt(x**2 + y**2)

def computeOptimal(x, y, t, r=0.0):
    return max(dist(x-t-r, y), t) + 1 - t

def computeTime(x, y, m, t, r=0.0):
    HM = dist(x-m, y)
    H = t/dist(x-m,y)
    xprime = x - (x-m)*(H)
    yprime = y*(1-H)  
    if t < HM: return dist(xprime - t, yprime) + 1
    if t <= m: return HM + 1 - 2*t + m
    else: return HM + 1 - m
    

def CR(x, y, m, t) :
    Opt = computeOptimal(x, y, t)
    T = computeTime(x,y, m, t)
    return T/Opt
 
def optimalForGivenM(x, y, m) :
    epsilon = 0.001
    l = 0.0
    r = m
    mid = (l + r)/2
    CRm = CR(x,y,m, mid)
    CRmE = CR(x,y,m, mid+epsilon)
    i = 1
    while True:
        #print(mid, CRmE, CRm, l, r)
        if CRm < CRmE: l = mid
        else: r = mid   
        if abs(l - r) < 0.00001: break
        mid = (l + r)/2
        CRm = CR(x,y, m, mid)
        CRmE = CR(x,y, m, mid+epsilon)
        
    return (CRm, mid)
    
def computeOptimalCR(x, y) :
    H = dist(x,y)
    r = min(H**2  / (2*x), 1)
    l = 0
    m = (l + r)/2
    CRright = CR(x,y,m,1)
    (CRLeft, worstCase) = optimalForGivenM(x, y, m)
    #print(CRLeft, worstCase, l, r)
    while True:
        if CRright < CRLeft: r = m
        else: l = m
        if abs(l - r) < 0.000001: break
        m = (l + r)/2
        #print(l, r, m, CRright, CRLeft)
        CRright = CR(x,y,m,1)
        (CRLeft, worstCase) = optimalForGivenM(x,y, m)
    #print(f"CR {CRright} with m {m} and worst cases error {worstCase}")
    return (m, CRright, CRLeft, worstCase)

def plotSetting(x, y, m, worst, xprime = -1, yprime = -1, m1 = -1) :
    if m1 <0:
        m1 = m
    if xprime <0:
        xprime = x
    if yprime <0:
        yprime = x

    H = worst/dist(x-m,y)
    xprime = x - (x-m)*(H)
    yprime = y*(1-H)  
    fig, ax = plt.subplots()
    ax.plot([1,0], [0,0])
    ax.plot([0,0], [0,1])
    ax.plot([x,m], [y,0])
    ax.plot(m, 0, 'x', markeredgewidth=2)
    ax.plot(worst, 0, 'x', markeredgewidth=2)
    ax.plot(x, y, 'x', markeredgewidth=2)
    ax.plot(xprime, yprime, 'x', markeredgewidth=2)
    ax.plot(m1, 0, 'x', markeredgewidth=2)
    ax.plot(xprime, yprime, 'x', markeredgewidth=2)
    circle = plt.Circle((x, y), worst, edgecolor='blue', fill=False)
    ax.add_patch(circle)
    circle = plt.Circle((0, 0), worst, edgecolor='blue', fill=False)
    ax.add_patch(circle)
        
    plt.show()

def CRAll(x, y,  plot):
    #print(CRLeft, worst, CRright)
    (m, CRRight, CRLeft, worst) = computeOptimalCR(x,y)
    print(f"CR Left {CRLeft} right {CRRight} \nworst case {worst} with m= {m}")
 
    if plot:
        plotSetting(x, y, m, worst)
    return (CRLeft, CRRight, m, worst)
        
def CRM(x, y, m,  plot):
    #print(CRLeft, worst, CRright)
    CRRight = CR(x,y,m,1)
    (CRLeft, worst) = optimalForGivenM(x,y, m)
    print(f"CR Left {CRLeft} right {CRRight} \n worst case {worst}")
    if plot:
        plotSetting(x, y, m, worst)
    return (CRLeft, CRRight, worst)

def CRM2(x, y, m, plot, s = 2.0):
    (CRLeft, CRright, worst) = CRM(x,y,m,False)
    H = (worst*s) / dist(x - m, y)
    xprime = x - (x - m) * (H)
    yprime = y * (1 - H)
    xdoubleprime = xprime / (1 - worst) - xprime
    ydoubleprime = yprime / (1 - worst)
    print("-----------")
    print(f"new x {xprime},  and y {yprime} at time {worst}")
    print(f"Scaled x {xdoubleprime},  and y {xdoubleprime} ")
    print("-----------")

    (CRLeft1, CRright2, m1, worst) = CRAll(xdoubleprime, ydoubleprime, False)

    print(CRLeft, CRLeft1, CRright2)
    if plot:
        plotSetting(x, y, m, worst, xprime, yprime, m1*(1 - worst))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some coordinates.")
    parser.add_argument('-x', type=float, required=True, help='X coordinate')
    parser.add_argument('-y', type=float, required=True, help='Y coordinate')
    parser.add_argument('-m', type=float, required=False, help='m coordinate')
    parser.add_argument('-p', type=float, required=False, help='p points')
    parser.add_argument('-s', type=float, required=False, help='s scale')

    parser.add_argument('--plot', action='store_true', help='Flag to plot the points')
    args = parser.parse_args() 
    if args.m is not None:
        if args.p == 1:
            s = 1
            if args.s is not None:
                s =args.s
            CRM2(args.x, args.y, args.m, args.plot, s)
        else:
            CRM(args.x, args.y, args.m, args.plot)
    else:
        CRAll(args.x, args.y, args.plot)