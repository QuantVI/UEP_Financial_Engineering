# to run a massive sim of a sturctured product and find its value
import numpy as np
# define the variables
N=2000
M=40000
s0,K,r,q,vol,T = 100,100,0.05,0,0.3,1
a_calc = int(0.2*N)
b_calc = int(0.8*N)
h1_calc = s0*(1+vol)
h2_calc = max(s0*(1-vol),0) # since stock price can't be negative

# define barriers and cut-off points
limits = {
    'a': a_calc,
    'b': b_calc,
    'H1': h1_calc,
    'H2': h2_calc
}
print(limits)

# where to write data
discountfile = 'out'

# define geo brownian mtn
def g_stock_path(s0=s0,K=K,r=r,q=q,vol=vol,T=T,N=N,M=M):
    """Generates N values of stock price from time 0 to T.
    Stock price follow geometric brownian motion.
    Uses lists, instead of numpy arrays"""
    dt=T/N
    s = [s0]
    temp1 = ((r - q) - (0.5 * vol*vol))*dt
    temp2 = vol * np.sqrt(dt)
    for step in range(N):
        next_val = s[step] * np.exp(temp1 + (temp2 * np.random.randn()))
        s.append(next_val)
    return s

# where to put each path
list_of_paths = []

# generate multiple paths and put them above
for sims in range(M):
    new_path = g_stock_path(s0=s0,K=K,r=r,q=q,vol=vol,T=T,N=N,M=M)
    list_of_paths.append(new_path)
    if (sims%2000==0):
        print(sims)

# define the payoffs
def payoff_calc(sp_matrix,check_dict):
    h_list = []
    # Garrison entry and exit times
    a,b = check_dict['a'], check_dict['b']
    # Full time to maturity
    Tm = len(sp_matrix[0])-1
    # High and Low price barriers
    H1,H2 = check_dict['H1'], check_dict['H2']
    
    def garrison_breach(some_path):
        """ find the payoff in the case the product was converted due to 
        exceeding any barrier, at all, between times a and b."""

        some_path = np.array(some_path)
        
        g_payoff = 0
        
        H1_br_bool = any(some_path[a:b+1] > H1)
        H2_br_bool = any(some_path[a:b+1] < H2)
        
        # If there was an H1 or H2 breach, find the earliest index of this breach.
        # Otherwise this index is infinity.
        if H1_br_bool:
            h1bt = some_path[a:b+1] > H1
            H1_br_loc = min(np.where(h1bt)[0]) + a
        else:
            H1_br_loc = np.infty
        if H2_br_bool:
            h2bt = some_path[a:b+1] < H2
            H2_br_loc = min(np.where(h2bt)[0]) + a
        else:
            H2_br_loc = np.infty
        
        # We use only 1 underlying stock.
        # It's impossible to breach both barriers at the same time, unless H1=H2
        if H1_br_loc < H2_br_loc:
            # converted to call with strike S(a)
            #g_payoff = max(path[H1_br_loc]-path[a],0)
            g_payoff = max(path[H1_br_loc]-path[a],0)
        elif H2_br_loc < H1_br_loc:
            # converted to a put with strike S(a)
            g_payoff = max(path[a]- path[H2_br_loc],0)
        else:
            # This is an error. We return junk to force an error.
            g_payoff = "Issue in Garrison Function."
        return g_payoff
    
    for path in sp_matrix:
        path = np.array(path)
        # At time a, stock price must be between the barriers.
        if ((path[a] > H1) or (path[a] <H2)):
            h_list.append(0)
        # Between a and b, we can't exceed the barriers at all.
        # The 'any' function checks for this
        elif ( any(path[a:b+1] > H1) or any(path[a:b+1] < H2) ):
            g_payment = garrison_breach(path)
            h_list.append(g_payment)
        else:
            v_call = max(path[Tm]-path[b],0)
            v_put  = max(path[b]-path[Tm],0)
            v_payment = max(v_call,v_put)
            h_list.append(v_payment)
    return h_list

the_payoffs = payoff_calc(list_of_paths,limits)
dsc_payoffs = np.average(the_payoffs) * np.exp(-r*T)

out_put = open(discountfile,'a')
myline = str(dsc_payoffs) + ',' + str(N) + ',' + str(M) + '\n'
out_put.write(myline)
out_put.close()
