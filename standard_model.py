import numpy as np

from scipy.optimize import minimize_scalar

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

#############################################################
# The agents problem and response function to a given       #
# strategy of the principal.                                #
#############################################################

def U_0(e, lam):
	return lam*e-0.5*pow(e,2)

def U_1(e, tmax, lam):
	return e*tmax+(1.-e)*lam*e-0.5*pow(e,2)

def U_2(e, tmin, tmax, lam):
	return e*tmax+(1.-e)*tmin-0.5*pow(e,2)

def U_3(e, tmin, lam):
	return (1-e)*tmin + (lam-0.5)*pow(e,2)

def U(e, tmin, tmax, lam):
	return max(U_0(e, lam), U_1(e, tmax, lam), U_2(e, tmin, tmax, lam), U_3(e,tmin,lam))

# return a tupel (s,e) where s is the agents strategy (0,1,2 as in the paper)
# and e the effort exercised
def agent_response(tmin, tmax, lam):
	x = [0.5*pow(lam,2), U_1((tmax+lam)/(2*lam+1.), tmax, lam), U_2(tmax-tmin, tmin, tmax, lam)]

	if abs(tmax-tmin) > 1: # interior maximum outside [0,1]
		x[2] = -1	

	if tmax > 1+lam: # interior maximum outside [0,1]
		x[1] = -1 

	if tmax-0.5 > max(x): # border case
		return (2,1.)

	s = np.argmax(x)
	e = [lam, (tmax+lam)/(2*lam+1.), tmax-tmin][s]

	return (s,e)

def agent_strategy(tmin, tmax, lam):
	return agent_response(tmin,tmax,lam)[0]

def agent_effort(tmin, tmax, lam):
	return agent_response(tmin,tmax,lam)[1]

#############################################################
# The problem of the principal given lambda,                #
# smin and smax                                             #
#############################################################

# return the principals payoff given his strategy
def principal_payoff(tmin, tmax, lam, smin, smax):
	(s,e) = agent_response(tmin,tmax,lam)

	return [0, e*(smax-tmax), e*(smax-tmax) + (1.-e)*(smin-tmin)][s]

# return the principals optimal no-separation tmin as a function of dt
def principal_no_separation_tmin(lam, dt):
	# use bisection
	tmin_0 = 0
	tmin_1 = 1+lam

	while tmin_1-tmin_0 > 1e-12:
		t = (tmin_1+tmin_0)/2.

		if agent_strategy(t,t+dt,lam) == 2:
			tmin_1 = t
		else:
			tmin_0 = t

	# assert we are up to an error of 1e-5
	assert(agent_strategy(tmin_1, tmin_1+dt,lam)==2)	
	assert(not(agent_strategy(tmin_1-1e5, tmin_1-1e5+dt,lam)==2))

	return tmin_1 

# return the principals optimal no-separation contract as (tmin, tmax)
def principal_optimal_no_separation(lam, smin, smax):
	# maximize the principals payoff as a function of dt
	res = minimize_scalar(lambda dt: -dt*(smax-smin-dt)-smin+principal_no_separation_tmin(lam,dt), method='bounded', bounds=(0,1), tol=1e-8)
	assert(res.success)

	tmin = principal_no_separation_tmin(lam,res.x)

	return (tmin,tmin+res.x)

# return the principals optimal partial-separation contract as (tmin, tmax)
def principal_optimal_partial_separation(lam, smin, smax):
	# the smallest tmax s.t. the agent still stays with the prinicipal
	t_pp = lam*(np.sqrt(1+2*lam)-1.)+1e-12

	# assert we are up to an error of 1e-5
	assert(agent_strategy(0,t_pp,lam)==1)
	assert(agent_strategy(0,t_pp-1e-5,lam)==0)
	
	# maximize the principals payoff
	res = minimize_scalar(lambda tmax: -(tmax+lam)/(1.+2*lam)*(smax-tmax), method='bounded', bounds=(t_pp,1+lam), tol=1e-8)
	assert(res.success)

	return (0,res.x)

# return the prinicpals globally optimal contract as (tmin, tmax)
def principal_optimal_strategy(lam, smin, smax):
	# find optimal no-separation contract
	t_ns = principal_optimal_no_separation(lam,smin,smax)

	# find optimal partial-separation contract
	t_ps = principal_optimal_partial_separation(lam,smin,smax)

	# choose the better of the two
	if principal_payoff(t_ns[0],t_ns[1],lam,smin,smax) >= principal_payoff(t_ps[0],t_ps[1],lam,smin,smax):
		return t_ns

	return t_ps

#############################################################
# Properties of the equilibrium contract                    #
#############################################################

# return the lowest value of smin for that we have no separation
def equilibirium_no_separation_smin(lam, smax):
	assert(smax>=lam)

	if equilibrium_agent_strategy(lam, lam, smax) == 2:
		return lam

	# use bisection
	smin_0 = lam
	smin_1 = smax

	while smin_1-smin_0 > 1e-12:
		s = (smin_1+smin_0)/2.

		if equilibrium_agent_strategy(lam,s,smax) == 2:
			smin_1 = s
		else:
			smin_0 = s

	# assert we are up to an error of 1e-5
	assert(equilibrium_agent_strategy(lam,smin_1,smax) == 2)
	assert(not(equilibrium_agent_strategy(lam,smin_1-1e-5,smax) == 2))

	return smin_1

#############################################################
# Plot the agents problem                                   #
#############################################################

def plot_U(tmin, tmax, lam):
	x = np.arange(0,1,0.01)
	y_u = x.copy()
	y_u_0 = x.copy()
	y_u_1 = x.copy()
	y_u_2 = x.copy()
	y_u_3 = x.copy()

	for i, xx in enumerate(x):
		y_u[i] = U(xx, tmin, tmax, lam)
		y_u_0[i] = U_0(xx, lam)
		y_u_1[i] = U_1(xx, tmax, lam)
		y_u_2[i] = U_2(xx, tmin, tmax, lam)
		y_u_3[i] = U_3(xx, tmin, lam)

	plt.plot(x, y_u_0)
	plt.plot(x, y_u_1)
	plt.plot(x, y_u_2)
	plt.plot(x, y_u_3)
	plt.plot(x, y_u, 'r--', linewidth=5.0, color='k')
	plt.xlabel("Effort level e")
	plt.ylabel("Expected utility")
	plt.xlim(0, 1)
	plt.show()

def plot_agent_strategies(lam):
	x = np.arange(0,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy<=xx:
				Z[j,i] = agent_strategy(yy,xx,lam)
			else:
				Z[j,i] = -1

	plt.figure()
	cs = plt.contourf(X, Y, Z, levels=[-0.1,0.9,1.9,2.1], colors=('b', 'g', 'r'))
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, x, linewidth='2', color='k')
	#plt.xlabel("t max")
	#plt.ylabel("t min")
	plt.show()

def plot_agent_effort(lam):
	x = np.arange(0,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z0 = np.zeros((len(x),len(y)))
	Z1 = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy<=xx:
				(s,e) = agent_response(yy,xx,lam)
				Z0[j,i] = e
				Z1[j,i] = s
			else:
				Z0[j,i] = -1
				Z1[j,i] = -1

	plt.figure()
	cs = plt.contourf(X, Y, Z0, cmap=cm.Blues, levels=np.arange(0,1.001,0.01))
	cbar = plt.colorbar(cs)
	#cbar.ax.set_ylabel('effort')
	plt.plot(x, x, linewidth='2', color='k')
	plt.contour(X, Y, Z1, linewidth='2', colors='k')
	#plt.xlabel("t max")
	#plt.ylabel("t min")
	plt.show()

def plot_agent_payoff(lam):
	x = np.arange(0,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z0 = np.zeros((len(x),len(y)))
	Z1 = np.zeros((len(x),len(y)))
	pmax = -1000

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy<=xx:
				(s,e) = agent_response(yy,xx,lam)
				Z0[j,i] = U(e,yy,xx,lam)
				Z1[j,i] = s
				pmax = max(pmax,Z0[j,i])
			else:
				Z0[j,i] = -1
				Z1[j,i] = -1

	pmax = np.ceil(pmax*10)/10.
	plt.figure()
	cs = plt.contourf(X, Y, Z0, cmap=cm.Greens, levels=np.arange(0,pmax,0.01)) 
	cbar = plt.colorbar(cs)
	#cbar.ax.set_ylabel('payoff')
	plt.plot(x, x, linewidth='2', color='k')
	plt.contour(X, Y, Z1, linewidth='2', colors='k')
	#plt.xlabel("t max")
	#plt.ylabel("t min")
	plt.show()

#############################################################
# Plot the principials problem                              #
#############################################################

def plot_principal_no_separation_problem(lam, smin, smax):
	# plots tmin as a function of dt
	x = np.arange(0,1,0.001)
	y = x.copy()

	for i, xx in enumerate(x):
		y[i] = principal_no_separation_tmin(lam,xx)

	plt.plot(x, y)
	plt.xlabel("dt")
	plt.ylabel("tmin")
	plt.xlim(0, 1)
	plt.show()

	# plots the principals payoff as a function of dt
	z = x.copy()

	for i, xx in enumerate(x):
		z[i] = xx*(smax-smin-xx)+smin-y[i]

	plt.plot(x, z)
	plt.xlabel("dt")
	plt.ylabel("principal expected payoff")
	plt.xlim(0, 1)
	plt.show()

def plot_principal_partial_separation_problem(lam, smin, smax):
	t_pp = lam*(np.sqrt(1+2*lam)-1.)+1e-12

	x = np.arange(t_pp,1+lam,0.01)
	y = x.copy()

	for i, xx in enumerate(x):
		y[i] = (xx+lam)/(1.+2*lam)*(smax-xx)

	# plots the principals payoff as a function of dt
	plt.plot(x, y)
	plt.xlabel("tmax")
	plt.ylabel("principal expected payoff")
	plt.xlim(t_pp, 1+lam)
	plt.show()

# plot the dependence of the principals payoff on tmin,tmax
def plot_principal_payoff(lam, smin, smax):
	x = np.arange(0,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z0 = np.zeros((len(x),len(y)))
	Z1 = np.zeros((len(x),len(y)))
	pmin = 1000
	pmax = -1000

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy<=xx:
				(s,e) = agent_response(yy,xx,lam)
				Z0[j,i] = principal_payoff(yy,xx,lam,smin,smax)
				Z1[j,i] = s
				pmin = min(pmin,Z0[j,i])
				pmax = max(pmax,Z0[j,i])
			else:
				Z0[j,i] = -10
				Z1[j,i] = -1

	pmin = np.floor(pmin*10)/10.
	pmax = np.ceil(pmax*10)/10.
	plt.figure()
	cs = plt.contourf(X, Y, Z0, cmap=cm.Greens, levels=np.arange(pmin,pmax,0.01))
	cbar = plt.colorbar(cs)
	#cbar.ax.set_ylabel('payoff')
	plt.plot(x, x, linewidth='2', color='k')
	plt.contour(X, Y, Z1, linewidth='2', colors='k')
	#plt.xlabel("t max")
	#plt.ylabel("t min")
	(tmin,tmax) = principal_optimal_strategy(lam,smin,smax)
	plt.plot(tmax,tmin,'r.')
	plt.show()

#############################################################
# Plot properties of the equilibirum contract               #
#############################################################

def plot_s_space(lam):
	x = np.arange(lam,3.5,0.01)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z0 = np.zeros((len(x),len(y))) # strategy
	Z1 = np.zeros((len(x),len(y))) # agent's surplus
	Z2 = np.zeros((len(x),len(y))) # first-best effort?
	rmin = 1000
	rmax = -1000

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy<=xx:
				(tmin,tmax) = principal_optimal_strategy(lam,yy,xx)
				(s,e) = agent_response(tmin,tmax,lam)
				Z0[j,i] = s
				Z1[j,i] = abs(U(e,tmin,tmax,lam)-0.5*pow(lam,2))
				rmin = min(rmin,Z1[j,i])
				rmax = max(rmax,Z1[j,i])
				if abs(e-min(1,xx-yy))<1e-4:
					Z2[j,i] = 1
				else:
					Z2[j,i] = 0
			else:
				Z0[j,i] = -1
				Z1[j,i] = -1
				Z2[j,i] = -1

	rmin = np.floor(rmin*10)/10.
	rmax = np.ceil(rmax*10)/10.
	plt.figure()
	#cs = plt.contourf(X, Y, Z1, cmap=cm.Blues, levels=np.arange(rmin,rmax,0.01))
	#cbar = plt.colorbar(cs)
	#cbar.ax.set_ylabel('rent')

	cs = plt.contour(X, Y, Z2, levels=[0.5,1.5], linestyles='dashed', linewidth='2', colors='k')
	
	plt.contour(X, Y, Z0, levels=[-0.1,0.9,1.9,2.1], linewidth='2', colors='k')
	#plt.xlabel("s max")
	#plt.ylabel("s min")
	plt.plot(x, x, linewidth='2', color='k')
	plt.show()

def plot_equilibrium_smin(lam):
	x = np.arange(lam,3.5,0.01)
	y = x.copy()

	for i, xx in enumerate(x):
		print xx
		y[i] = equilibirium_no_separation_smin(lam, xx)

	plt.plot(x, y, linewidth='2', color='k')
	plt.plot(x, x, linewidth='2', color='k')
	plt.xlabel("smax")
	plt.ylabel("smin")
	plt.xlim(lam, 3.5)
	plt.ylim(lam, 3.5)
	plt.show()

#############################################################
# Executing code for figure generation                      #
#############################################################

# figure 1
#plot_U(0.4,0.6,0.8)
#plot_U(0.4,0.9,0.8)

# figure 2
#plot_agent_strategies(0.9)
#plot_agent_effort(0.9)

# figure 3
#plot_agent_payoff(0.9)
#plot_principal_payoff(0.9,1.5,2.5)

# figure 4
#plot_s_space(0.5)
#plot_s_space(0.9)