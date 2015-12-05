import numpy as np

from scipy.optimize import minimize_scalar

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

#############################################################
# The agents problem and response function to a given       #
# strategy of the principal.                                #
#############################################################

#################### Custom model ###########################
def u(x):
	assert(x<=2.5)
	return x-0.2*pow(x,2)

def c(e):
	return 0.25*pow(e,4)

def v(e):
	return 0.5*(2*e-pow(e,2))

#################### Standard model #########################
# def u(x):
# 	return x

# def c(e):
# 	return 0.5*pow(e,2)

# def v(e):
# 	return 0.5*e

def U_0(e):
	return u(v(e)-c(e))

def U_1(e, tmax):
	return e*u(tmax-c(e)) + (1-e)*u(v(e)-c(e))

def U_2(e, tmin, tmax):
	return e*u(tmax-c(e)) + (1-e)*u(tmin-c(e))

def U(e, tmin, tmax):
	return max(U_0(e), U_1(e,tmax), U_2(e,tmin,tmax))	

# return a tupel (s,e) where s is the agents strategy (0,1,2 as in the paper)
# and e the effort exercised
def agent_response(tmin, tmax):
	# maximize the respective strategies
	res0 = minimize_scalar(lambda e: -U_0(e), method='bounded', bounds=(0,1))
	assert(res0.success)

	res1 = minimize_scalar(lambda e: -U_1(e,tmax), method='bounded', bounds=(0,1))
	assert(res1.success)

	res2 = minimize_scalar(lambda e: -U_2(e,tmin,tmax), method='bounded', bounds=(0,1))
	assert(res2.success)

	s = np.argmin([res0.fun, res1.fun, res2.fun])
	e = [res0.x, res1.x, res2.x][s]

	return (s,e)

def agent_strategy(tmin, tmax):
	return agent_response(tmin,tmax)[0]

def agent_effort(tmin, tmax):
	return agent_response(tmin,tmax)[1]

#############################################################
# The problem of the principal given smin and smax          #
#############################################################

# return the principals payoff given his strategy
def principal_payoff(tmin, tmax, smin, smax):
	(s,e) = agent_response(tmin,tmax)

	return [0, e*(smax-tmax), e*(smax-tmax) + (1.-e)*(smin-tmin)][s]

# return the principals optimal no-separation tmin as a function of dt
def principal_no_separation_tmin(dt):
	# use bisection
	tmin_0 = 0
	tmin_1 = 2

	while tmin_1-tmin_0 > 1e-12:
		t = (tmin_1+tmin_0)/2.

		if agent_strategy(t,t+dt) == 2:
			tmin_1 = t
		else:
			tmin_0 = t

	# assert we are up to an error of 1e-5
	assert(agent_strategy(tmin_1, tmin_1+dt)==2)	
	assert(not(agent_strategy(tmin_1-1e5, tmin_1-1e5+dt)==2))

	return tmin_1 

# return the principals optimal no-separation contract as (tmin, tmax)
def principal_optimal_no_separation(smin, smax):
	# maximize the principals payoff as a function of dt
	res = minimize_scalar(lambda dt: -dt*(smax-smin-dt)-smin+principal_no_separation_tmin(dt), method='bounded', bounds=(0,1))
	assert(res.success)

	tmin = principal_no_separation_tmin(res.x)

	return (tmin,tmin+res.x)

# return the smallest tmax s.t. the agent still stays with the prinicipal
def principal_partial_separation_tmax():
	# use bisection
	tmax_0 = 0
	tmax_1 = 2

	while tmax_1-tmax_0 > 1e-12:
		t = (tmax_1+tmax_0)/2.

		if agent_strategy(0,t) == 1:
			tmax_1 = t
		else:
			tmax_0 = t

	# assert we are up to an error of 1e-5
	assert(agent_strategy(0, tmax_1)==1)	
	assert(agent_strategy(0, tmax_1-1e5)==0)

	return tmax_1 

# return the principals optimal partial-separation contract as (tmin, tmax)
def principal_optimal_partial_separation(smin, smax):
	tmax_0 = principal_partial_separation_tmax()
	
	# maximize the principals payoff
	res = minimize_scalar(lambda tmax: -agent_effort(0,tmax)*(smax-tmax), method='bounded', bounds=(tmax_0,2))
	assert(res.success)

	return (0,res.x)

# return the prinicpals globally optimal contract as (tmin, tmax)
def principal_optimal_strategy(smin, smax):
	# find optimal no-separation contract
	t_ns = principal_optimal_no_separation(smin,smax)

	# find optimal partial-separation contract
	t_ps = principal_optimal_partial_separation(smin,smax)

	# choose the better of the two
	if principal_payoff(t_ns[0],t_ns[1],smin,smax) >= principal_payoff(t_ps[0],t_ps[1],smin,smax):
		return t_ns

	return t_ps

#############################################################
# Properties of the equilibrium contract                    #
#############################################################

# return the agents strategy in equilibrium
def equilibrium_agent_strategy(smin, smax):
	t = principal_optimal_strategy(smin,smax)

	return agent_strategy(t[0],t[1])

# return the lowest value of smin for that we have no separation
# (must have no separation for (smax, smax))
def equilibirium_no_separation_smin(smax):
	assert(equilibrium_agent_strategy(smax,smax) == 2)
 
	if equilibrium_agent_strategy(0, smax) == 2:
		return 0

	# use bisection
	smin_0 = 0
	smin_1 = smax

	while smin_1-smin_0 > 1e-1:
		s = (smin_1+smin_0)/2.

		if equilibrium_agent_strategy(s,smax) == 2:
			smin_1 = s
		else:
			smin_0 = s

	# assert we are up to an error of 1e-1
	assert(equilibrium_agent_strategy(smin_1,smax) == 2)
	assert(not(equilibrium_agent_strategy(smin_1-1e-1,smax) == 2))

	return smin_1

def equilibrium_principal_payoff(smin, smax):
	t = principal_optimal_strategy(smin,smax)

	return principal_payoff(t[0],t[1],smin,smax)


#############################################################
# Plot the agents problem                                   #
#############################################################

def plot_U(tmin, tmax):
	x = np.arange(0,1,0.01)
	y_u = x.copy()
	y_u_0 = x.copy()
	y_u_1 = x.copy()
	y_u_2 = x.copy()

	for i, xx in enumerate(x):
		y_u[i] = U(xx, tmin, tmax)
		y_u_0[i] = U_0(xx)
		y_u_1[i] = U_1(xx, tmax)
		y_u_2[i] = U_2(xx, tmin, tmax)

	plt.plot(x, y_u_0)
	plt.plot(x, y_u_1)
	plt.plot(x, y_u_2)
	plt.plot(x, y_u, 'r--', linewidth=5.0, color='k')
	plt.xlabel("Effort level e")
	plt.ylabel("Expected utility")
	plt.xlim(0, 1)

	plt.show()

def plot_agent_strategies():
	x = np.arange(0,2,0.01)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if yy <= xx:
				Z[j,i] = agent_strategy(yy,xx)
			else:
				Z[j,i] = -1

	plt.figure()
	cs = plt.contourf(X, Y, Z, cmap=cm.coolwarm)
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, x, linewidth='2', color='k')
	plt.xlabel("t max")
	plt.ylabel("t min")

	plt.show()

def plot_agent_effort():
	x = np.arange(0,2,0.025)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			Z[j,i] = agent_effort(yy,xx)

			if yy > xx:
				Z[j,i] = 0

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1)

	ax.set_xlabel("t max")
	ax.set_ylabel("t min")
	ax.set_zlabel("effort e")

	plt.show()

#############################################################
# Plot the principials problem                              #
#############################################################

def plot_principal_no_separation_problem(smin, smax):
	# plots tmin as a function of dt
	x = np.arange(0,1,0.01)
	y = x.copy()

	for i, xx in enumerate(x):
		print xx
		y[i] = principal_no_separation_tmin(xx)

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

def plot_principal_partial_separation_problem(smin, smax):
	tmax_0 = principal_partial_separation_tmax()

	x = np.arange(tmax_0,2,0.01)
	y = x.copy()

	for i, xx in enumerate(x):
		print xx
		y[i] = agent_effort(0,xx)*(smax-xx)

	# plots the principals payoff as a function of dt
	plt.plot(x, y)
	plt.xlabel("tmax")
	plt.ylabel("principal expected payoff")
	plt.xlim(tmax_0, 2)
	plt.show()

# plot the dependence of the principals payoff on smin,smax
def plot_principal_payoff(lam):
	x = np.arange(lam,1+lam,0.05)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(y),len(x)))

	for i,xx in enumerate(x):
		print xx
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			Z[j,i] = equilibrium_principal_payoff(lam,yy,xx)

			if yy > xx:
				Z[j,i] = 0

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1)
	
	ax.set_xlabel("s max")
	ax.set_ylabel("s min")
	ax.set_zlabel("payoff")

	plt.show()

#############################################################
# Plot properties of the equilibirum contract               #
#############################################################

def plot_strategy_versus_s():
	x = np.arange(0,3.5,0.1)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(y),len(x)))

	for i,xx in enumerate(x):
		print xx

		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			if xx >= yy:
				Z[j,i] = equilibrium_agent_strategy(yy,xx)
			else:
				Z[j,i] = -1

	plt.figure()
	cs = plt.contourf(X, Y, Z)
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, y, linewidth='2', color='k')
	plt.xlabel("s max")
	plt.ylabel("s min")
	plt.show()

def plot_equilibrium_smin():
	x = np.arange(0,3.5,0.1)
	y = x.copy()

	for i, xx in enumerate(x):
		print xx

		if equilibrium_agent_strategy(xx,xx) == 2:
			y[i] = equilibirium_no_separation_smin(xx)
		else:
			y[i] = smax
		
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

# x = np.arange(-1,2.5,0.001)
# y = x.copy()

# for i,xx in enumerate(x):
# 	y[i] = u(xx)

# plt.figure()
# plt.plot(x,x)
# plt.plot(x,y)
# plt.show()

#plot_U(0.4,0.6)
#plot_U(0.7,1.6)

#print agent_response(0.9, 1.5)

#plot_agent_strategies()

#plot_agent_effort()



smin = 1.2
smax = 3.2
#plot_principal_no_separation_problem(smin,smax)
#plot_principal_partial_separation_problem(smin,smax)
#print principal_optimal_strategy(smin,smax)

plot_equilibrium_smin()

#plot_principal_payoff(0.9)


#plot_strategy_versus_s()


#print equilibrium_agent_strategy(0.5, 1.3, 3.3)
