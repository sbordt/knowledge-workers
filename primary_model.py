import numpy as np

from scipy.optimize import minimize

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def U_0(e, lam):
	return lam*e-0.5*pow(e,2)

def U_1(e, t_max, lam):
	return e*t_max+(1-e)*lam*e-0.5*pow(e,2)

def U_2(e, t_min, t_max, lam):
	return e*(t_max-t_min)+t_min-0.5*pow(e,2) 

#	return e*t_max+(1-e)*t_min-0.5*pow(e,2)

def U(e, t_min, t_max, lam):
	return max(U_0(e, lam), U_1(e, t_max, lam), U_2(e, t_min, t_max, lam))	

# returns 0 for full separation, 1 for partial separation, 2 for no separation
def agent_strategy(t_min, t_max, lam):
	if t_max-t_min <= 1:
		return np.argmax([0.5*pow(lam,2), U_1((t_max+lam)/(2*lam+1), t_max, lam), U_2(t_max-t_min, t_min, t_max, lam)])

	return np.argmax([0.5*pow(lam,2),(lam+0.5)*pow(t_max+lam,2)/pow(2*lam+1,2)])

def agent_effort(t_min, t_max, lam):
	s = agent_strategy(t_min, t_max, lam)

	# assuming we are at an interior solution
	return [lam, (t_max+lam)/(2*lam+1), t_max-t_min][s]

def principal_payoff(t_min, t_max, lam, s_min, s_max):
	s = agent_strategy(t_min, t_max, lam)
	e = agent_effort(t_min, t_max, lam)

	return [0, e*(s_max-t_max), e*(s_max-t_max) + (1-e)*(s_min-t_min)][s]

# returns a tupel (t_min, t_max)
def principal_optimal_no_separation(lam, s_min, s_max):
	# for a given level of delta t, find the optimal t_min (using bisection)
	def optimal_t_min(dt, tol = 1e-12):
		t_min_0 = 0
		t_min_1 = lam

		while t_min_1-t_min_0 > tol:
			t = (t_min_1+t_min_0)/2.

			if agent_strategy(t,t+dt,lam) == 2:
				t_min_1 = t
			else:
				t_min_0 = t

		return t_min_1 

	# find t_min(dt) for all values of dt between 0 and 1
	# x = np.arange(0,1,0.001)
	# y = x.copy()

	# for i, xx in enumerate(x):
	# 	y[i] = optimal_t_min(xx)
		
	# 	assert(agent_strategy(y[i],y[i]+xx,lam)==2)
	# 	assert(not(agent_strategy(y[i]-1e-5,y[i]-1e-5+xx,lam)==2))

	# plots t_min versus dt
	#plt.plot(x, y)
	#plt.xlabel("dt")
	#plt.ylabel("t_min")
	#plt.xlim(0, 1)
	#plt.show()

	# find the princiapls payoff
	# z = x.copy()

	# for i, xx in enumerate(x):
	# 	z[i] = xx*(s_max-s_min-xx)+s_min-y[i]

	# # plots the principals payoff as a function of dt
	# plt.plot(x, z)
	# plt.xlabel("dt")
	# plt.ylabel("principal expected payoff")
	# plt.xlim(0, 1)
	# plt.show()

	# maximize the principals payoff as a function of dt
	res = minimize(lambda dt: -dt*(s_max-s_min-dt)-s_min+optimal_t_min(dt), (0.5,), method='L-BFGS-B', bounds=((0,1),))
	assert(res.success)

	t_min = optimal_t_min(res.x[0])

	return (t_min,t_min+res.x[0])

# returns a tupel (t_min, t_max)
def principal_optimal_partial_separation(lam, s_min, s_max):
	# determine the smallest t_max s.t. the agent still stays with the prinicipal
	# (we knwo that t_min = 0 in this context)
	# use bisection
	t_max_0 = 0
	t_max_1 = 1+lam

	while t_max_1-t_max_0 > 1e-12:
		t = (t_max_1+t_max_0)/2.

		if agent_strategy(0,t,lam) == 1:
			t_max_1 = t
		else:
			t_max_0 = t

	# what we want is now stored in t_max_1 
	assert(agent_strategy(0,t_max_1,lam)==1)
	assert(agent_strategy(0,t_max_1-1e-5,lam)==0)

	# plot the principals payoff as a function of t_max
	x = np.arange(t_max_1,1+lam,0.01)
	y = x.copy()

	for i, xx in enumerate(x):
		y[i] = (xx+lam)/(1.+2*lam)*(s_max-xx)

	# plots the principals payoff as a function of dt
	# plt.plot(x, y)
	# plt.xlabel("t_max")
	# plt.ylabel("principal expected payoff")
	# plt.xlim(t_max_1, 1+lam)
	# plt.show()
	
	# maximize the principals payoff
	res = minimize(lambda t_max: -(t_max+lam)/(1.+2*lam)*(s_max-t_max), ((t_max_1+1+lam)/2.,), method='L-BFGS-B', bounds=((t_max_1,1+lam),))
	assert(res.success)

	return (0,res.x[0])

# returns a tupel (t_min, t_max)
def principal_strategy(lam, s_min, s_max):
	bnds = ((0, None), (0, None))

	# find optimal no-separation contract
	def f_2(x):
		if agent_strategy(x[0],x[1],lam) == 2:
			return -principal_payoff(x[0],x[1],lam,s_min,s_max)
			
		return 0

	x_2 = minimize(f_2 , (0.7, 1.0), method='TNC', bounds=bnds)
	#print x_2

	# find optimal partial-separation contract
	def f_1(x):
		if agent_strategy(x[0],x[1],lam) == 1:
			return -principal_payoff(x[0],x[1],lam,s_min,s_max)
			
		return 0

	x_1 = minimize(f_1 , (0.0, 1.0), method='TNC', bounds=bnds)
	#print x_1

	if principal_payoff(x_1.x[0],x_1.x[1],lam,s_min,s_max) > principal_payoff(x_2.x[0],x_2.x[1],lam,s_min,s_max):
		return x_1.x

	return x_2.x

def plot_U(t_min,t_max,lam):
	x = np.arange(0,1,0.01)
	y_u = x.copy()
	y_u_0 = x.copy()
	y_u_1 = x.copy()
	y_u_2 = x.copy()

	for i, xx in enumerate(x):
		y_u[i] = U(xx, t_min, t_max, lam)
		y_u_0[i] = U_0(xx, lam)
		y_u_1[i] = U_1(xx, t_max, lam)
		y_u_2[i] = U_2(xx, t_min, t_max, lam)

	plt.plot(x, y_u_0)
	plt.plot(x, y_u_1)
	plt.plot(x, y_u_2)
	plt.plot(x, y_u, 'r--', linewidth=5.0, color='k')
	plt.xlabel("Effort level e")
	plt.ylabel("Expected utility")
	plt.xlim(0, 1)

	plt.show()

def plot_strategy_sets(lam):
	x = np.arange(0,1+lam,0.0025)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			Z[j,i] = agent_strategy(yy,xx,lam)

	plt.figure()
	cs = plt.contourf(X, Y, Z)
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, y, linewidth='2', color='k')
	plt.xlabel("t max")
	plt.ylabel("t min")
	plt.show()

def plot_agent_effort(lam):
	x = np.arange(0,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			Z[j,i] = agent_effort(yy,xx,lam)

			if yy > xx:
				Z[j,i] = 0

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=5, cstride=5)

	ax.set_xlabel("t max")
	ax.set_ylabel("t min")
	ax.set_zlabel("effort e")

	plt.show()

def plot_principal_payoff(lam):
	x = np.arange(lam,1+lam,0.005)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			Z[j,i] = principal_payoff(yy,xx,lam)

			i

	plt.figure()
	cs = plt.contourf(X, Y, Z)
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, y, linewidth='2', color='k')
	plt.xlabel("s max")
	plt.ylabel("s min")
	plt.show()

def plot_resulting_strategy(lam):
	x = np.arange(lam,3.5,0.05)
	y = x.copy()
	X, Y = np.meshgrid(x,y)	
	Z = np.zeros((len(x),len(y)))

	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			# The x values correspond to the column indices of Z and the y values correspond to the row indices of Z
			t = principal_strategy(lam,yy,xx)

			Z[j,i] = agent_strategy(t[0],t[1],lam)

	plt.figure()
	cs = plt.contourf(X, Y, Z)
	plt.contour(cs, linewidth='2', colors='k')
	plt.plot(x, y, linewidth='2', color='k')
	plt.xlabel("s max")
	plt.ylabel("s min")
	plt.show()


# figure 1
#plot_U(0.4,0.6,0.8)

# figure 2
#plot_U(0.4,0.9,0.8)

# figure 3 and 4
#plot_strategy_sets(0.9)
#plot_strategy_sets(0.5)

#plot_agent_effort(0.9)

#print principal_strategy(0.9, 1.1, 1.5)

#plot_resulting_strategy(0.5)

#print principal_optimal_no_separation(0.9, 0.6, 2.5)

# three possible cases
#print principal_optimal_partial_separation(0.5, 0.2, 0.5)
#print principal_optimal_partial_separation(0.5, 0.2, 2)
#print principal_optimal_partial_separation(0.5, 0.2, 6)