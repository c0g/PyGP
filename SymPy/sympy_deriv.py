from numpy import sqrt, exp, pi
import numpy as np

def normal_sym(Z1,Z2,hyp):
	return sqrt(2)*exp(-0.5*(Z1[0] - Z2[0])**2*exp(-2*hyp[0]) - 0.5*(Z1[1] - Z2[1])**2*exp(-2*hyp[1]) - 0.5*(Z1[2] - Z2[2])**2*exp(-2*hyp[2]))/(4*pi**(3./2.)*sqrt(exp(2*hyp[0])*exp(2*hyp[1])*exp(2*hyp[2])))
def normal(Z1,Z2,hyp):
	el2 = np.exp(2*hyp)
	Sigma2 = np.diag(el2)
	scale = 1/np.sqrt(np.linalg.det(Sigma2) * (2.*np.pi)**3.)
	rbf = np.exp(-0.5*(Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2))))
	return rbf * scale

Z1 = np.array([1.,2.,3.])
Z2 = np.array([3.,2.,1.])
hyp = np.log(np.array([1.,5.,3.]))

def dfunc1(Z1,Z2,hyp):
	return sqrt(2)*(1.0*(Z1[0] - Z2[0])**2 - exp(2*hyp[0]))*exp(-2*hyp[0])*exp(-0.5*(Z1[0] - Z2[0])**2*exp(-2*hyp[0]) - 0.5*(Z1[1] - Z2[1])**2*exp(-2*hyp[1]) - 0.5*(Z1[2] - Z2[2])**2*exp(-2*hyp[2]))/(4*pi**(3/2)*sqrt(exp(2*hyp[0])*exp(2*hyp[1])*exp(2*hyp[2])))

def dfunc2(Z1,Z2,hyp):
	return sqrt(2)*(1.0*(Z1[1] - Z2[1])**2 - exp(2*hyp[1]))*exp(-2*hyp[1])*exp(-0.5*(Z1[0] - Z2[0])**2*exp(-2*hyp[0]) - 0.5*(Z1[1] - Z2[1])**2*exp(-2*hyp[1]) - 0.5*(Z1[2] - Z2[2])**2*exp(-2*hyp[2]))/(4*pi**(3/2)*sqrt(exp(2*hyp[0])*exp(2*hyp[1])*exp(2*hyp[2])))


def dfunc_tom(Z1,Z2,hyp):
	el2 = np.exp(hyp*2)
	Sigma2 = np.diag(el2)
	return sqrt(2)*(1.0*(Z1[0] - Z2[0])**2/el2[0] - 1
		)*np.exp(-0.5 * (Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2)))
		)/(4*pi**(3/2)*sqrt(np.linalg.det(Sigma2)))
def dfunc_tom_vec(Z1, Z2, hyp):
	el2 = np.exp(hyp*2)
	Sigma2 = np.diag(el2)
	return sqrt(2)*((Z1 - Z2)**2/el2 - 1
		)*np.exp(-0.5 * (Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2)))
		)/(4*pi**(3/2)*sqrt(np.linalg.det(Sigma2)))
def dfunc_vec(Z1, Z2,hyp):
    el2 = np.exp(hyp*2)
    Sigma2 = np.diag(el2)
    return np.sqrt(2)*((Z1 - Z2)**2/el2 - 1
        )*np.exp(-0.5 * (Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2)))
        )/(4*np.pi**(3/2)*np.sqrt(np.linalg.det(Sigma2)))

print(normal_sym(Z1, Z2, hyp))
print(normal(Z1, Z2, hyp))

print(dfunc1(Z1, Z2, hyp))
print(dfunc_tom(Z1, Z2, hyp))
print(dfunc_tom_vec(Z1, Z2, hyp)[0])

print(dfunc2(Z1, Z2, hyp))
print(dfunc_tom_vec(Z1, Z2, hyp)[1])
print(dfunc_vec(Z1, Z2, hyp)[1])

dist = 1e-10
delta = np.zeros(hyp.shape)
delta[0] = dist/2
print((normal(Z1, Z2, hyp+delta) - normal(Z1, Z2, hyp-delta)) / dist)