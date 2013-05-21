from sympy import *
import numpy as np
hyp = [Symbol('hyp[0]'),
		Symbol('hyp[1]'),
		Symbol('hyp[2]')]
el2 = [exp(hyp[0])**2,
		exp(hyp[1])**2,
		exp(hyp[2])**2]

Z1 = [Symbol('Z1[0]'),
		Symbol('Z1[1]'),
		Symbol('Z1[2]')]

Z2 = [Symbol('Z2[0]'),
		Symbol('Z2[1]'),
		Symbol('Z2[2]')]

rbfs = exp(-0.5 * ( 
	(Z1[0]-Z2[0])**2 / el2[0] + 
	(Z1[1]-Z2[1])**2 / el2[1] + 
	(Z1[2]-Z2[2])**2 / el2[2] 
	) )

normaliser = sqrt(
	el2[0] * el2[1] * el2[2] * ((2 * pi) ** 3)
	)

func = rbfs / normaliser



dfunc1 = (diff(func,hyp[0]))
dfunc2 = (diff(func,hyp[1]))
dfunc3 = (diff(func,hyp[2]))
