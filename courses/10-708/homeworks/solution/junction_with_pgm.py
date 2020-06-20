"""
Script to verify my results using the library
pgmpy
"""

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation


def factors():
    """
    initialise the initial factor
    """
    phi = dict()

    # marginal on A
    phi['a'] = TabularCPD(variable='a',
            variable_card = 2,
            values = np.array([[0.05,0.95]])
            )

    #CPD on B|A
    phi['ab'] = TabularCPD('b', 2, np.array([[0.1, 0.9],[0.2,0.8]]).T,
            ['a'], [2])

    ##CPD ON E|A
    #phi['ae'] = DiscreteFactor(['a','e'],[2,2], np.array([[0.3,
    #    0.7],[0.4,0.6]]))
    phi['ae'] = TabularCPD('e', 2, np.array([[0.3,0.4],[0.7,0.6]]),
            ['a'],[2])

    ##CPD ON c|b
    #phi['bc'] = DiscreteFactor(['b','c'],[2,2], np.array([[0.5,
    #    0.5],[0.6,0.4]]))
    phi['bc'] = TabularCPD('c',2,np.array([[0.5,0.6],[0.5,0.4]]),
            ['b'], [2])


    #CPD on D|B,c
    A = np.array([[[0.7,0.3],[0.8,0.2]],
        [[0.9, 0.1], [0.99,0.01]]
        ]).T
    A = A.reshape(A.shape[0],-1)
    phi['ced'] = TabularCPD('d',2,A ,
        ['c','e'],[2,2])


    return phi

def model():
    """
    Define the bayesian model
    """
    #getting the factors
    phi = factors()

    # A model in pgmpy is defined by a list of edges
    edges =[('a','b'),('a', 'e'),('b','c'),('c','d'),('e','d')]

    #creating the model
    M = BayesianModel(edges)
    for cpd in phi:
        M.add_cpds(phi[cpd])

    return M


def main2():

    #creating the model
    M = model()
    M.check_model()

    #some operation on the model
    print("initial edges")
    print(M.edges())

    #triangulation
    print("triangulating the model")
    H = M.to_markov_model()
    factors = [f for f in H.get_factors()]
    f = factors[0]
    for v in factors[1:3]:
        f.product(v, inplace=True)
    print(f)


def main():
    """
    Main functiont to lunch the belief propagation algorithm
    """

    M = model()
    M.check_model()

    #Belief propagation
    bp  = BeliefPropagation(M)
    bp.calibrate()
    print("maximal cliques are:")
    print(bp.get_cliques())

    # first query 
    print("computing probability of B=")
    query1 = bp.query(variables=list('b'),show_progress=True)
    print(query1)


    #second query 
    print("computing probability of B|C")
    query2 = bp.query(variables = ['b','c'])
    query2.marginalize(['c'])
    print(query2)

    #Third query
    print("computing joint")
    query3 = bp.query(['a','b','c','d','e'])
    query3.normalize()
    print(query3)


if __name__ == "__main__":

    #Create the initial factors
    main()
