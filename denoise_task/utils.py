import numpy as np

    
def Cayley_Transform_Deriv(grads, A, W, D, _AplusIinv):
    # Calculate Update Matrix
    Update = np.dot(np.dot(_AplusIinv.T, grads), D + W.T) 
    DFA = Update.T - Update    
    return DFA    
    
def makeW(A, D, _AplusIinv, DFA, lr, exact=False):
    # Computing hidden to hidden matrix using the relation 
    I = np.identity(A.shape[0])
    Temp = np.linalg.lstsq(I+A, I, rcond=None)[0]
    Temporary = neuman_series_appx(I, _AplusIinv, DFA, lr) # ~(I+A)^-1
    if exact:
        W = np.dot(np.matmul(Temp, I - A), D)
        return W, Temp
    else:
        W = np.dot(np.matmul(Temporary, I - A), D)
        return W, Temporary
    
def neuman_series_appx(I,_AplusIinv,DFA,lr,order=2):
    Ainv_deltaA = np.dot(_AplusIinv, lr*DFA)
    if order==2:
        return np.dot(I + Ainv_deltaA + np.linalg.matrix_power(Ainv_deltaA, 2), _AplusIinv)
    else:
        return np.dot(I + Ainv_deltaA, _AplusIinv)
