import numpy

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
    
    Parameters
    ------------
    eta : float
        Tasa de aprendizaje (un valor entre 0.0 y 1.0)
    n_iter : int
        Iteraciones sobre el conjunto de entrenamiento.
    random_state : int
        Semilla del generador de números aleatorios para
        la inicialización de los pesos aleatorios.
    
    Attributes
    -----------
    w_ : 1d-array
        Pesos después del entrenamiento.
    b_ : Scalar
        Unidad de sesgo después del fitting.
    losses_ : list
        Mean squared error loss function values en cada época.

    """

    