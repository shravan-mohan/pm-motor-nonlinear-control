import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.integrate import odeint

def getControllerParams(y, M):
    """
    This function generates the response time optimal controller parameters for a permanent
    magnet motor, resulting in zero torque ripple under no measurement noise. It is assumed, for simplicity,
    that the control inputs are the currents instead of voltages. These currents, once known, can be
    used to determine the control voltage inputs.
    :param y: This is the torque vs theta function, which makes the permanent magnet motor
    model nonlinear.
    :param M: This is the order of the Fourier series approximation of the nonlinearity.
    :return: Returns the controller parameters, without the proportionality constant.
    """
    plt.figure()
    plt.plot(np.linspace(-np.pi,np.pi,len(y)),y)
    plt.title('The plot of f(theta)')


    p = np.zeros(M)
    q = np.zeros(M)
    for k in range(M):
        p[k] = 2*(np.cos((k+1)*np.linspace(-np.pi,np.pi,len(y)))@y)/len(y)
        q[k] = 2*(np.sin((k+1)*np.linspace(-np.pi,np.pi,len(y)))@y)/len(y)

    s = np.zeros(len(y))
    x = np.linspace(-np.pi,np.pi,len(y))
    for k in range(len(y)):
        s[k] = (np.cos(np.linspace(1,M,M)*x[k])@p + np.sin(np.linspace(1,M,M)*x[k])@q)

    plt.figure()
    plt.plot(s)
    plt.plot(y)
    plt.title('Comparison with the Fourier truncation')

    Z = np.zeros([4*M+1,2*M])

    #cos and cos
    for k in range(1,M+1):
      for l in range(1,M+1):
        if(l==k):
          Z[0,l-1] = Z[0,l-1] + p[k-1]/2
          Z[2*(k+l)-1,l-1] = Z[2*(k+l)-1,l-1] + p[k-1]/2
        else:
          Z[2*(k+l)-1,l-1] = Z[2*(k+l)-1,l-1] + p[k-1]/2
          Z[2*np.abs(k-l)-1,l-1] = Z[2*abs(k-l)-1,l-1] + p[k-1]/2

    #cos and sin
    for k in range(1,M+1):
      for l in range(1,M+1):
        if(l==k):
          Z[2*(k+l)+1-1,l+M-1] = Z[2*(k+l)+1-1,l+M-1] + p[k-1]/2
        else:
          Z[2*(k+l)+1-1,l+M-1] = Z[2*(k+l)+1-1,l+M-1] + p[k-1]/2
          Z[2*abs(k-l)+1-1,l+M-1] = Z[2*abs(k-l)+1-1,l+M-1] + np.sign(l-k)*p[k-1]/2

    # sin and cos
    for k in range(1,M+1):
        for l in range(1,M+1):
            if (l == k):
                Z[2 * (k + l) + 1-1, l-1] = Z[2 * (k + l) + 1-1, l-1] + q[k-1] / 2
            else:
                Z[2 * (k + l) + 1-1, l-1] = Z[2 * (k + l) + 1-1, l-1] + q[k-1] / 2
                Z[2 * np.abs(k - l) + 1-1, l-1] = Z[2 * np.abs(k - l) + 1-1, l-1] + np.sign(k - l) * q[k-1] / 2

    # sin and sin
    for k in range(1,M+1):
        for l in range(1,M+1):
            if (l == k):
                Z[1-1, l + M-1] = Z[1-1, l + M-1] + q[k-1] / 2
                Z[2 * (k + l)-1, l + M-1] = Z[2 * (k + l)-1, l + M-1] - q[k-1] / 2
            else:
                Z[2 * (k + l)-1, l + M-1] = Z[2 * (k + l)-1, l + M-1] - q[k-1] / 2
                Z[2 * abs(k - l)-1, l + M-1] = Z[2 * abs(k - l)-1, l + M-1] + q[k-1] / 2

    A = np.zeros([4 * M, 4 * M])
    for i in range(1, 2 * M + 1):
        A[2 * (i - 1), 2 * (i - 1)] = np.cos(2 * np.pi * i / 3)
        A[2 * (i - 1), 2 * i - 1] = np.sin(2 * np.pi * i / 3)
        A[2 * i - 1, 2 * (i - 1)] = -np.sin(2 * np.pi * i / 3)
        A[2 * i - 1, 2 * i - 1] = np.cos(2 * np.pi * i / 3)

    A = np.vstack((np.hstack((1, np.zeros(4 * M))), np.hstack((np.zeros([4 * M, 1]), A))))

    B = np.zeros([4 * M, 4 * M])
    for i in range(1, 2 * M + 1):
        B[2 * (i - 1), 2 * (i - 1)] = np.cos(4 * np.pi * i / 3)
        B[2 * (i - 1), 2 * i - 1] = np.sin(4 * np.pi * i / 3)
        B[2 * i - 1, 2 * (i - 1)] = -np.sin(4 * np.pi * i / 3)
        B[2 * i - 1, 2 * i - 1] = np.cos(4 * np.pi * i / 3)

    B = np.vstack((np.hstack((1, np.zeros(4 * M))), np.hstack((np.zeros([4 * M, 1]), B))))

    G = np.hstack((Z, A @ Z, B @ Z))

    As1 = np.vstack((np.hstack((np.zeros(M-1), 0)), np.hstack((np.eye(M-1), np.zeros([M-1,1])))))
    Bs1 = np.vstack((1, np.zeros([M-1,1])))
    As2 = np.vstack((np.hstack((np.zeros(M-1), 0)), np.hstack((np.eye(M-1), np.zeros([M-1,1])))))
    Bs2 = np.vstack((1, np.zeros([M-1,1])))
    As3 = np.vstack((np.hstack((np.zeros(M-1), 0)), np.hstack((np.eye(M-1), np.zeros([M-1,1])))))
    Bs3 = np.vstack((1, np.zeros([M-1,1])))

    p1 = cvx.Variable((1,M))
    q1 = cvx.Variable((1,M))
    p2 = cvx.Variable((1,M))
    q2 = cvx.Variable((1,M))
    p3 = cvx.Variable((1,M))
    q3 = cvx.Variable((1,M))
    Q1l = cvx.Variable((M,M), hermitian=True)
    Q2l = cvx.Variable((M,M), hermitian=True)
    Q3l = cvx.Variable((M,M), hermitian=True)
    Q1u = cvx.Variable((M,M), hermitian=True)
    Q2u = cvx.Variable((M,M), hermitian=True)
    Q3u = cvx.Variable((M,M), hermitian=True)
    Ds1u = cvx.Variable()
    Ds2u = cvx.Variable()
    Ds3u = cvx.Variable()
    Ds1l = cvx.Variable()
    Ds2l = cvx.Variable()
    Ds3l = cvx.Variable()
    z = cvx.Variable()
    Cs1u = cvx.Variable((1,M), complex=True)
    Cs2u = cvx.Variable((1,M), complex=True)
    Cs3u = cvx.Variable((1,M), complex=True)
    Cs1l = cvx.Variable((1,M), complex=True)
    Cs2l = cvx.Variable((1,M), complex=True)
    Cs3l = cvx.Variable((1,M), complex=True)
    r1u = cvx.Variable((1,M), complex=True)
    r2u = cvx.Variable((1,M), complex=True)
    r3u = cvx.Variable((1,M), complex=True)
    r1l = cvx.Variable((1,M), complex=True)
    r2l = cvx.Variable((1,M), complex=True)
    r3l = cvx.Variable((1,M), complex=True)

    constraints = []
    constraints = constraints + [G@(cvx.hstack((p1, q1, p2, q2, p3, q3)).T) == np.vstack((1, np.zeros([4*M,1])))]
    constraints = constraints + [cvx.real(r1u) == -p1/2,  cvx.imag(r1u) == q1/2]
    constraints = constraints + [cvx.real(r2u) == -p2/2,  cvx.imag(r2u) == q2/2]
    constraints = constraints + [cvx.real(r3u) == -p3/2,  cvx.imag(r3u) == q3/2]
    constraints = constraints + [cvx.real(r1l) == p1/2,  cvx.imag(r1l) == -q1/2]
    constraints = constraints + [cvx.real(r2l) == p2/2,  cvx.imag(r2l) == -q2/2]
    constraints = constraints + [cvx.real(r3l) == p3/2,  cvx.imag(r3l) == -q3/2]
    constraints = constraints + [Cs1u == r1u]
    constraints = constraints + [Cs2u == r2u]
    constraints = constraints + [Cs3u == r3u]
    constraints = constraints + [Cs1l == r1l]
    constraints = constraints + [Cs2l == r2l]
    constraints = constraints + [Cs3l == r3l]
    constraints = constraints + [Ds1u == z / 2]
    constraints = constraints + [Ds2u == z / 2]
    constraints = constraints + [Ds3u == z / 2]
    constraints = constraints + [Ds1l == z / 2]
    constraints = constraints + [Ds2l == z / 2]
    constraints = constraints + [Ds3l == z / 2]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q1u-(As1.T)@Q1u@As1, -(As1.T)@Q1u@Bs1)), cvx.hstack((-cvx.conj((As1.T)@Q1u@Bs1).T, -(Bs1.T)@Q1u@Bs1)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs1u).T)), cvx.hstack((-Cs1u, cvx.reshape(Ds1u+Ds1u, [1,1]))))) >> 0]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q2u-(As2.T)@Q2u@As2, -(As2.T)@Q2u@Bs2)), cvx.hstack((-cvx.conj((As2.T)@Q2u@Bs2).T, -(Bs2.T)@Q2u@Bs2)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs2u).T)), cvx.hstack((-Cs2u, cvx.reshape(Ds2u+Ds2u, [1,1]))))) >> 0]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q3u-(As3.T)@Q3u@As3, -(As3.T)@Q3u@Bs3)), cvx.hstack((-cvx.conj((As3.T)@Q3u@Bs3).T, -(Bs3.T)@Q3u@Bs3)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs3u).T)), cvx.hstack((-Cs3u, cvx.reshape(Ds3u+Ds3u, [1,1]))))) >> 0]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q1l-(As1.T)@Q1l@As1, -(As1.T)@Q1l@Bs1)), cvx.hstack((-cvx.conj((As1.T)@Q1l@Bs1).T, -(Bs1.T)@Q1l@Bs1)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs1l).T)), cvx.hstack((-Cs1l, cvx.reshape(Ds1l+Ds1l, [1,1]))))) >> 0]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q2l-(As2.T)@Q2l@As2, -(As2.T)@Q2l@Bs2)), cvx.hstack((-cvx.conj((As2.T)@Q2l@Bs2).T, -(Bs2.T)@Q2l@Bs2)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs2l).T)), cvx.hstack((-Cs2l, cvx.reshape(Ds2l+Ds2l, [1,1]))))) >> 0]
    constraints = constraints + [cvx.vstack((cvx.hstack((Q3l-(As3.T)@Q3l@As3, -(As3.T)@Q3l@Bs3)), cvx.hstack((-cvx.conj((As3.T)@Q3l@Bs3).T, -(Bs3.T)@Q3l@Bs3)))) + cvx.vstack((cvx.hstack((np.zeros([M,M]), -cvx.conj(Cs3l).T)), cvx.hstack((-Cs3l, cvx.reshape(Ds3l+Ds3l, [1,1]))))) >> 0]

    prob = cvx.Problem(cvx.Minimize(z), constraints)
    prob.solve(solver=cvx.SCS,verbose=True)

    return p, q, z.value, p1.value, p2.value, p3.value, q1.value, q2.value, q3.value


def motor_model(x,t,y,p,q,p1,p2,p3,q1,q2,q3,M,omega_r,KTAU,Tin,K):
    """
    This model returns the vector field of the motor model.
    :param x: This vector is the state space vector (theta, omega)
    :param t: This is the time instant.
    :param y: Torque vs theta nonlinear function defining the motor nonlinearity.
    :param p: The cosine Fourier coefficients of the M-order approximation of y.
    :param q: The sine Fourier coefficients of the M-order approximation of y.
    :param p1: The cosine Fourier coefficients of the M-order controller for phase-1.
    :param p2: The sine Fourier coefficients of the M-order controller for phase-1.
    :param p3: The cosine Fourier coefficients of the M-order controller for phase-2.
    :param q1: The sine Fourier coefficients of the M-order controller for phase-2.
    :param q2: The cosine Fourier coefficients of the M-order controller for phase-3.
    :param q3: The sine Fourier coefficients of the M-order controller for phase-3.
    :param M: The order of the Fourier series approximation of the nonlinearity y.
    :param omega_r: The desired speed of the motor.
    :param KTAU: The torque constant.
    :param Tin: The opposing constant input torque.
    :param K: The proportionality constant of the controller, which ensures that the currents lie
     within the actuation bounds.
    :return: The vector field.
    """
    xdot = []
    tmpx2 = x[1]+0*(np.random.rand()-0.5)
    xdot = xdot + [tmpx2]
    h1 = (np.cos(np.linspace(1,M,M)*x[0])@p + np.sin(np.linspace(1,M,M)*x[0])@q + 0*(np.random.rand()-0.5))*(np.cos(np.linspace(1,M,M)*x[0])@p1.flatten() + np.sin(np.linspace(1,M,M)*x[0])@q1.flatten())
    h2 = (np.cos(np.linspace(1,M,M)*(x[0]+2*np.pi/3))@p + np.sin(np.linspace(1,M,M)*(x[0]+2*np.pi/3))@q + 0*(np.random.rand()-0.5))*(np.cos(np.linspace(1,M,M)*(x[0]+2*np.pi/3))@p2.flatten() + np.sin(np.linspace(1,M,M)*(x[0]+2*np.pi/3))@q2.flatten())
    h3 = (np.cos(np.linspace(1,M,M)*(x[0]+4*np.pi/3))@p + np.sin(np.linspace(1,M,M)*(x[0]+4*np.pi/3))@q + 0*(np.random.rand()-0.5))*(np.cos(np.linspace(1,M,M)*(x[0]+4*np.pi/3))@p3.flatten() + np.sin(np.linspace(1,M,M)*(x[0]+4*np.pi/3))@q3.flatten())
    Tin = Tin + 0*(np.random.rand()-0.5)
    xdot = xdot + [KTAU*K*(omega_r-tmpx2+Tin/(K*KTAU))*(h1+h2+h3+0*(np.random.rand()-0.5)) - Tin]

    return xdot


def simulate(model, y0, t):
    y = odeint(model, y0, t)

    plt.figure()
    plt.plot(t, y[:, 1])
    plt.title('Plot of angular speed with time')

    I1 = np.zeros(len(y[:, 0]))
    I2 = np.zeros(len(y[:, 0]))
    I3 = np.zeros(len(y[:, 0]))

    for k in range(len(y[:, 0])):
        I1[k] = K * (omega_r - y[k, 1] + Tin / (K * KTAU)) * (
                    np.cos(np.linspace(1, M, M) * y[k, 0]) @ p1.flatten() + np.sin(
                np.linspace(1, M, M) * y[k, 0]) @ q1.flatten())
        I2[k] = K * (omega_r - y[k, 1] + Tin / (K * KTAU)) * (
                    np.cos(np.linspace(1, M, M) * (y[k, 0] + 2 * np.pi / 3)) @ p2.flatten() + np.sin(
                np.linspace(1, M, M) * (y[k, 0] + 2 * np.pi / 3)) @ q2.flatten())
        I3[k] = K * (omega_r - y[k, 1] + Tin / (K * KTAU)) * (
                    np.cos(np.linspace(1, M, M) * (y[k, 0] + 4 * np.pi / 3)) @ p3.flatten() + np.sin(
                np.linspace(1, M, M) * (y[k, 0] + 4 * np.pi / 3)) @ q3.flatten())

    plt.figure()
    plt.plot(I1)
    plt.plot(I2)
    plt.plot(I3)
    plt.title('Plot of currents with time')

if(__name__=='__main__'):

    # Motor Params
    Imax = 30
    TAUmax = 10
    KTAU = 5

    # y is the torque angle function
    y = np.hstack((np.linspace(0,np.pi/4,100),(np.pi/4+0.5*np.linspace(0,np.pi/4,100)),(np.pi/4+np.pi/8-0.2*np.linspace(0,np.pi/4,100))))
    y = np.hstack((y, np.flip(y)))
    y = np.hstack((-y, y))
    M = 10

    ## Get Controller
    p, q, z, p1, p2, p3, q1, q2, q3 = getControllerParams(y, M)

    ## Define operating conditions
    omega_r = 40*np.pi
    Tin  = 5
    t = np.linspace(0,5,10000)
    x0 = [0,0]

    K = (Imax/z + TAUmax/KTAU)/np.abs(omega_r-x0[1])
    model_n = lambda x,t : motor_model(x,t,y,p,q,p1,p2,p3,q1,q2,q3,M,omega_r,KTAU,Tin,K)

    # Simulate the motor model
    simulate(model_n, x0, t)


