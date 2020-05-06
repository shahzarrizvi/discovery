import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from scipy import special
from scipy import optimize
from scipy import spatial

np.random.seed(666)


# Inhomogeneous Diffusion Condensation

names = ['Barbell', 'Tree', 'Noisy Tree', 'Clusters', 'Uniform Circle',
             'Hyperuniform Circle', 'Hyperuniform Ellipse', 'Two Spirals']

def barbell(N, beta = 1):
    '''Generate Uniformly-Sampled 2-D Barbell'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting
    k = 1
    while k <= N:
        x = (2 + beta/2)*np.random.uniform()
        y = (2 + beta/2)*np.random.uniform()
            
        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.25:
            X[0].append(x)
            X[1].append(y)
            C.append(0)
            k += 1
                
        elif abs(x - 1 - beta / 4) < beta / 4 and abs(y - 0.5) < 0.125:
            X[0].append(x)
            X[1].append(y)
            C.append(1)
            k += 1
                
        elif (x - 1.5 - beta/2)**2 + (y - 0.5)**2 <= 0.25:
            X[0].append(x)
            X[1].append(y)
            C.append(2)
            k += 1
                
    return np.asarray(X), np.asarray(C)
    
    
def tree(N, radius = 1, levels = 3):
    '''Generate Uniformly-Sampled 2-D Tree with 2**(levels) Branches'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting
        
    s = 0; root = [s, s] # root node position
    omega = np.pi / 4 # half of anlge between branches
    xtop = np.cos(omega); ytop = np.sin(omega)
    xbot = np.cos(-omega); ybot = np.sin(-omega)
        
    for l in range(levels): # nuber of fork nodes
        for n in range(2**l): # quantify branch doubling 
            for i in range(int(N / (levels * 2 * (2**l)))): # uniform sample
                ## Top branch of current node
                top = np.random.uniform() # top branch sample
                X[0].append(root[0] + radius * top * xtop) # x
                X[1].append(root[1] + radius * top * ytop) # y
                    
                ## Bottom branch of current node
                bottom = np.random.uniform() # bottom branch sample
                X[0].append(root[0] + radius * bottom * xbot) # x
                X[1].append(root[1] + radius * bottom * ybot) # y
                C.extend([l,l])
                
            root[1] -= 2 * s # decrease y coordinate of root node
                
        root[0] += radius * xtop # increase x to end of current line
        root[1] = radius * ytop # move y to end of currrent line (reset y)
        radius =  radius / 2 # decrease radius
        s = np.sqrt(2) * radius # compute new branch length
        root[1] += n * 2 * s # set next y coordinate
                
    return np.asarray(X), np.asarray(C)

def noisy_tree(N, radius = 1, levels = 3):
    X, C = tree(N, radius, levels)
    X += np.random.normal(0, 0.05, X.shape)
    return X, C

    
    
def clusters(N, num_clusters = 3, sigma_min = .1, sigma_max = .3):
    '''Generate (3) Gaussian Clusters'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting
    
    for cluster in range(int(num_clusters)):
        cov = np.random.uniform(sigma_min, sigma_max) * np.diag(np.ones(2))
        mu = [[0.0, 0.5],[1.0, 1.0],[1.0, 0.0]]
           
        for _ in range(int(N / num_clusters)):
            sx, sy = np.random.multivariate_normal(mu[cluster], cov)
            X[0].append(sx); X[1].append(sy) # x; y
            C.append(cluster)
            
    return np.asarray(X), np.asarray(C)
    
def uniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
        
    theta = np.random.uniform(0, 2*np.pi, N)
    theta.sort()
        
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
            
    return np.asarray(X), np.asarray(C)

def hyperuniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
        
    theta = np.linspace(0, 2 * np.pi, N, endpoint = False)
        
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
            
    return np.asarray(X), np.asarray(C)
    
    
def hyperuniform_ellipse(N, a = 1, b = 2):
    '''Generate Hyperuniformly-Sampled 2-D Ellipse'''
    assert(a < b) # a must be length of minor semi-axis; b major semi-axis
        
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
        
    angles = 2*np.pi*np.arange(N)/N
        
    if a != b:
        '''Given N points, combine scipy elliptic integral + optimize to find 
           N equidistant points along ellilpse manifold, then convert to angles'''
        e = np.sqrt(1.0 - a**2 / b**2)
        tot_size = special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / N
        arcs = np.arange(N) * arc_size
        res = optimize.root(
                lambda x: (special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
            
        arcs = special.ellipeinc(angles, e)
    
    for t in angles:
        X[0].append(a * np.cos(t)) # x
        X[1].append(b * np.sin(t)) # y
            
    return np.asarray(X), np.asarray(C)
    
    
def two_spirals(n_points, noise = .2):
    '''Generate two nested spirals'''
    n = np.sqrt(np.random.rand(n_points,1)) * 220 * (2 * np.pi) / 360
    dx = -np.cos(n) * n + np.random.rand(n_points,1) * noise
    dy = np.sin(n) * n + np.random.rand(n_points,1) * noise
        
    X = np.vstack((np.hstack((dx,dy)),np.hstack((-dx,-dy)))).T # data 
    C = np.hstack((np.zeros(n_points),np.ones(n_points))) # colors for plot
        
    return X, C


def plot_datasets(N):
    '''Plot the 8 datasets'''

    barX, barC = barbell(N)
    treX, treC = tree(N)
    ntrX, ntrC = noisy_tree(N)
    cluX, cluC = clusters(N)
    uncX, uncC = uniform_circle(N)
    hucX, hucC = hyperuniform_circle(N)
    hueX, hueC = hyperuniform_ellipse(N)
    tspX, tspC = two_spirals(N)

    Xs = [barX, treX, ntrX, cluX, uncX, hucX, hueX, tspX]
    Cs = [barC, treC, ntrC, cluC, uncC, hucC, hueC, tspC]

    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    for i in range(2):
        for j in range(4):
            index = i * 4 + j
            ax[i][j].scatter(Xs[index][0], Xs[index][1], c = Cs[index], s = 5)
            ax[i][j].set_title(names[index])
            ax[i][j].set_aspect('equal')


def generate_data(dataset, N):
    dataset = dataset.lower()

    if dataset == 'barbell':
        X, C = barbell(N)
    elif dataset == 'tree':
        X, C = tree(N)
    elif dataset == 'noisy tree':
        X, C = noisy_tree(N)
    elif dataset == 'clusters':
        X, C = clusters(N)
    elif dataset == 'uniform circle':
        X, C = uniform_circle(N)
    elif dataset == 'hyperuniform circle':
        X, C = hyperuniform_circle(N)
    elif dataset == 'hyperuniform ellipse':
        X, C = hyperuniform_ellipse(N)
    elif dataset == 'two spirals':
        X, C = two_spirals(N)
    else:
        print('That is not a valid dataset name. Please enter one of the 8 dataset names.')
        return None, None

    return X, C


def condense(X, eps = None, num_steps = math.inf):
    N = np.shape(X)[0]
    Q_p = np.eye(N)
    Q_diff = math.inf
    Xs = [X.copy()]

    if eps is None:
        eps = np.pi / N
    
    i = 0
    i_prev = -2
    steps = 0
    while i - i_prev > 1:
        i_prev = i
        while Q_diff >= 1e-4:
            i += 1
            
            #Construct the kernel K
            A = spatial.distance.squareform(spatial.distance.pdist(X, metric = 'euclidean'))
            A = np.exp(-(A ** 2) / eps)
            Q = np.diag(np.sum(A, axis = 0))
            Q_inv = np.diag(1./np.sum(A, axis = 0))
            K = Q_inv @ A @ Q_inv
            
            #Construct the diffusion operator P
            D_inv = np.diag(1./np.sum(K, axis = 0))
            P = D_inv @ K
            X_1 = P @ X
            
            #Update matrices
            Q_diff = np.max(np.abs(np.diag(Q) - np.diag(Q_p)))
            Q_p = Q.copy()
            
            #Save data
            X = X_1.copy()
            Xs.append(X)

            steps += 1
            if steps == num_steps:
                return Xs
        eps *= 2
        Q_diff = math.inf
    return Xs


def plot_diffusion(Xs, C, dataset):
    '''Plot the diffusion process over time.'''
    cols = 5
    rows = np.shape(Xs)[0] // cols
    if np.shape(Xs)[0] % cols != 0:
        rows += 1

    if dataset == 'barbell':
        h = 2
    elif dataset in {'clusters', 'uniform circle', 'hyperuniform circle', 'two spirals'}:
        h = 3
    else:
        h = 3.25

    fig, ax = plt.subplots(rows, cols, figsize = (20, rows * h))
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < np.shape(Xs)[0]:
                ax[i][j].scatter(Xs[index].T[0], Xs[index].T[1], c = C, s = 5)
                ax[i][j].set_title('t = {}'.format(index))
                if i == 0 and j == 0:
                    plt_xlim = ax[0][0].get_xlim()
                    plt_ylim = ax[0][0].get_ylim()
            ax[i][j].tick_params(axis='both', which='both', bottom=False, top=False,
                                 labelbottom=False, right=False, left=False,
                                 labelleft=False)
            ax[i][j].set_xlim(plt_xlim)
            ax[i][j].set_ylim(plt_ylim)
            ax[i][j].set_aspect('equal')


def make_plots(dataset, N, num_steps = 50):
    X, C = generate_data(dataset, N)
    if X is None:
        return
    Xs = condense(X.T, num_steps = num_steps)
    plot_diffusion(Xs, C, dataset)


# Graph Kernel

def adj_matrix(g):
    n = len(g[0])
    A = np.zeros((n, n))
    
    for e, wl in g[1].items():
        A[e[0], e[1]] = wl[0]
    
    A += A.T
    return A


def edge_matrix(g):
    n = len(g[0])
    E = np.zeros((n, n))
    
    for e, wl in g[1].items():
        E[e[0], e[1]] = wl[1]
        
    E += E.T
    return E


# Hyperparameters
q = 0.01
s = 1
lam = 1


def indicator_kernel(v_1, v_2):
    '''Returns 1 if they are the same vertex label
    and 0.5 otherwise.'''
    return v_1 == v_2 and 1 or 0.5


def gaussian_kernel(e_1, e_2):
    '''Returns the Gaussian kernel on the edges.'''
    return np.exp(-(e_1 - e_2) ** 2 / (2 * lam ** 2))


def tree_kernel(g1, g2, k_v = indicator_kernel, k_e = gaussian_kernel):
    '''Returns the value of Vxroo'''
    n1 = len(g1[0])
    n2 = len(g2[0])
    
    A1 = adj_matrix(g1)
    A2 = adj_matrix(g2)
    
    q1 = np.array(n1 * [q])
    q2 = np.array(n2 * [q])
    
    d1 = np.sum(A1, axis = 0) / (1 - q1)
    d2 = np.sum(A2, axis = 0) / (1 - q2)
    
    E1 = edge_matrix(g1)
    E2 = edge_matrix(g2)
    
    v1 = g1[0]
    v2 = g2[0]
    
    Ax = np.kron(A1, A2)
    qx = np.kron(q1, q2)
    Dx = np.kron(d1, d2)
    Ex = np.array([k_e(j, E2) for i in E1 for j in i]).reshape(n1, -1, n2, n2).swapaxes(1, 2).reshape(n1 * n2, n1 * n2)
    Vx = np.array([k_v(i, j) for i in v1 for j in v2])
    
    t1 = np.diag(Dx * 1./Vx)
    t2 = np.multiply(Ax, Ex)
    t3 = Dx * qx

    return np.linalg.inv(t1 - t2) @ t3


'''Return the marginalized graph kernel between
   g1 and g2 using the provided vector and edge
   kernels.'''
def mgk(g1, g2, k_v = indicator_kernel, k_e = gaussian_kernel):
    n1 = len(g1[0])
    n2 = len(g2[0])
    
    p1 = np.array(n1 * [s])
    p2 = np.array(n2 * [s])
    
    px = np.kron(p1, p2)
    
    Vxroo = tree_kernel(g1, g2)
    return px.T @ Vxroo


def rand_graph(n):
    v = np.random.choice(['K', 'R'], size = n)
    e_m = np.random.choice(range(10), size = (n, n), p = [0.5] + 9 * [0.5 / 9])
    e_m = e_m + e_m.T
    e_m = e_m - np.diag(np.diag(e_m))

    e = dict()
    for row in range(n):
        for col in range(row + 1, n):
            e[(row, col)] = (e_m[row][col], 1)
    
    return (v, e)


v0 = ['R', 'K']
e0 = {(0, 1): (1, 1)}
g0 = (v0, e0)


v1 = ['K', 'R', 'K', 'R']
e1 = {(0, 1): (1, 1), (1, 2): (1, 1), (1, 3): (1, 1)}
g1 = (v1, e1)


g2 = rand_graph(10)


def plot_adj():
    fig, ax = plt.subplots(1, 3, figsize = (20, 5))
    img0 = ax[0].imshow(adj_matrix(g0))
    img1 = ax[1].imshow(adj_matrix(g1))
    img2 = ax[2].imshow(adj_matrix(g2))
    ax[0].set_title(r'Adjacency Matrix of $G_0$')
    ax[1].set_title(r'Adjacency Matrix of $G_1$')
    ax[2].set_title(r'Adjacency Matrix of $G_2$')
    fig.colorbar(img0, ax = ax[0])
    fig.colorbar(img1, ax = ax[1])
    fig.colorbar(img2, ax = ax[2])

def plot_z():
    fig, ax = plt.subplots(1, 3, figsize = (20, 5))
    n0 = len(g0[0])
    n1 = len(g1[0])
    n2 = len(g2[0])
    z0 = tree_kernel(g0, g0).reshape(n0, n0)
    z1 = tree_kernel(g1, g1).reshape(n1, n1)
    z2 = tree_kernel(g2, g2).reshape(n2,  n2)
    img0 = ax[0].imshow(z0)
    img1 = ax[1].imshow(z1)
    img2 = ax[2].imshow(z2)
    ax[0].set_title(r'$V_{\times}r_{\infty}$ of $G_0$')
    ax[1].set_title(r'$V_{\times}r_{\infty}$ of $G_1$')
    ax[2].set_title(r'$V_{\times}r_{\infty}$ of $G_2$')
    fig.colorbar(img0, ax = ax[0])
    fig.colorbar(img1, ax = ax[1])
    fig.colorbar(img2, ax = ax[2])