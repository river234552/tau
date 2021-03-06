from numba import jit, njit,prange
from numba import cuda, int32, complex128, float64, int64
import numpy as np
import threading
import math
import random
import torch
import weibull
import itertools
from scipy.spatial import distance as compute_distance
from scipy.spatial.distance import squareform
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

#####################################################################################
#Customize CUDA Kernels
@cuda.jit(device = True)
def cosine_gpu(u, v):
    dot_product = 0
    norm_v = 0
    norm_u = 0
    for m, n in zip(u, v):
        dot_product += m * n
    for m, n in zip(u, u):
        norm_u += m * n
    for m, n in zip(v, v):
        norm_v += m * n
    return 1.0 - dot_product / (math.sqrt(norm_u) * math.sqrt(norm_v))

@cuda.jit(device = True)
def euclidean_gpu(u, v):
    norm = 0
    for m, n in zip(u, v):
        norm += (m - n) * (m - n)
    norm = math.sqrt(norm)
    return norm

@cuda.jit
def cosine_dis_gpu(X, Y, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        u = X[i]
        v = Y[j]
        out[i, j] = cosine_gpu(u, v)

@cuda.jit
def euclidean_dis_gpu(X, Y, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        u = X[i]
        v = Y[j]
        out[i, j] = euclidean_gpu(u, v)
#####################################################################################

def tau(args, features, gpus):
    #Now only support Cosine and Euclidean on GPU
    if args.distance_metric:
        metrics = [args.distance_metric]
    else:
        metrics =['cosine','euclidean']
    print("The Distance Metric is: ", metrics)

    #CUDA parallel distance computing, support multi-gpus
    def gpu_pairwise_distance(chunks, step_i, gpu):
        #lock.acquire()#no need to lock threads in this case
        cuda.select_device(gpu)
        for i, chunk1 in enumerate(chunks):
            print("Computing distance chunk: ", i + 1)
            #Define chunk id x axis
            step_j = 0
            n_i = chunk1.shape[0]
            for j, chunk2 in enumerate(chunks):
                #Copy data to gpu
                X_global = cuda.to_device(chunk1)
                Y_global = cuda.to_device(chunk2)
                #Define chunk id y axis
                n_j = chunk2.shape[0]
                out_global = cuda.device_array((chunk1.shape[0], chunk2.shape[0]))
                # Define gpu's grid
                threadsperblock = (16, 16)
                blockspergrid_x = int(math.ceil(out_global.shape[0] / threadsperblock[0]))
                blockspergrid_y = int(math.ceil(out_global.shape[1] / threadsperblock[1]))
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                #Compute distance on gpu
                if metric == "cosine":
                    cosine_dis_gpu[blockspergrid, threadsperblock](X_global, Y_global, out_global)
                elif metric == "euclidean":
                    euclidean_dis_gpu[blockspergrid, threadsperblock](X_global, Y_global, out_global)
                #Find mean and max for each loop
                mean_distances.append(np.mean(out_global.copy_to_host()))
                max_dis.append(np.max(out_global.copy_to_host()))
                #Select 2% points to EVT randomly
                k = int(len(out_global.copy_to_host()) * 0.02)
                number_of_rows = out_global.copy_to_host().shape[0]
                random_indices = np.random.choice(number_of_rows, size=k, replace=False)
                #Copy gpu distance data to cpu numpy
                if len(out_global.copy_to_host()[random_indices, :]) > 0:
                    whole_distances.extend(out_global.copy_to_host()[random_indices, :].flatten())
                #update chunk id
                step_j += n_j
                step_i += n_i
                del X_global, Y_global, out_global
    
    for metric in metrics:
        distances = []
        mean_distances = []
        max_dis = []
        #Split cpu's data to gpus
        n = int(len(features) / len(gpus))
        mutilple_features = [features[i * n:(i + 1) * n] for i in range((len(features) + n - 1) // n )]
        threads = []
        from split import split_double
        #Compute chunks in multi-gpus
        for p, gpu in enumerate(gpus):
            whole_distances = []
            split = split_double(args, mutilple_features[p])
            n = int(len(mutilple_features[p]) / split)
            chunks = [mutilple_features[p][i * n:(i + 1) * n] for i in range((len(mutilple_features[p]) + n - 1) // n )]
            step_i = 0
            threads.append(threading.Thread(target=gpu_pairwise_distance, args=[chunks, step_i, int(gpu),]))
        #Number of threads depend on how many gpus you have
        for t in threads:
            t.setDaemon(True)
            t.start()
        #Re-group final distance data from gpus
        for t in threads:
            whole_distances = []
            t.join()
            distances.extend(np.array(whole_distances).flatten())

        #Process data
        random_distances = np.array(distances).flatten()
        random_distances = random_distances.reshape((random_distances.shape[0], 1)).T
        mean_distances = np.mean(mean_distances)
        print("mean_distances: ",mean_distances)
        print("max dis:", max(max_dis))#original max dis before EVT
        ###################################################################

        ########################################################################################
        print("Finding Nearest Points......")
        #Find nearest points on GPUs
        from gpu_functions import gpu_nearest
        nearest_cluster = np.zeros((len(features)), dtype = 'int')
        nearest_points_dis = np.zeros((len(features)))
        n = int(len(features) / len(gpus))
        features = list(features)
        mutilple_features = [features[i * n:(i + 1) * n] for i in range((len(features) + n - 1) // n )]
        if len(gpus) > 1:
            if len(mutilple_features) > len(gpus):
                mutilple_features[len(gpus) - 1].extend(mutilple_features[len(gpus)])
                del mutilple_features[len(gpus)]
        ind = []
        step = 0
        steps = []
        for i, j in enumerate(mutilple_features[0:len(gpus)]):
            ind.append(range(step, len(j)+step))
            steps.append(step)
            step += len(j)
        threads = []
        for p, gpu in enumerate(gpus):
            threads.append(threading.Thread(target=gpu_nearest, args=[mutilple_features[p], features, int(gpu), ind[p], steps[p], metric, nearest_cluster, nearest_points_dis]))
        thread(threads)
        del mutilple_features
        # In round 1 the centroids is the points no matter what's linkage
        nearest_cluster_with_distance_round_1 = [[j, [k, i]] for k, (i, j) in enumerate(zip(nearest_cluster, nearest_points_dis))]
        nearest_cluster_with_distance_round_1 = sorted(nearest_cluster_with_distance_round_1)  # Sort by distance, process the smallest one first
        nearest_points = nearest_cluster
        ########################################################################################

        print("Computing the appearance of nearest_points")
        threadsperblock = 32
        blockspergrid = math.ceil(nearest_points.shape[0] / threadsperblock)
        X_global = cuda.to_device(nearest_points)
        out_global = cuda.device_array((nearest_points.shape[0]))
        from cuda_kernels import count_appear
        count_appear[blockspergrid, threadsperblock](X_global, out_global)
        appear = np.array(out_global.copy_to_host(), dtype = int)
        appear_count = [[j, i] for i, j in enumerate(appear)]
        # count the appearance of each kernel points
        # generate order
        order = [i[1] for i in sorted(appear_count, reverse=True)]
        # add non kernel points to order
        processed = set()
        init = []
        for count, i in enumerate(order):
            j = nearest_points[i]
            if i not in processed and j not in processed:
                init.append([i, j])
                processed.add(i)
                processed.add(j)
        init = init[0: int(len(init))]
        N = len(init)
        init_length = N
        init_features = [[features[i[0]], features[i[1]]] for i in init] #features of initial groups.
        ######################################################################################################
        print("Finding Nearest Intial Pairs")
        #Computing nearest centroids on GPUs
        centroids = [np.mean(i,axis=0) for i in init_features]
        X = centroids
        from gpu_functions import gpu_nearest_init_centroids
        gs = np.zeros((len(init_features)))
        nearest_init = np.zeros((len(init_features)), dtype = 'int')
        n = int(len(centroids) / len(gpus))
        mutilple_centroids = [centroids[i * n:(i + 1) * n] for i in range((len(centroids) + n - 1) // n )]
        if len(gpus) > 1:
            if len(mutilple_centroids) > len(gpus):
                mutilple_centroids[len(gpus) - 1].extend(mutilple_centroids[len(gpus)])
                del mutilple_centroids[len(gpus)]
        ind = []
        step = 0
        steps = []
        for i, j in enumerate(mutilple_centroids[0:len(gpus)]):
            ind.append(range(step, len(j) + step))
            steps.append(step)
            step += len(j)
        threads = []
        for p, gpu in enumerate(gpus):
            threads.append(threading.Thread(target=gpu_nearest_init_centroids, args=[mutilple_centroids[p], X, int(gpu), ind[p], metric, gs, nearest_init]))
        thread(threads)
        del mutilple_centroids
        ##########################################################################################################
        #Nearest initial pairs combo
        nearest_init_combo = [[m, init[n]] for m, n in zip(init, nearest_init)]
        ##########################################################################################################
        gxs = []
        print("Computing Gaps")
        # Computing gaps on GPUs
        from gpu_functions import gpu_distance
        for pair1, pair2 in nearest_init_combo:
            round_features = np.array([features[k] for k in [pair1[0], pair1[1], pair2[0], pair2[1]]])
            features0 = [features[k] for k in pair1] #extract features of cluster0
            features1 = [features[k] for k in pair2] #extract features of cluster1
            centroid0 = np.mean(features0, axis=0) # Get controid of initial pair0
            centroid1 = np.mean(features1, axis=0) # Get controid of initial pair1
            if metric == "cosine":
                gx = scipy.spatial.distance.cosine(centroid0, centroid1)
            elif metric == "euclidean":
                gx = scipy.spatial.distance.euclidean(centroid0, centroid1)
            gxs.append(gx) #gaps

    #Our tau
    tau = get_tau(torch.Tensor(gxs), 1, tailfrac=.97, pcent=.999)



    return 0, T, tau, nearest_points, metric, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis, gx, 0

def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)

def get_tau(data,maxval,tailfrac=.25,pcent=.999):
    #tw =  weibull.weibull(translateAmountTensor=.001)
    tw = weibull.weibull()
    nbin=200
    nscale = 10
    #fullrange = torch.linspace(0,torch.max(ijbbdata),nbin)
    fullrange = torch.linspace(0,maxval,nbin)
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    #print( name , "Data mean, max", torch.mean(ijbbdata),torch.max(ijbbdata))
    imean = torch.mean(data)
    istd = torch.std(data)
    imax = torch.max(data)
    tw.FitHighTrimmed(data.view(1,-1),int(tailfrac*len(data)))
    parms = tw.return_all_parameters()
    wscoresj = tw.wscore(fullrange)
    probj = nan_to_num(tw.prob(fullrange))
    if(torch.sum(probj) > .001):
        probj = probj/torch.sum(probj)
    tau=  parms['Scale']*np.power(-np.log((1-pcent)),(1/parms['Shape'])) - parms['translateAmountTensor'] + parms['smallScoreTensor']
    return tau.numpy()


def thread(threads):
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()

def takeSecond(elem):
    return elem[1]
