import time
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from scipy.spatial import cKDTree as KDTree
import os
import random
import numpy as np
import pandas as pd 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import OPTICS
from glob import glob 
from tqdm import tqdm
from scipy.spatial.distance import pdist , squareform
np.random.seed(5)


#Fix Directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

path = os.getcwd()
PATH_storage = path
print(PATH_storage)

patient_frame_to_store = 'Patient_generated'
if not os.path.exists(patient_frame_to_store):
    os.makedirs(patient_frame_to_store)
path_to_store_patient_frame = os.path.join(PATH_storage, patient_frame_to_store)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

path = os.getcwd()
PATH_storage = path
print(PATH_storage)

            #Chose to work on continuous data
#figures_storage, frame_storage = 'Figures_LDA_on_continuous_data_generated' , 'Data_frame_LDA_on_continuous_data_generated'
figures_storage ,frame_storage = 'Figures_LDA_on_binary_data_generated' , 'Data_frame_LDA_on_binary_data_generated'

if not os.path.exists(figures_storage):
    os.makedirs(figures_storage)
path_to_store_figures = os.path.join(PATH_storage, figures_storage)

if not os.path.exists(frame_storage):
    os.makedirs(frame_storage)
path_to_store_frame = os.path.join(PATH_storage, frame_storage)


#Generate patients
number_of_patients = 50
number_of_cell_phenotypes = 7
number_of_feature_to_generate = 1
#Patient_phenotype
patient_phenotypes = 2
total_number_of_cell = 10000

n_sample = 5000 #Number of cell subset randomly chosen
strating_point = 0
number_of_cluster = 2 #Number_of_cluster_for_k_means
max_val = 0.01
#std for phenotype 1 and 2 updated at every run
cluster_std_1 = 0.1
cluster_std_2= 0.2

        #Define parameters for LDA
diviser_of_matrix = 100
number_of_calculation = 50
number_of_topic_to_test = 10
        #Parameters for cutting dendrogram
cutting_tree = 1


#left_part = random.randint(0,total_number_of_cell+1)
#right_part = total_number_of_cell - left_part

left_part, right_part = 1000,9000
cell_distrib = [left_part,right_part]
cell_distrib_2 = [right_part,left_part]

left_part_2, right_part_2 = 1000,9000
cell_2_distrib = [left_part_2, right_part_2]
cell_2_distrib_2 = [right_part_2, left_part_2]

phenotype_code = []

for patient in tqdm(range(number_of_patients)):
    random_state = 41
    phenotype_continuous_data = []
    phenotype_binary_data = []
    if patient < number_of_patients/patient_phenotypes:
        phenotype_code.append(1)
        X, y = make_blobs(n_samples=np.array(cell_distrib), centers=None, n_features= number_of_feature_to_generate, cluster_std=cluster_std_1, random_state=random_state)
        phenotype_continuous_data.append(X)
        phenotype_binary_data.append(y)
        X, y = make_blobs(n_samples=np.array(cell_2_distrib), centers=None, n_features= number_of_feature_to_generate, cluster_std=cluster_std_1, random_state=random_state)
        phenotype_continuous_data.append(X)
        phenotype_binary_data.append(y)
        random_state =+ 10
        cluster_std_1 =+ random.uniform(0, 0.5)
    else:
        phenotype_code.append(2)
        X, y = make_blobs(n_samples=np.array(cell_distrib_2), centers=None, n_features= number_of_feature_to_generate, cluster_std=cluster_std_2, random_state=random_state)
        phenotype_continuous_data.append(X)
        phenotype_binary_data.append(y)
        X, y = make_blobs(n_samples=np.array(cell_2_distrib_2), centers=None, n_features= number_of_feature_to_generate, cluster_std=cluster_std_1, random_state=random_state)
        phenotype_continuous_data.append(X)
        phenotype_binary_data.append(y)
        random_state =+ 10
        cluster_std_1 =+ random.uniform(0, 0.5)


    phenotype_continuous_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_continuous_data))).T
    phenotype_binary_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_binary_data))).T


    phenotype_continuous_data_patient.to_csv(path_to_store_patient_frame +f'/Generated_file_for_patient_N°{str(patient).zfill(3)}.csv')
    phenotype_binary_data_patient.to_csv(path_to_store_patient_frame + f'/Generated_binary_file_for_patient_N°{str(patient).zfill(3)}.csv')


dirs = [path_to_store_patient_frame]
for dir in dirs:
    #files = glob(f'{dir}/Generated_file_*.csv')
    files = glob(f'{dir}/Generated_binary_*.csv')

    for file in files:
        data = pd.read_csv(file)
        name = '_'.join(file.split("_")[-2:]).replace('.csv','')
        print(file)

        if len(pd.read_csv(file)) > n_sample:
            if strating_point == 0 :
                row_data = pd.read_csv(file, index_col=0)
                               
                #data =  np.log10(np.maximum(row_data.sample(n=n_sample, random_state=42),max_val))
                data = row_data.sample(n=n_sample, random_state=42)
                subset_data = stats.zscore(data)
                print(subset_data.shape)
                #time_start = time.time()
                #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                #tsne_results = tsne.fit_transform(subset_data)
                #print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
                        
                #Calculate Kmeans Clustering data previsouly reduced with fixed number of cluster
                time_start = time.time()
                kmeans = KMeans(n_clusters=number_of_cluster, random_state=41).fit(subset_data)
                print('Kmeans done: Time elapsed: {} seconds'.format(time.time()-time_start))
                labels = kmeans.labels_
                centroids_ref  = kmeans.cluster_centers_

                counting_occurence_in_cluster_ref = Counter(labels)
                reference_dataframe = pd.DataFrame.from_dict(counting_occurence_in_cluster_ref, orient='index').reset_index()
                reference_dataframe = reference_dataframe.rename(columns={'index':'Cluster', 0:f'Count_{name}'})

                vals_reference = np.fromiter(counting_occurence_in_cluster_ref.values(), dtype=float)

                strating_point += 1
                    
            else: 
                
                row_data_to_compare = pd.read_csv(file, index_col=0)
                data_to_compare = row_data_to_compare.sample(n=n_sample, random_state=42)
                #Apply Tsne with 2 components on datas
                subset_data_unref = stats.zscore(data_to_compare)
                print(subset_data_unref.shape)
                #time_start = time.time()
                #tsne = TSNE(n_components=n_comp, verbose=1, perplexity=40, n_iter=300)
                #tsne_results_unref = tsne.fit_transform(subset_data_unref)
                #print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

                #Calculate Kmeans Clustering data previsouly reduced with fixed number of cluster
                time_start = time.time()
                kmeans_unref = KMeans(n_clusters=number_of_cluster, random_state=41).fit(subset_data_unref)
                print('Kmeans done: Time elapsed: {} seconds'.format(time.time()-time_start))
                labels_unref = kmeans_unref.labels_
                centroids_unref = kmeans_unref.cluster_centers_

                counting_occurence_in_patient_compare = Counter(labels_unref) 

                vals_unref = np.fromiter(counting_occurence_in_patient_compare.values(), dtype=float)
                
                #COMPARING USING KDTREE
                k = KDTree(centroids_unref)
                (dists, idxs) = k.query(centroids_ref)

                vals_unref[idxs] 

                reference_dataframe[f'Count_{name}'] = vals_unref[idxs] 

                print(reference_dataframe.shape, reference_dataframe.columns)

reference_dataframe.sort_values(by=['Cluster'], inplace=True)
reference_dataframe.sort_index(axis = 1, ascending = True, inplace=True)

reference_dataframe.to_csv(path_to_store_frame + f'/Data_for_LDA_from_generate_data_with_n_{number_of_cluster}.csv')



for runrun in range(2, number_of_topic_to_test + 1):
#for runrun in range(17, 18):
    excluded_patient = 0
    vocabulary  = list(range(number_of_cluster))
    raw_data_T4 = pd.read_csv(path_to_store_frame + f'/Data_for_LDA_from_generate_data_with_n_{number_of_cluster}.csv' ,index_col= 'Cluster')
    raw_data_T4.drop(raw_data_T4.columns[0], axis=1, inplace=True)  
    data_T4 = raw_data_T4.T
    t4_ = data_T4.to_numpy()/diviser_of_matrix

    #t4_ = np.around(t4_)
    docs = []
    npatients, nvocabulary = t4_.shape
    for n in range (npatients):
        current_doc = []
        doc = t4_[n,:]
        for i in range(nvocabulary):
            for _ in range(int(doc[i])):
                current_doc.append(i)
        docs.append(current_doc)
                
            

    D = len(docs)        # number of documents
    V = len(vocabulary)  # size of the vocabulary 
    T = runrun            # number of topics

    alpha = 1 / T         # the parameter of the Dirichlet prior on the per-document topic distributions
    beta = 1 / T        # the parameter of the Dirichlet prior on the per-topic word distribution


    z_d_n = [[0 for _ in range(len(d))] for d in docs]  # z_i_j
    theta_d_z = np.zeros((D, T))
    phi_z_w = np.zeros((T, V))
    n_d = np.zeros((D))
    n_z = np.zeros((T))

    ## Initialize the parameters
    # m: doc id
    for d, doc in enumerate(docs):  
        # n: id of word inside document, w: id of the word globally
        for n, w in enumerate(doc):
            # assign a topic randomly to words
            z_d_n[d][n] = int(np.random.randint(T))
            # get the topic for word n in document m
            z = z_d_n[d][n]
            # keep track of our counts
            theta_d_z[d][z] += 1
            phi_z_w[z, w] += 1
            n_z[z] += 1
            n_d[d] += 1

    for iteration in tqdm(range(number_of_calculation)):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # get the topic for word n in document m
                z = z_d_n[d][n]

                # decrement counts for word w with associated topic z
                theta_d_z[d][z] -= 1
                phi_z_w[z, w] -= 1
                n_z[z] -= 1

                # sample new topic from a multinomial according to our formular
                p_d_t = (theta_d_z[d] + alpha) / (n_d[d] - 1 + T * alpha)
                p_t_w = (phi_z_w[:, w] + beta) / (n_z + V * beta)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                #new_z = np.random.multinomial(1, p_z).argmax()
                new_z = np.random.choice(len(p_z), 1, p=p_z)[0] 

                # set z as the new topic and increment counts
                z_d_n[d][n] = new_z
                theta_d_z[d][new_z] += 1
                phi_z_w[new_z, w] += 1
                n_z[new_z] += 1


    norm_theta = theta_d_z.copy()
    ns = np.sum(theta_d_z, axis=1)
    for i in range(ns.shape[0]):
        norm_theta[i, :] /= ns[i]
    print(np.max(norm_theta))
    corr_matrix = np.dot(norm_theta, norm_theta.T)
    print(corr_matrix.shape)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(corr_matrix, index=raw_data_T4.columns, columns=raw_data_T4.columns).corr())
    plt.title(f'Correlation matrix for topic N°%s, and N {number_of_cluster} for clustering' %runrun)
    #plt.savefig(path_to_store_figures + f'/corr_matrix_on_continuous_data_generated_topic_%s_N_{number_of_cluster}_for_kmeans.png'%runrun)
    plt.savefig(path_to_store_figures + f'/corr_matrix_on_binary_data_generated_topic_%s_N_{number_of_cluster}_for_kmeans.png'%runrun)
    (pd.DataFrame(norm_theta)).to_csv(path_to_store_frame + f'/LDA_DATAFRAME_Topic_{runrun}.csv')
    plt.close()


    linked = linkage(norm_theta, 'complete')
    labelList = raw_data_T4.columns.to_list()

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.savefig(path_to_store_figures + f'/Cluster_dendrogram_for_topic_%s.png'%runrun)
    plt.close()

    hierarchical_result = fcluster(linked, cutting_tree*norm_theta.max(),'distance')
    frame_binary_with_threshold = pd.DataFrame(norm_theta,index=data_T4.index,columns= [f'Topic {i}' for i in range(runrun)])

    frame_binary_with_threshold['Statut_number'] =  frame_binary_with_threshold.index
    frame_binary_with_threshold.reset_index(drop=True, inplace=True)
    frame_binary_with_threshold['Hierarchical_clustering'] = hierarchical_result
    frame_binary_with_threshold['Phenotype'] = phenotype_code
    frame_binary_with_threshold.groupby(['Phenotype','Hierarchical_clustering']).size().unstack(fill_value=0).to_csv(path_to_store_frame + f'/Confusion_matrix_Topic_{runrun}_with_n_{number_of_cluster}.csv')

    sorted_frame_cell_info_threshold_and_analysis = frame_binary_with_threshold.sort_values(by=['Hierarchical_clustering'])

    result_to_analyse = pdist(sorted_frame_cell_info_threshold_and_analysis[[col for col in frame_binary_with_threshold if col.startswith('Topic')]], 'euclidean')
    squareform(result_to_analyse)
    distance_topics = pd.DataFrame(squareform(result_to_analyse), index = sorted_frame_cell_info_threshold_and_analysis['Statut_number'], columns=sorted_frame_cell_info_threshold_and_analysis['Statut_number'] )
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_topics)
    plt.title(f'Euclidean distance matrix sorted by Hierarchical Clustering for topic N°%s' %runrun)
    plt.savefig(path_to_store_figures + f'/Euclidian_distance_matrix_sorted_by_Hierarchical_Clustering_for_topic_%s_and_N_{number_of_cluster}.png'%runrun)
    plt.close()

