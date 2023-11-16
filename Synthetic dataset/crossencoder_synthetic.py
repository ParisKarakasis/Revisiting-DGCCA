import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from crossenc_utils import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import random
import copy
import sys
import os
from scipy.io import savemat

a = sys.argv[1:]
ID = str(a[0])  # ID OF THE EXPERIMENT.
o_dim = int(a[1])  # Dimension of the latent space

lr = 0.1 * 1e-1
epoch_num = 20 40
batch_size = 100
reg_par = 0.001
dropout = 0
seed = 35
num_init = 10
SPR = -18 # in dB

i_shape = [64] * 2
m = [3000, 1500, 1500]
d = o_dim
d_dp = [4] * 2

folder_path = str(ID)
path = os.path.join(os.getcwd(), folder_path)
os.mkdir(folder_path)
device = torch.device('cuda')
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

rng = np.random.default_rng(seed=seed)

#########################
#  Run the experiments  #
#########################
# Load the dataset
dataset = 0
print("Synthetic")
X1, X2, XV1, XV2, XTe1, XTe2, Z, Z_val, Z_test = generate_synthetic_dataset2(m, d_dp, d, i_shape, rng, SPR)
print(i_shape)

X1, X2, XV1, XV2, XTe1, XTe2, data_tr, data_val, data_te, X1_mean, X1_std, X2_mean, X2_std = my_standardize3(X1, X2, XV1, XV2, XTe1, XTe2, batch_size)

# Copy the data
data_tr_cp = copy.deepcopy(data_tr)
data_val_cp = copy.deepcopy(data_val)
data_te_cp = copy.deepcopy(data_te)

# Run DGCCA
t0_DGCCA = time.perf_counter()
DGCCA_best_model, DGCCA_best_loss, DGCCA_all_models = train_DGCCA(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te, dataset)
t_DGCCA = time.perf_counter()-t0_DGCCA
torch.save({'model_state_dict': DGCCA_best_model, 'loss': DGCCA_best_loss, 'models_state_dict': DGCCA_all_models, 'time':t_DGCCA}, path + '/DGCCAModel')
#t = torch.load(folder_path+'/DGCCAModel', map_location=torch.device('cuda'))
#DGCCA_best_model = t['model_state_dict']
#DGCCA_best_loss = t['loss']
#DGCCA_all_models = t['models_state_dict']
#t_DGCCA = t['time']

# Run DCCAE
data_tr = copy.deepcopy(data_tr_cp)
data_val = copy.deepcopy(data_val_cp)
data_te = copy.deepcopy(data_te_cp)
t0_DCCAE = time.perf_counter()
DCCAE_best_models, DCCAE_best_losses, DCCAE_all_models = train_DCCAE(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te, dataset)
t_DCCAE = time.perf_counter()-t0_DCCAE
torch.save({'model_state_dict': DCCAE_best_models, 'loss': DCCAE_best_losses, 'models_state_dict': DCCAE_all_models, 'time':t_DCCAE}, path + '/DCCAEModels')
#t = torch.load(folder_path+'/DCCAEModels', map_location=torch.device('cuda'))
#DCCAE_best_models = t['model_state_dict']
#DCCAE_best_losses = t['loss']
#DCCAE_all_models = t['models_state_dict']
#t_DCCAE = t['time']

# Run XE (proposed method)
data_tr = copy.deepcopy(data_tr_cp)
data_val = copy.deepcopy(data_val_cp)
data_te = copy.deepcopy(data_te_cp)
t0_XE = time.perf_counter()
XE_best_models, XE_best_losses, XE_all_models = train_XE(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te, dataset)
t_XE = time.perf_counter()-t0_XE
torch.save({'model_state_dict': XE_best_models, 'loss': XE_best_losses, 'models_state_dict': XE_all_models, 'time':t_XE}, path + '/XEModels')
#t = torch.load(folder_path+'/XEModels', map_location=torch.device('cuda'))
#XE_best_models = t['model_state_dict']
#XE_best_losses = t['loss']
#XE_all_models = t['models_state_dict']
#t_XE = t['time']

# Run Lyu et al. method
data_tr = copy.deepcopy(data_tr_cp)
data_val = copy.deepcopy(data_val_cp)
data_te = copy.deepcopy(data_te_cp)

if True:
    args = EmptyObject()
    args.o_dim = o_dim
    args.o_dim2 = [4] * 2 #d_dp
    args.num_iters = epoch_num
    args.batchsize1 = batch_size
    args.batchsize2 = 300
    args.lr_max = 1e-2
    args.lr_min = lr
    args.weight_decay = reg_par
    args.phi_num_layers = 3
    args.phi_hidden_size = 32
    args.tau_num_layers = 3
    args.tau_hidden_size = 32
    args.inner_epochs = 1
    args._lambda = 0*1e-3

t0_Lyu0 = time.perf_counter()
Lyu_best_models_0, Lyu_best_losses_0, Lyu_all_models_0 = train_Lyu_et_all(i_shape, args, num_init, data_tr, data_val, data_te)
t_Lyu0 = time.perf_counter() - t0_Lyu0
torch.save({'model_state_dict': Lyu_best_models_0, 'loss': Lyu_best_losses_0, 'models_state_dict': Lyu_all_models_0, 'time':t_Lyu0}, path + '/LyuModels_0')
#t = torch.load(folder_path+'/LyuModels_0', map_location=torch.device('cuda'))
#Lyu_best_models_0 = t['model_state_dict']
#Lyu_best_losses_0 = t['loss']
#Lyu_all_models_0 = t['models_state_dict']
#t_Lyu0 = t['time']

# Run Lyu et al. method (setting one)
data_tr = copy.deepcopy(data_tr_cp)
data_val = copy.deepcopy(data_val_cp)
data_te = copy.deepcopy(data_te_cp)

if True:
    args.num_iters = epoch_num
    args._lambda = 1*1e-3

t0_Lyu1 = time.perf_counter()
Lyu_best_models_1, Lyu_best_losses_1, Lyu_all_models_1 = train_Lyu_et_all(i_shape, args, num_init, data_tr, data_val, data_te)
t_Lyu1 = time.perf_counter() - t0_Lyu1
torch.save({'model_state_dict': Lyu_best_models_1, 'loss': Lyu_best_losses_1, 'models_state_dict': Lyu_all_models_1, 'time':t_Lyu1}, path + '/LyuModels_1')
#t = torch.load(folder_path+'/LyuModels_1', map_location=torch.device('cuda'))
#Lyu_best_models_1 = t['model_state_dict']
#Lyu_best_losses_1 = t['loss']
#Lyu_all_models_1 = t['models_state_dict']
#t_Lyu1 = t['time']

# Compute Linear CCA
t0_CCA = time.perf_counter()
linear_cca_corr, Q1, Q2, Red_Tr, Red_Te = compute_linear_cca(X1, X2, XV1, XV2, XTe1, XTe2, o_dim)
t_CCA = time.perf_counter()-t0_CCA

# Compute the Correlation Coefficients for the other methods
torch.cuda.empty_cache()
data_tr = copy.deepcopy(data_tr_cp)
data_val = copy.deepcopy(data_val_cp)
data_te = copy.deepcopy(data_te_cp)

best_corr_DGCCA_te, best_Ds_DGCCA, best_join_enc_DGCCA = compute_corr_coef_DGCCA(data_tr, data_val, data_te, DGCCA_best_model, o_dim)
mean_corr_DGCCA_te, std_corr_DGCCA_te, all_Ds_DGCCA, all_join_encs_DGCCA = compute_corr_coef_DGCCA_over_init(data_tr, data_val, data_te, DGCCA_all_models, o_dim)
best_corrs_DCCAE_te, best_Ds_DCCAE, best_join_encs_DCCAE = compute_corr_coef_over_lambda(data_tr, data_val, data_te, DCCAE_best_models, o_dim)
mean_corrs_DCCAE_te, std_corrs_DCCAE_te, all_Ds_DCCAE, all_join_encs_DCCAE = compute_corr_coef_over_lambda_and_init(data_tr, data_val, data_te, DCCAE_all_models, o_dim)
best_corrs_XE_te, best_Ds_XE, best_join_encs_XE = compute_corr_coef_over_lambda(data_tr, data_val, data_te, XE_best_models, o_dim)
mean_corrs_XE_te, std_corrs_XE_te, all_Ds_XE, all_join_encs_XE = compute_corr_coef_over_lambda_and_init(data_tr, data_val, data_te, XE_all_models, o_dim)
best_corrs_Lyu_te_0, best_Ds_Lyu_0, best_join_encs_Lyu_0 = compute_corr_coef_over_lambda_for_Lyu(data_tr, data_val, data_te, Lyu_best_models_0, o_dim, i_shape)
mean_corrs_Lyu_te_0, std_corrs_Lyu_te_0, all_Ds_Lyu_0, all_join_encs_Lyu_0 = compute_corr_coef_over_lambda_and_init_for_Lyu(data_tr, data_val, data_te, Lyu_all_models_0, o_dim, i_shape)
best_corrs_Lyu_te_1, best_Ds_Lyu_1, best_join_encs_Lyu_1 = compute_corr_coef_over_lambda_for_Lyu(data_tr, data_val, data_te, Lyu_best_models_1, o_dim, i_shape)
mean_corrs_Lyu_te_1, std_corrs_Lyu_te_1, all_Ds_Lyu_1, all_join_encs_Lyu_1 = compute_corr_coef_over_lambda_and_init_for_Lyu(data_tr, data_val, data_te, Lyu_all_models_1, o_dim, i_shape)

# Predicting the true latent components
print('Prediction begins')
label = Z + 1
label_val = Z_val + 1
labelTe = Z_test + 1
num_clusters = len(np.unique(Z))
p_l = [0.1, 0.3, 0.5, 0.7, 0.9]

# Compute Clustering Accuracy and the other metrics
if True:
    with torch.no_grad():
        # DGCCA
        if True:
            ARI_DGCCA = []
            NMI_DGCCA = []
            ACC_DGCCA = []
            CACC_DGCCA = []
            for i in range(num_init):
                E1 = all_join_encs_DGCCA[i][0].to(torch.device('cpu'))
                E2 = all_join_encs_DGCCA[i][2].to(torch.device('cpu'))
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
                pred_label = kmeans.predict(E2)
                ARI_DGCCA.append(adjusted_rand_score(labelTe, pred_label))
                NMI_DGCCA.append(normalized_mutual_info_score(labelTe, pred_label))
                ACC_DGCCA.append(cluster_acc(labelTe, pred_label))
                clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=True))
                clf.fit(E1, label)
                pred_label = clf.predict(E2)
                CACC_DGCCA.append(accuracy_score(labelTe, pred_label))

        # LCCA
        if True:
            E1 = Red_Tr
            E2 = Red_Te
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
            pred_label = kmeans.predict(E2)
            ARI_LCCA = adjusted_rand_score(labelTe, pred_label)
            NMI_LCCA = normalized_mutual_info_score(labelTe, pred_label)
            ACC_LCCA = cluster_acc(labelTe, pred_label)
            clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
            clf.fit(E1, label)
            pred_label = clf.predict(E2)
            CACC_LCCA = accuracy_score(labelTe, pred_label)

        # The remaining methods
        if True:
            ARI_XE = np.zeros((len(p_l), num_init))
            NMI_XE = np.zeros((len(p_l), num_init))
            ACC_XE = np.zeros((len(p_l), num_init))
            CACC_XE = np.zeros((len(p_l), num_init))
            ARI_DCCAE = np.zeros((len(p_l), num_init))
            NMI_DCCAE = np.zeros((len(p_l), num_init))
            ACC_DCCAE = np.zeros((len(p_l), num_init))
            CACC_DCCAE = np.zeros((len(p_l), num_init))
            ARI_Lyu_0 = np.zeros((len(p_l), num_init))
            NMI_Lyu_0 = np.zeros((len(p_l), num_init))
            ACC_Lyu_0 = np.zeros((len(p_l), num_init))
            CACC_Lyu_0 = np.zeros((len(p_l), num_init))
            ARI_Lyu_1 = np.zeros((len(p_l), num_init))
            NMI_Lyu_1 = np.zeros((len(p_l), num_init))
            ACC_Lyu_1 = np.zeros((len(p_l), num_init))
            CACC_Lyu_1 = np.zeros((len(p_l), num_init))
            for i in range(len(p_l)):
                for s in range(num_init):
                    E1 = all_join_encs_DCCAE[i][s][0].to(torch.device('cpu'))
                    E2 = all_join_encs_DCCAE[i][s][2].to(torch.device('cpu'))
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
                    pred_label = kmeans.predict(E2)
                    ARI_DCCAE[i, s] = adjusted_rand_score(labelTe, pred_label)
                    NMI_DCCAE[i, s] = normalized_mutual_info_score(labelTe, pred_label)
                    ACC_DCCAE[i, s] = cluster_acc(labelTe, pred_label)
                    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
                    clf.fit(E1, label)
                    pred_label = clf.predict(E2)
                    CACC_DCCAE[i, s] = accuracy_score(labelTe, pred_label)

                    E1 = all_join_encs_XE[i][s][0].to(torch.device('cpu'))
                    E2 = all_join_encs_XE[i][s][2].to(torch.device('cpu'))
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
                    pred_label = kmeans.predict(E2)
                    ARI_XE[i, s] = adjusted_rand_score(labelTe, pred_label)
                    NMI_XE[i, s] = normalized_mutual_info_score(labelTe, pred_label)
                    ACC_XE[i, s] = cluster_acc(labelTe, pred_label)
                    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
                    clf.fit(E1, label)
                    pred_label = clf.predict(E2)
                    CACC_XE[i, s] = accuracy_score(labelTe, pred_label)

                    E1 = all_join_encs_Lyu_0[i][s][0].to(torch.device('cpu'))
                    E2 = all_join_encs_Lyu_0[i][s][2].to(torch.device('cpu'))
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
                    pred_label = kmeans.predict(E2)
                    ARI_Lyu_0[i, s] = adjusted_rand_score(labelTe, pred_label)
                    NMI_Lyu_0[i, s] = normalized_mutual_info_score(labelTe, pred_label)
                    ACC_Lyu_0[i, s] = cluster_acc(labelTe, pred_label)
                    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
                    clf.fit(E1, label)
                    pred_label = clf.predict(E2)
                    CACC_Lyu_0[i, s] = accuracy_score(labelTe, pred_label)

                    E1 = all_join_encs_Lyu_1[i][s][0].to(torch.device('cpu'))
                    E2 = all_join_encs_Lyu_1[i][s][2].to(torch.device('cpu'))
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(E1)
                    pred_label = kmeans.predict(E2)
                    ARI_Lyu_1[i, s] = adjusted_rand_score(labelTe, pred_label)
                    NMI_Lyu_1[i, s] = normalized_mutual_info_score(labelTe, pred_label)
                    ACC_Lyu_1[i, s] = cluster_acc(labelTe, pred_label)
                    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
                    clf.fit(E1, label)
                    pred_label = clf.predict(E2)
                    CACC_Lyu_1[i, s] = accuracy_score(labelTe, pred_label)

#########################
# Visualize the results #
#########################
# Print the metric results
if True:
    print('NMI')
    print('XE mean:', np.mean(NMI_XE, 1))
    print('XE std:', np.std(NMI_XE, 1))
    print('DCCAE mean:', np.mean(NMI_DCCAE, 1))
    print('DCCAE std:', np.std(NMI_DCCAE, 1))
    print('Lyu 0 mean:', np.mean(NMI_Lyu_0, 1))
    print('Lyu 0 std:', np.std(NMI_Lyu_0, 1))
    print('Lyu 1 mean:', np.mean(NMI_Lyu_1, 1))
    print('Lyu 1 std:', np.std(NMI_Lyu_1, 1))
    print('DGCCA mean:', np.mean(NMI_DGCCA))
    print('DGCCA std:', np.std(NMI_DGCCA))
    print('CCA', NMI_LCCA)

    print('ARI')
    print('XE mean:', np.mean(ARI_XE, 1))
    print('XE std:', np.std(ARI_XE, 1))
    print('DCCAE mean:', np.mean(ARI_DCCAE, 1))
    print('DCCAE std:', np.std(ARI_DCCAE, 1))
    print('Lyu 0 mean:', np.mean(ARI_Lyu_0, 1))
    print('Lyu 0 std:', np.std(ARI_Lyu_0, 1))
    print('Lyu 1 mean:', np.mean(ARI_Lyu_1, 1))
    print('Lyu 1 std:', np.std(ARI_Lyu_1, 1))
    print('DGCCA mean:', np.mean(ARI_DGCCA))
    print('DGCCA std:', np.std(ARI_DGCCA))
    print('CCA', ARI_LCCA)

    print('ACC')
    print('XE mean:', np.mean(ACC_XE, 1))
    print('XE std:', np.std(ACC_XE, 1))
    print('DCCAE mean:', np.mean(ACC_DCCAE, 1))
    print('DCCAE std:', np.std(ACC_DCCAE, 1))
    print('Lyu 0 mean:', np.mean(ACC_Lyu_0, 1))
    print('Lyu 0 std:', np.std(ACC_Lyu_0, 1))
    print('Lyu 1 mean:', np.mean(ACC_Lyu_1, 1))
    print('Lyu 1 std:', np.std(ACC_Lyu_1, 1))
    print('DGCCA mean:', np.mean(ACC_DGCCA))
    print('DGCCA std:', np.std(ACC_DGCCA))
    print('CCA', ACC_LCCA)

    print('CACC')
    print('XE mean:', np.mean(CACC_XE, 1))
    print('XE std:', np.std(CACC_XE, 1))
    print('DCCAE mean:', np.mean(CACC_DCCAE, 1))
    print('DCCAE std:', np.std(CACC_DCCAE, 1))
    print('Lyu 0 mean:', np.mean(CACC_Lyu_0, 1))
    print('Lyu 0 std:', np.std(CACC_Lyu_0, 1))
    print('Lyu 1 mean:', np.mean(CACC_Lyu_1, 1))
    print('Lyu 1 std:', np.std(CACC_Lyu_1, 1))
    print('DGCCA mean:', np.mean(CACC_DGCCA))
    print('DGCCA std:', np.std(CACC_DGCCA))
    print('CCA', CACC_LCCA)

    print('Times')
    print('DGCCA:', t_DGCCA/(num_init*2))
    print('DCCAE:', t_DCCAE/(num_init*5))
    print('proposed:', t_XE/(num_init*5))
    print('Lyu0:', t_Lyu0/(num_init*5))
    print('Lyu1:', t_Lyu1/(num_init*5))
    print('CCA:', t_CCA)

# Plot the metric results
if True:
    plt.clf()
    plt.plot(p_l, np.mean(ARI_XE, 1), color='blue')
    plt.plot(p_l, np.mean(ARI_DCCAE, 1), color='orange')
    plt.plot(p_l, np.mean(ARI_Lyu_0, 1), color='cyan')
    plt.plot(p_l, np.mean(ARI_Lyu_1, 1), color='black')
    plt.plot(p_l, np.mean(ARI_DGCCA) * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, ARI_LCCA * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, np.mean(ARI_XE, 1) - np.std(ARI_XE, 1), np.mean(ARI_XE, 1) + np.std(ARI_XE, 1), alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, np.mean(ARI_DCCAE, 1) - np.std(ARI_DCCAE, 1), np.mean(ARI_DCCAE, 1) + np.std(ARI_DCCAE, 1), alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, np.mean(ARI_Lyu_0, 1) - np.std(ARI_Lyu_0, 1), np.mean(ARI_Lyu_0, 1) + np.std(ARI_Lyu_0, 1), alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, np.mean(ARI_Lyu_1, 1) - np.std(ARI_Lyu_1, 1), np.mean(ARI_Lyu_1, 1) + np.std(ARI_Lyu_1, 1), alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(p_l, (np.mean(ARI_DGCCA) - np.std(ARI_DGCCA)) * np.ones(len(p_l)), (np.mean(ARI_DGCCA) + np.std(ARI_DGCCA)) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al. (beta = 0)', 'Lyu et al. (beta = 0.001)', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/ari.pdf', format="pdf")

    plt.clf()
    plt.plot(p_l, np.mean(NMI_XE, 1), color='blue')
    plt.plot(p_l, np.mean(NMI_DCCAE, 1), color='orange')
    plt.plot(p_l, np.mean(NMI_Lyu_0, 1), color='cyan')
    plt.plot(p_l, np.mean(NMI_Lyu_1, 1), color='black')
    plt.plot(p_l, np.mean(NMI_DGCCA) * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, NMI_LCCA * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, np.mean(NMI_XE, 1) - np.std(NMI_XE, 1), np.mean(NMI_XE, 1) + np.std(NMI_XE, 1), alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, np.mean(NMI_DCCAE, 1) - np.std(NMI_DCCAE, 1), np.mean(NMI_DCCAE, 1) + np.std(NMI_DCCAE, 1), alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, np.mean(NMI_Lyu_0, 1) - np.std(NMI_Lyu_0, 1), np.mean(NMI_Lyu_0, 1) + np.std(NMI_Lyu_0, 1), alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, np.mean(NMI_Lyu_1, 1) - np.std(NMI_Lyu_1, 1), np.mean(NMI_Lyu_1, 1) + np.std(NMI_Lyu_1, 1), alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(p_l, (np.mean(NMI_DGCCA) - np.std(NMI_DGCCA)) * np.ones(len(p_l)), (np.mean(NMI_DGCCA) + np.std(NMI_DGCCA)) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al. (beta = 0)', 'Lyu et al. (beta = 0.001)', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("NMI")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/nmi.pdf', format="pdf")

    plt.clf()
    plt.plot(p_l, np.mean(ACC_XE, 1), color='blue')
    plt.plot(p_l, np.mean(ACC_DCCAE, 1), color='orange')
    plt.plot(p_l, np.mean(ACC_Lyu_0, 1), color='cyan')
    plt.plot(p_l, np.mean(ACC_Lyu_1, 1), color='black')
    plt.plot(p_l, np.mean(ACC_DGCCA) * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, ACC_LCCA * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, np.mean(ACC_XE, 1) - np.std(ACC_XE, 1), np.mean(ACC_XE, 1) + np.std(ACC_XE, 1), alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, np.mean(ACC_DCCAE, 1) - np.std(ACC_DCCAE, 1), np.mean(ACC_DCCAE, 1) + np.std(ACC_DCCAE, 1), alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, np.mean(ACC_Lyu_0, 1) - np.std(ACC_Lyu_0, 1), np.mean(ACC_Lyu_0, 1) + np.std(ACC_Lyu_0, 1), alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, np.mean(ACC_Lyu_1, 1) - np.std(ACC_Lyu_1, 1), np.mean(ACC_Lyu_1, 1) + np.std(ACC_Lyu_1, 1), alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(p_l, (np.mean(ACC_DGCCA) - np.std(ACC_DGCCA)) * np.ones(len(p_l)), (np.mean(ACC_DGCCA) + np.std(ACC_DGCCA)) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al. (beta = 0)', 'Lyu et al. (beta = 0.001)', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("ACC")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/acc.pdf', format="pdf")

    plt.clf()
    plt.plot(p_l, np.mean(CACC_XE, 1), color='blue')
    plt.plot(p_l, np.mean(CACC_DCCAE, 1), color='orange')
    plt.plot(p_l, np.mean(CACC_Lyu_0, 1), color='cyan')
    plt.plot(p_l, np.mean(CACC_Lyu_1, 1), color='black')
    plt.plot(p_l, np.mean(CACC_DGCCA) * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, CACC_LCCA * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, np.mean(CACC_XE, 1) - np.std(CACC_XE, 1), np.mean(CACC_XE, 1) + np.std(CACC_XE, 1), alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, np.mean(CACC_DCCAE, 1) - np.std(CACC_DCCAE, 1), np.mean(CACC_DCCAE, 1) + np.std(CACC_DCCAE, 1), alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, np.mean(CACC_Lyu_0, 1) - np.std(CACC_Lyu_0, 1), np.mean(CACC_Lyu_0, 1) + np.std(CACC_Lyu_0, 1), alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, np.mean(CACC_Lyu_1, 1) - np.std(CACC_Lyu_1, 1), np.mean(CACC_Lyu_1, 1) + np.std(CACC_Lyu_1, 1), alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(p_l, (np.mean(CACC_DGCCA) - np.std(CACC_DGCCA)) * np.ones(len(p_l)), (np.mean(CACC_DGCCA) + np.std(CACC_DGCCA)) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al. (beta = 0)', 'Lyu et al. (beta = 0.001)', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("CACC")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/cacc.pdf', format="pdf")

# Plot Correlation Coefficients and the walltimes of all the methods
if True:
    plt.clf()
    plt.plot(p_l, mean_corrs_XE_te, color='blue')
    plt.plot(p_l, mean_corrs_DCCAE_te, color='orange')
    plt.plot(p_l, mean_corrs_Lyu_te_0, color='cyan')
    plt.plot(p_l, mean_corrs_Lyu_te_1, color='black')
    plt.plot(p_l, mean_corr_DGCCA_te * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, linear_cca_corr[2] * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, mean_corrs_XE_te - std_corrs_XE_te, mean_corrs_XE_te + std_corrs_XE_te, alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, mean_corrs_DCCAE_te - std_corrs_DCCAE_te, mean_corrs_DCCAE_te + std_corrs_DCCAE_te, alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, mean_corrs_Lyu_te_0 - std_corrs_Lyu_te_0, mean_corrs_Lyu_te_0 + std_corrs_Lyu_te_0, alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, mean_corrs_Lyu_te_1 - std_corrs_Lyu_te_1, mean_corrs_Lyu_te_1 + std_corrs_Lyu_te_1, alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(p_l, (mean_corr_DGCCA_te - std_corr_DGCCA_te) * np.ones(len(p_l)), (mean_corr_DGCCA_te + std_corr_DGCCA_te) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al. (beta = 0)', 'Lyu et al. (beta = 0.001)', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("Aver. Corr. Coef.")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/corrs.pdf', format="pdf")

    times = [t_XE/(num_init*5), t_DCCAE/(num_init*5), t_Lyu0/(num_init*5), t_Lyu1/(num_init*5), t_DGCCA/(2*num_init), t_CCA]
    methods = ['proposed', 'DCCAE', 'Lyu et al.\n(beta = 0)', 'Lyu et al.\n(beta = 0.001)', 'DGCCA', 'LCCA']
    plt.clf()
    plt.bar(methods, times, color='blue', width=0.4)
    plt.xlabel("Methods")
    plt.ylabel("Walltime (sec)")
    plt.xticks(rotation=30, ha='right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + '/times.pdf', format="pdf")


# Save results
if True:
    mdic = {"CACC_XE": CACC_XE,  "CACC_DCCAE": CACC_DCCAE, "CACC_DGCCA": CACC_DGCCA, "CACC_Lyu_0": CACC_Lyu_0, "CACC_Lyu_1": CACC_Lyu_1, "CACC_LCCA": CACC_LCCA,
             "ACC_XE": ACC_XE,  "ACC_DCCAE": ACC_DCCAE, "ACC_DGCCA": ACC_DGCCA, "ACC_Lyu_0": ACC_Lyu_0, "ACC_Lyu_1": ACC_Lyu_1, "ACC_LCCA": ACC_LCCA, "NMI_XE": NMI_XE,
            "NMI_DCCAE": NMI_DCCAE, "NMI_DGCCA": NMI_DGCCA, "NMI_LCAA": NMI_LCCA, "NMI_Lyu_0": NMI_Lyu_0, "NMI_Lyu_1": NMI_Lyu_1, "ARI_XE": ARI_XE,
            "ARI_DCCAE": ARI_DCCAE, "ARI_DGCCA": ARI_DGCCA, "ARI_LCCA": ARI_LCCA, "ARI_Lyu_0": ARI_Lyu_0, "ARI_Lyu_1": ARI_Lyu_1, "all_Ds_XE": all_Ds_XE,
            "all_Ds_DCCAE": all_Ds_DCCAE, "all_Ds_DGCCA": all_Ds_DGCCA, "all_Ds_Lyu_0": all_Ds_Lyu_0, "all_Ds_Lyu_1": all_Ds_Lyu_1,
            "linear_cca_corr": linear_cca_corr[2]}
    savemat(path + "/results.mat", mdic)

