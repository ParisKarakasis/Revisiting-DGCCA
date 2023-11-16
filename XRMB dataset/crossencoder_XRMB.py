import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from crossenc_utils import *
import random
import copy
import sys
import os
import time
from scipy.io import savemat

a = sys.argv[1:]
ID = str(a[0])  # ID OF THE EXPERIMENT.
o_dim = int(a[1])  # Dimension of the latent space

lr = 0.01 * 1e-1
epoch_num = 200
batch_size = 500
reg_par = 0.001
dropout = 0
seed = 65
num_init = 5

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

# Load the dataset
dataset = 0
print("Phoneme")
X1, X2, X3, XV1, XV2, XV3, XTe1, XTe2, XTe3, label, labelTe = load_reduced_phoneme_dataset()
i_shape = [X1.shape[1], X2.shape[1], X3.shape[1]]
print(i_shape)
num_clusters = 40

# Standardize the data (to generalize)
X1, X2, X3, XV1, XV2, XV3, XTe1, XTe2, XTe3, data_tr, data_val, data_te, X1_mean, X1_std, X2_mean, X2_std, X3_mean, X3_std = my_standardize4(
    X1, X2, X3, XV1, XV2, XV3, XTe1, XTe2, XTe3, batch_size)

# Copy the data
data_tr_cp = copy.deepcopy(data_tr)
data_val_cp = copy.deepcopy(data_val)
data_te_cp = copy.deepcopy(data_te)

# Run DGCCA
t0_DGCCA = time.perf_counter()
DGCCA_best_model, DGCCA_best_loss, DGCCA_all_models = train_DGCCA(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te, dataset)
t_DGCCA = time.perf_counter() - t0_DGCCA
torch.save({'model_state_dict': DGCCA_best_model, 'loss': DGCCA_best_loss, 'models_state_dict': DGCCA_all_models, 'time': t_DGCCA}, path + '/DGCCAModel')
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
t_DCCAE = time.perf_counter() - t0_DCCAE
torch.save({'model_state_dict': DCCAE_best_models, 'loss': DCCAE_best_losses, 'models_state_dict': DCCAE_all_models, 'time': t_DCCAE}, path + '/DCCAEModels')
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
t_XE = time.perf_counter() - t0_XE
torch.save({'model_state_dict': XE_best_models, 'loss': XE_best_losses, 'models_state_dict': XE_all_models, 'time': t_XE}, path + '/XEModels')
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
    args.o_dim2 = [60, 20, 0]
    args.num_iters = epoch_num
    args.batchsize1 = batch_size
    args.batchsize2 = 1000
    args.lr_max = 1e-6
    args.lr_min = lr
    args.weight_decay = reg_par
    args.phi_num_layers = 3
    args.phi_hidden_size = 64
    args.tau_num_layers = 3
    args.tau_hidden_size = 64
    args.inner_epochs = 1
    args._lambda = 1e-6

t_Lyu = time.perf_counter()
Lyu_best_models, Lyu_best_losses, Lyu_all_models = train_Lyu_et_all(i_shape, args, num_init, data_tr, data_val, data_te)
t_Lyu = time.perf_counter() - t_Lyu
torch.save({'model_state_dict': Lyu_best_models, 'loss': Lyu_best_losses, 'models_state_dict': Lyu_all_models, 'time': t_Lyu}, path + '/LyuModels')
#t = torch.load(folder_path+'/LyuModels', map_location=torch.device('cuda'))
#Lyu_best_models = t['model_state_dict']
#Lyu_best_losses = t['loss']
#Lyu_all_models = t['models_state_dict']
#t_Lyu = t['time']

# Compute Linear CCA
i_shape = [X1.shape[1], X2.shape[1], X3.shape[1]]
X = [X1, X2, X3]
XV = [XV1, XV2, XV3]
XTe = [XTe1, XTe2, XTe3]
t0_CCA = time.perf_counter()
linear_cca_corr, Q, MAXVAR_enc, Lambda = compute_MAXVAR(X, XV, XTe, o_dim)
t_CCA = time.perf_counter()-t0_CCA

# Compute the Correlation Coefficients for the other baselines
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
best_corrs_Lyu_te, best_Ds_Lyu, best_join_encs_Lyu = compute_corr_coef_over_lambda_for_Lyu(data_tr, data_val, data_te, Lyu_best_models, o_dim, i_shape)
mean_corrs_Lyu_te, std_corrs_Lyu_te, all_Ds_Lyu, all_join_encs_Lyu = compute_corr_coef_over_lambda_and_init_for_Lyu(data_tr, data_val, data_te, Lyu_all_models, o_dim, i_shape)

print('Computing the accuracy metric')
p_l = [0.1, 0.3, 0.5, 0.7, 0.9]

# Compute Clustering Accuracy and the other metrics
if True:
    with torch.no_grad():
        # DGCCA
        ACC_DGCCA = []
        for i in range(num_init):
            print('DGCCA')
            E1 = all_join_encs_DGCCA[i][0].to(torch.device('cpu'))
            E2 = all_join_encs_DGCCA[i][2].to(torch.device('cpu'))
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(E1, label)
            pred_label = neigh.predict(E2)
            ACC_DGCCA.append(accuracy_score(labelTe, pred_label))

        # LCCA
        print('LCCA')
        E1 = MAXVAR_enc[0]
        E2 = MAXVAR_enc[2]
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(E1, label)
        pred_label = neigh.predict(E2)
        ACC_LCCA = accuracy_score(labelTe, pred_label)

        # The remaining baselines
        ACC_XE = np.zeros((len(p_l), num_init))
        ACC_DCCAE = np.zeros((len(p_l), num_init))
        ACC_Lyu = np.zeros((len(p_l), num_init))
        for i in range(len(p_l)):
            print('Other baselines')
            for s in range(num_init):
                print('DCCAE')
                E1 = all_join_encs_DCCAE[i][s][0].to(torch.device('cpu'))
                E2 = all_join_encs_DCCAE[i][s][2].to(torch.device('cpu'))
                neigh = KNeighborsClassifier(n_neighbors=5)
                neigh.fit(E1, label)
                pred_label = neigh.predict(E2)
                ACC_DCCAE[i, s] = accuracy_score(labelTe, pred_label)

                print('XE')
                E1 = all_join_encs_XE[i][s][0].to(torch.device('cpu'))
                E2 = all_join_encs_XE[i][s][2].to(torch.device('cpu'))
                neigh = KNeighborsClassifier(n_neighbors=5)
                neigh.fit(E1, label)
                pred_label = neigh.predict(E2)
                ACC_XE[i, s] = accuracy_score(labelTe, pred_label)

                print('Lyu')
                E1 = all_join_encs_Lyu[i][s][0].to(torch.device('cpu'))
                E2 = all_join_encs_Lyu[i][s][2].to(torch.device('cpu'))
                neigh = KNeighborsClassifier(n_neighbors=5)
                neigh.fit(E1, label)
                pred_label = neigh.predict(E2)
                ACC_Lyu[i, s] = accuracy_score(labelTe, pred_label)

#########################
# Visualize the results #
#########################
# Print the metric results
if True:
    print('ACC')
    print('XE mean:', np.mean(ACC_XE, 1))
    print('XE std:', np.std(ACC_XE, 1))
    print('DCCAE mean:', np.mean(ACC_DCCAE, 1))
    print('DCCAE std:', np.std(ACC_DCCAE, 1))
    print('Lyu mean:', np.mean(ACC_Lyu, 1))
    print('Lyu std:', np.std(ACC_Lyu, 1))
    print('DGCCA mean:', np.mean(ACC_DGCCA))
    print('DGCCA std:', np.std(ACC_DGCCA))
    print('CCA', ACC_LCCA)

# Plot the results
if True:
    plt.clf()
    plt.plot(p_l, np.mean(ACC_XE, 1), color='blue')
    plt.plot(p_l, np.mean(ACC_DCCAE, 1), color='orange')
    plt.plot(p_l, np.mean(ACC_Lyu, 1), color='cyan')
    plt.plot(p_l, np.mean(ACC_DGCCA) * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, ACC_LCCA * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, np.mean(ACC_XE, 1) - np.std(ACC_XE, 1), np.mean(ACC_XE, 1) + np.std(ACC_XE, 1), alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, np.mean(ACC_DCCAE, 1) - np.std(ACC_DCCAE, 1), np.mean(ACC_DCCAE, 1) + np.std(ACC_DCCAE, 1), alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, np.mean(ACC_Lyu, 1) - np.std(ACC_Lyu, 1), np.mean(ACC_Lyu, 1) + np.std(ACC_Lyu, 1), alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, (np.mean(ACC_DGCCA) - np.std(ACC_DGCCA)) * np.ones(len(p_l)), (np.mean(ACC_DGCCA) + np.std(ACC_DGCCA)) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al.', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("ACC")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/acc.pdf', format="pdf")

# Plot Correlation Coefficients and the walltimes of all the methods
if True:
    plt.clf()
    plt.plot(p_l, mean_corrs_XE_te, color='blue')
    plt.plot(p_l, mean_corrs_DCCAE_te, color='orange')
    plt.plot(p_l, mean_corrs_Lyu_te, color='cyan')
    plt.plot(p_l, mean_corr_DGCCA_te * np.ones((len(p_l), 1)), color='green')
    plt.plot(p_l, linear_cca_corr[2] * np.ones((len(p_l), 1)), color='red')
    plt.fill_between(p_l, mean_corrs_XE_te - std_corrs_XE_te, mean_corrs_XE_te + std_corrs_XE_te, alpha=0.2, edgecolor='blue', facecolor='blue')
    plt.fill_between(p_l, mean_corrs_DCCAE_te - std_corrs_DCCAE_te, mean_corrs_DCCAE_te + std_corrs_DCCAE_te, alpha=0.2, edgecolor='orange', facecolor='orange')
    plt.fill_between(p_l, mean_corrs_Lyu_te - std_corrs_Lyu_te, mean_corrs_Lyu_te + std_corrs_Lyu_te, alpha=0.2, edgecolor='cyan', facecolor='cyan')
    plt.fill_between(p_l, (mean_corr_DGCCA_te - std_corr_DGCCA_te) * np.ones(len(p_l)), (mean_corr_DGCCA_te + std_corr_DGCCA_te) * np.ones(len(p_l)), alpha=0.2, edgecolor='green', facecolor='green')
    plt.legend(['proposed', 'DCCAE', 'Lyu et al.', 'DGCCA', 'LCCA'])
    plt.xlabel("lambda")
    plt.ylabel("Aver. Corr. Coef.")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/corrs.pdf', format="pdf")

    times = [t_XE/(num_init*5), t_DCCAE/(num_init*5), t_Lyu/(num_init*5), t_DGCCA/(2*num_init), t_CCA]
    methods = ['proposed', 'DCCAE', 'Lyu et al.', 'DGCCA', 'LCCA']
    plt.clf()
    plt.bar(methods, times, color='blue', width=0.4)
    plt.xlabel("Methods")
    plt.ylabel("Walltime (sec)")
    plt.xticks(rotation=30, ha='right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + '/times.pdf', format="pdf")

# Store results
mdic = {"ACC_XE": ACC_XE, "ACC_DCCAE": ACC_DCCAE, "ACC_DGCCA": ACC_DGCCA, "ACC_LCCA": ACC_LCCA, "ACC_LCCA": ACC_LCCA, "ACC_Lyu": ACC_Lyu,
        "all_Ds_XE": all_Ds_XE, "all_Ds_DCCAE": all_Ds_DCCAE, "all_Ds_DGCCA": all_Ds_DGCCA, "all_Ds_Lyu": all_Ds_Lyu,
        "linear_cca_corr": linear_cca_corr}

savemat(path + "/results2.mat", mdic)

# Plot encodings
if False:
    cmap = matplotlib.cm.get_cmap('jet')
    colors_te = np.zeros((len(labelTe), 4))
    for i in range(len(labelTe)):
        colors_te[i, :] = cmap((labelTe[i] + 1) / (num_clusters + 1))

    X1_emb = TSNE(n_components=2, init='random').fit_transform(MAXVAR_enc[2])
    plt.clf()
    plt.scatter(X1_emb[:, 0], X1_emb[:, 1], c=colors_te)
    plt.title("Multiview Mnist LCCA")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/LCCA.pdf', format="pdf")

    E1 = best_join_enc_DGCCA[2].to(torch.device('cpu'))
    X1_emb = TSNE(n_components=2, init='random').fit_transform(E1)
    plt.clf()
    plt.scatter(X1_emb[:, 0], X1_emb[:, 1], c=colors_te)
    plt.title("Multiview Mnist DGCCA")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    # plt.show()
    plt.savefig(path + '/DGCCA.pdf', format="pdf")

    for i in range(len(p_l)):
        E1 = best_join_encs_XE[i][2].to(torch.device('cpu'))
        X1_emb = TSNE(n_components=2, init='random').fit_transform(E1)
        plt.clf()
        plt.scatter(X1_emb[:, 0], X1_emb[:, 1], c=colors_te)
        plt.title('Multiview Mnist ' + str(p_l[i]) + ' XE')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        # plt.show()
        plt.savefig(path + '/XE' + str(p_l[i]) + '.pdf', format="pdf")

        E1 = best_join_encs_DCCAE[i][2].to(torch.device('cpu'))
        X1_emb = TSNE(n_components=2, init='random').fit_transform(E1)
        plt.clf()
        plt.scatter(X1_emb[:, 0], X1_emb[:, 1], c=colors_te)
        plt.title('Multiview Mnist ' + str(p_l[i]) + ' DCCAE')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        # plt.show()
        plt.savefig(path + '/DCCAE' + str(p_l[i]) + '.pdf', format="pdf")

        E1 = best_join_encs_Lyu[i][2].to(torch.device('cpu'))
        X1_emb = TSNE(n_components=2, init='random').fit_transform(E1)
        plt.clf()
        plt.scatter(X1_emb[:, 0], X1_emb[:, 1], c=colors_te)
        plt.title('Multiview Mnist ' + str(p_l[i]) + ' Lyu et al.')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        # plt.show()
        plt.savefig(path + '/Lyu' + str(p_l[i]) + '.pdf', format="pdf")
