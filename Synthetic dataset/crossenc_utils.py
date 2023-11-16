import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import *
from scipy.optimize import linear_sum_assignment as linear_assignment
import sklearn.datasets as skd
import scipy.io
import copy
import Lyu_Understanding.multiview_and_self_supervision.utils as l_utils
import Lyu_Understanding.multiview_and_self_supervision.train as l_train
import Lyu_Understanding.multiview_and_self_supervision.model as l_model
from torch import cat

device = torch.device('cuda')


def standardize_1(x):
    """
    FOR STANDARDIZING A 1D VECTOR.
    """
    mean_x = np.mean(x)
    x = x - mean_x
    var = np.dot(x.T, x)
    x = x / np.sqrt(var)
    return x


def my_standardize2(X, batch_size):
    """
    STANDARDIZING D DIMENSIONAL DATA
    """
    X_mean = 0 * np.mean(X, axis=0)
    X = X - X_mean
    X_std = np.linalg.norm(X, axis=0) / np.sqrt(batch_size)
    X_std[X_std == 0] = 1
    X = X / X_std
    return X, X_mean, X_std


def my_standardize3(X1, X2, XV1, XV2, XTe1, XTe2, batch_size):
    X1, X1_mean, X1_std = my_standardize2(X1, batch_size)
    XV1 = (XV1 - X1_mean) / X1_std
    XTe1 = (XTe1 - X1_mean) / X1_std
    X2, X2_mean, X2_std = my_standardize2(X2, batch_size)
    XV2 = (XV2 - X2_mean) / X2_std
    XTe2 = (XTe2 - X2_mean) / X2_std

    # Move data to device
    X1 = torch.from_numpy(X1).float().to(device)
    XV1 = torch.from_numpy(XV1).float().to(device)
    XTe1 = torch.from_numpy(XTe1).float().to(device)
    X2 = torch.from_numpy(X2).float().to(device)
    XV2 = torch.from_numpy(XV2).float().to(device)
    XTe2 = torch.from_numpy(XTe2).float().to(device)

    data_tr = torch.cat((X1, X2), 1).to(device)
    data_val = torch.cat((XV1, XV2), 1).to(device)
    data_te = torch.cat((XTe1, XTe2), 1).to(device)

    return X1, X2, XV1, XV2, XTe1, XTe2, data_tr, data_val, data_te, X1_mean, X1_std, X2_mean, X2_std


def my_standardize4(X1, X2, X3, XV1, XV2, XV3, XTe1, XTe2, XTe3, batch_size):
    X1, X1_mean, X1_std = my_standardize2(X1, batch_size)
    XV1 = (XV1 - X1_mean) / X1_std
    XTe1 = (XTe1 - X1_mean) / X1_std
    X2, X2_mean, X2_std = my_standardize2(X2, batch_size)
    XV2 = (XV2 - X2_mean) / X2_std
    XTe2 = (XTe2 - X2_mean) / X2_std
    X3, X3_mean, X3_std = my_standardize2(X3, batch_size)
    XV3 = (XV3 - X3_mean) / X3_std
    XTe3 = (XTe3 - X3_mean) / X3_std

    # Move data to device
    X1 = torch.from_numpy(X1).float().to(device)
    XV1 = torch.from_numpy(XV1).float().to(device)
    XTe1 = torch.from_numpy(XTe1).float().to(device)
    X2 = torch.from_numpy(X2).float().to(device)
    XV2 = torch.from_numpy(XV2).float().to(device)
    XTe2 = torch.from_numpy(XTe2).float().to(device)
    X3 = torch.from_numpy(X3).float().to(device)
    XV3 = torch.from_numpy(XV3).float().to(device)
    XTe3 = torch.from_numpy(XTe3).float().to(device)

    data_tr = torch.cat((X1, X2, X3), 1).to(device)
    data_val = torch.cat((XV1, XV2, XV3), 1).to(device)
    data_te = torch.cat((XTe1, XTe2, XTe3), 1).to(device)

    return X1, X2, X3, XV1, XV2, XV3, XTe1, XTe2, XTe3, data_tr, data_val, data_te, X1_mean, X1_std, X2_mean, X2_std, X3_mean, X3_std


def generate_synthetic_dataset2(m, d_dp, d, i_shape, rng, SPR):
    # Create synthetic data in the latent space (for training, validation, testing) -- Gaussian mixture version

    # Sample the common factor (one and discrete in this case)
    K = np.int(d)
    alphabet = np.arange(K)
    PMF = np.arange(K)+1
    PMF = np.squeeze(PMF / np.sum(PMF))
    Z = rng.choice(alphabet, m[0], p=PMF)
    Z_val = rng.choice(alphabet, m[1], p=PMF)
    Z_test = rng.choice(alphabet, m[2], p=PMF)

    # Create the mean vectors and the corresponding covariance matrices for the gaussian dists of the mixtures
    mean = []
    Cov = []
    for i in range(len(d_dp)):
        tmp = []
        for k in range(K):
            tmp.append(skd.make_spd_matrix(d_dp[i], random_state=k+50*i))
        mean.append(0 * np.random.randn(d_dp[i], K))
        Cov.append(tmp)

    # Sample from the mixtures
    V1 = np.zeros((m[0], d_dp[0]))
    V2 = np.zeros((m[0], d_dp[1]))
    for i in range(m[0]):
        z = Z[i]
        V1[i] = rng.multivariate_normal(mean[0][:, z], Cov[0][z], 1)
        V2[i] = rng.multivariate_normal(mean[1][:, z], Cov[1][z], 1)

    V1_val = np.zeros((m[1], d_dp[0]))
    V2_val = np.zeros((m[1], d_dp[1]))
    for i in range(m[1]):
        z = Z_val[i]
        V1_val[i] = rng.multivariate_normal(mean[0][:, z], Cov[0][z], 1)
        V2_val[i] = rng.multivariate_normal(mean[1][:, z], Cov[1][z], 1)

    V1_test = np.zeros((m[2], d_dp[0]))
    V2_test = np.zeros((m[2], d_dp[1]))
    for i in range(m[2]):
        z = Z_test[i]
        V1_test[i] = rng.multivariate_normal(mean[0][:, z], Cov[0][z], 1)
        V2_test[i] = rng.multivariate_normal(mean[1][:, z], Cov[1][z], 1)

    j = 10 ** (SPR / 10)
    alpha = (m[0] * d_dp[0]) / (d * j * (np.linalg.norm(V1, 'fro') ** 2))
    V1 = np.sqrt(alpha) * V1
    alpha = (m[0] * d_dp[1]) / (d * j * (np.linalg.norm(V2, 'fro') ** 2))
    V2 = np.sqrt(alpha) * V2

    alpha = (m[1] * d_dp[0]) / (d * j * (np.linalg.norm(V1_val, 'fro') ** 2))
    V1_val = np.sqrt(alpha) * V1_val
    alpha = (m[1] * d_dp[1]) / (d * j * (np.linalg.norm(V2_val, 'fro') ** 2))
    V2_val = np.sqrt(alpha) * V2_val

    alpha = (m[2] * d_dp[0]) / (d * j * (np.linalg.norm(V1_test, 'fro') ** 2))
    V1_test = np.sqrt(alpha) * V1_test
    alpha = (m[2] * d_dp[1]) / (d * j * (np.linalg.norm(V2_test, 'fro') ** 2))
    V2_test = np.sqrt(alpha) * V2_test

    # Create the transformations
    f1, f2 = GenModel(d_dp[0] + d, i_shape[0]), GenModel(d_dp[1] + d, i_shape[1])
    f1.apply(weight_init)
    f2.apply(weight_init)

    # Compose the dataset
    U = np.eye(K)
    X1, X2 = torch.zeros((m[0], i_shape[0])), torch.zeros((m[0], i_shape[1]))
    for i, (z, v1, v2) in enumerate(zip(Z, V1, V2)):
        z_p = torch.from_numpy(U[z]).float()
        v1p = torch.from_numpy(v1).float()
        v2p = torch.from_numpy(v2).float()
        X1[i], X2[i] = f1(torch.cat((z_p, v1p), 0)), f2(torch.cat((z_p, v2p), 0))

    XV1, XV2 = torch.zeros((m[1], i_shape[0])), torch.zeros((m[1], i_shape[1]))
    for i, (z, v1, v2) in enumerate(zip(Z_val, V1_val, V2_val)):
        z_p = torch.from_numpy(U[z]).float()
        v1p = torch.from_numpy(v1).float()
        v2p = torch.from_numpy(v2).float()
        XV1[i], XV2[i] = f1(torch.cat((z_p, v1p), 0)), f2(torch.cat((z_p, v2p), 0))

    XTe1, XTe2 = torch.zeros((m[2], i_shape[0])), torch.zeros((m[2], i_shape[1]))
    for i, (z, v1, v2) in enumerate(zip(Z_test, V1_test, V2_test)):
        z_p = torch.from_numpy(U[z]).float()
        v1p = torch.from_numpy(v1).float()
        v2p = torch.from_numpy(v2).float()
        XTe1[i], XTe2[i] = f1(torch.cat((z_p, v1p), 0)), f2(torch.cat((z_p, v2p), 0))

    X1 = X1.detach().numpy()
    X2 = X2.detach().numpy()
    XV1 = XV1.detach().numpy()
    XV2 = XV2.detach().numpy()
    XTe1 = XTe1.detach().numpy()
    XTe2 = XTe2.detach().numpy()

    return X1, X2, XV1, XV2, XTe1, XTe2, Z, Z_val, Z_test


def train_DGCCA(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te,
                dataset):
    lambda_ = 0
    val_loss = 1000
    model_file_name = "best_DGCCA_model.pth"
    models_over_inits =[]
    print('Training DGCCA Started')
    torch.cuda.empty_cache()
    for i in range(num_init):
        model = CEModel(i_shape, o_dim, dataset, dropout).to(device)
        model_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_par, amsgrad=True)
        model, losses, best_epoch = train(data_tr, data_val, data_te, model, model_opt, batch_size, epoch_num, i_shape,
                                          lambda_)
        models_over_inits.append(model)
        if val_loss > losses[best_epoch, 1]:
            best_model = model
            best_losses = losses
            val_loss = losses[best_epoch, 1]


    # torch.save({'model_state_dict': best_model.state_dict(), 'loss': best_losses}, model_file_name)
    # print('Saved the best DGCCA model')

    return best_model, best_losses, models_over_inits


def train_DCCAE(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te,
                 dataset):
    models = []
    models_over_all = []
    loss_l = []
    model_file_name = "best_DCCAE_model.pth"
    print('Training DCCAE Started')
    p_l = [0.1, 0.3, 0.5, 0.7, 0.9]
    for lambda_ in p_l:
        val_loss = 1000
        torch.cuda.empty_cache()
        models_over_inits = []
        for i in range(num_init):
            model = DCCAEModel(i_shape, o_dim, dataset, dropout).to(device)
            model_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_par, amsgrad=True)
            model, losses, best_epoch = train(data_tr, data_val, data_te, model, model_opt, batch_size, epoch_num,
                                              i_shape, lambda_)
            models_over_inits.append(model)
            if val_loss > losses[best_epoch, 1]:
                best_model = model
                best_losses = losses
                val_loss = losses[best_epoch, 1]

        models_over_all.append(models_over_inits)
        models.append(best_model)
        loss_l.append(best_losses)

    # torch.save({'model_state_dict': best_model.state_dict(), 'loss': best_losses}, model_file_name)
    # print('Saved the best DGCCA model')

    return models, loss_l, models_over_all


def train_XE(i_shape, o_dim, dropout, lr, reg_par, batch_size, epoch_num, num_init, data_tr, data_val, data_te,
             dataset):
    models = []
    models_over_all = []
    loss_l = []
    model_file_name = "best_XE_model.pth"
    print('Training XE Started')
    p_l = [0.1, 0.3, 0.5, 0.7, 0.9]
    for lambda_ in p_l:
        val_loss = 1000
        torch.cuda.empty_cache()
        models_over_inits = []
        for i in range(num_init):
            model = CEModel(i_shape, o_dim, dataset, dropout).to(device)
            model_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_par, amsgrad=True)
            model, losses, best_epoch = train(data_tr, data_val, data_te, model, model_opt, batch_size, epoch_num,
                                              i_shape, lambda_)
            models_over_inits.append(model)
            if val_loss > losses[best_epoch, 1]:
                best_model = model
                best_losses = losses
                val_loss = losses[best_epoch, 1]

        models_over_all.append(models_over_inits)
        models.append(best_model)
        loss_l.append(best_losses)

    # torch.save({'model_state_dict': best_model.state_dict(), 'loss': best_losses}, model_file_name)
    # print('Saved the best DGCCA model')

    return models, loss_l, models_over_all


def train_Lyu_et_all(i_shape, args, num_init, data_tr, data_val, data_te):
    models = []
    models_over_all = []
    loss_l = []
    model_file_name = 'best_Lyu_model.pth'
    print('Training Lyu model Started')
    p_l = [0.1, 0.3, 0.5, 0.7, 0.9]
    for lambda_ in p_l:
        val_loss = 1000
        args.beta = lambda_
        torch.cuda.empty_cache()
        models_over_inits = []
        for i in range(num_init):
            if True:
                # Initiate the models
                ae_model = l_model.DAE(args.o_dim, args.o_dim2, i_shape, 0).to(device)

                if args._lambda > 0:
                    # View1 independence regularization network
                    mmcca1 = l_model.MMDCCA(args.o_dim, args.o_dim2[0], [args.phi_hidden_size] * args.phi_num_layers,
                                            [args.tau_hidden_size] * args.tau_num_layers).to(device)
                    # View2 independence regularization network
                    mmcca2 = l_model.MMDCCA(args.o_dim, args.o_dim2[1], [args.phi_hidden_size] * args.phi_num_layers,
                                            [args.tau_hidden_size] * args.tau_num_layers).to(device)

                    # Optimizer
                    optimizer = torch.optim.AdamW([
                        {'params': mmcca1.parameters(), 'lr': args.lr_max},
                        {'params': mmcca2.parameters(), 'lr': args.lr_max},
                        {'params': ae_model.parameters()}
                    ], lr=args.lr_min, weight_decay=args.weight_decay, amsgrad=True)
                else:
                    mmcca1 = []
                    mmcca2 = []
                    optimizer = torch.optim.AdamW(ae_model.parameters(), lr=args.lr_min, weight_decay=args.weight_decay,
                                                  amsgrad=True)

                # Construct data loaders
                view1 = data_tr[:, :i_shape[0]]
                view2 = data_tr[:, i_shape[0]:]
                norm_views = [torch.linalg.norm(view1, 'fro') ** 2, torch.linalg.norm(view2, 'fro') ** 2]

                view1_val = data_val[:, :i_shape[0]]
                view2_val = data_val[:, i_shape[0]:]
                norm_views_val = [torch.linalg.norm(view1_val, 'fro') ** 2, torch.linalg.norm(view2_val, 'fro') ** 2]

                train_loader_b1 = l_utils.get_dataloader(view1, view2, args.batchsize1, True)
                eval_loader = l_utils.get_dataloader(view1, view2, args.batchsize2, False)
                eval_loader_val = l_utils.get_dataloader(view1_val, view2_val, args.batchsize2, False)
                train_loader_b2 = l_utils.get_dataloader(view1, view2, args.batchsize2, True)

                # Batch iterator for the independence regularizer
                if args._lambda > 0:
                    corr_iter = iter(train_loader_b2)
                else:
                    corr_iter = []

                # Start training
                best_obj_val = float('inf')

            print("Start training ...")
            for itr in range(1, args.num_iters + 1):

                # Solve the U subproblem
                U = l_train.update_U(ae_model, eval_loader, args.o_dim, device)

                # Update network theta and eta for multiple epochs
                for _ in range(args.inner_epochs):

                    # Backprop to update
                    corr_iter = l_train.train(ae_model, mmcca1, mmcca2, U, norm_views, train_loader_b1, train_loader_b2,
                                              corr_iter, args, optimizer, device)

                    # Evaluate on the whole training set
                    match_err_tr, recons_err_tr, corr_tr = l_train.eval_train(ae_model, mmcca1, mmcca2, itr, U,
                                                                              norm_views, eval_loader, args, device)

                    # Evaluate on the whole validation set
                    match_err_val, recons_err_val, corr_val = l_train.eval_val(ae_model, mmcca1, mmcca2, itr,
                                                                               norm_views_val, eval_loader_val, args,
                                                                               device)
                    # Save the model
                    if (1 - args.beta) * match_err_val + args.beta * recons_err_val + args._lambda * corr_val < best_obj_val:
                        print('Saving Model')
                        best_ae_model = ae_model
                        best_obj_tr = (1 - args.beta) * match_err_tr + args.beta * recons_err_tr + args._lambda * corr_tr
                        best_obj_val = (1 - args.beta) * match_err_val + args.beta * recons_err_val + args._lambda * corr_val
                        best_obj = [best_obj_tr, best_obj_val]

            models_over_inits.append(best_ae_model)
            if val_loss > best_obj_val:
                best_model = best_ae_model
                best_losses = best_obj
                val_loss = best_obj_val

        models_over_all.append(models_over_inits)
        models.append(best_model)
        loss_l.append(best_losses)

    return models, loss_l, models_over_all


def model_loss(y_act, g_act, y_pred, i_shape, lambda_):
    mse_loss = nn.MSELoss()
    latent = y_pred[0]
    rec = y_pred[1]
    K = len(i_shape)
    mses1 = torch.zeros(K)
    mses2 = torch.zeros(K)
    for i in range(K):
        mses1[i] = mse_loss(latent[i], g_act)
        if len(rec[i]) == len(latent[i]):
            mses2[i] = mse_loss(rec[i], y_act[:, sum(i_shape[:i]): sum(i_shape[:i + 1])])
        else:
            for j in range(K):
                if i != j:
                    mses2[i] = mses2[i] + mse_loss(rec[i][j], y_act[:, sum(i_shape[:i]): sum(i_shape[:i + 1])])
            mses2[i] = mses2[i] / (K - 1)

    mse1 = torch.mean(mses1)
    mse2 = torch.mean(mses2)
    lambda_1 = (1 - lambda_)
    lambda_2 = lambda_
    total_loss = lambda_1 * mse1 + lambda_2 * mse2
    return total_loss, mses1, mses2


def update_G(data_tr, model, batch_size):
    with torch.no_grad():
        Z = model(data_tr)
        Y = sum(Z[0])

        # Ensure Y is zero mean
        Y = Y - torch.mean(Y, axis=0)
        G, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        G = G @ Vh
        G = np.sqrt(batch_size) * G

        #clear_output(wait=True)
        #plt.clf()
        #plt.hist(G.detach().cpu().numpy(), density=True, bins=100)  # density=False would make counts
        #plt.ylabel('Probability')
        #plt.xlabel('Data')
        #plt.draw()
        #plt.pause(0.0001)

    return G


def get_loss(A, G, model, i_shape, batch_size, lambda_):
    Z = model(A)
    loss, mv, xrec = model_loss(A, G, Z, i_shape, lambda_)
    loss = loss * G.size(dim=0) / batch_size
    mv = mv * G.size(dim=0) / batch_size
    xrec = xrec * G.size(dim=0) / batch_size
    return loss, mv, xrec


def get_loss2(A, model, i_shape, batch_size, lambda_):
    Z = model(A)
    Y = sum(Z[0]) / len(Z[0])
    loss, mses1, mses2 = model_loss(A, Y, Z, i_shape, lambda_)
    #mse1 = torch.mean(mses1) * (Y.size(dim=0) * Y.size(dim=1) / (torch.linalg.norm(Y, 'fro') ** 2))
    #mse2 = torch.mean(mses2) * Y.size(dim=0) / batch_size
    mses1 = mses1.to(device) * (Y.size(dim=0) * Y.size(dim=1) / (torch.linalg.norm(Y, 'fro') ** 2))
    mses2 = mses2.to(device) * Y.size(dim=0) / batch_size
    lambda_1 = (1 - lambda_)
    lambda_2 = lambda_
    loss = lambda_1 * torch.mean(mses1) + lambda_2 * torch.mean(mses2)
    return loss, mses1.to('cpu'), mses2.to('cpu')


def train(data_tr, data_val, data_te, model, model_optimizer, batch_size, epoch_num, i_shape, lambda_):
    # Initialize
    loss_epochs = np.zeros((epoch_num, 3))
    min_loss = 10000
    last = 0

    # Initialize G for fixed DNNs
    G = update_G(data_tr, model, batch_size)

    # Start training
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        torch.cuda.empty_cache()

        model.train()
        mydataloader = DataLoader(TensorDataset(data_tr), batch_size, shuffle=True)
        for (trs, idx) in mydataloader:
            model_optimizer.zero_grad()
            outputs = model(trs)
            g_local = G[idx, :]
            loss, _, _ = model_loss(trs, g_local, outputs, i_shape, lambda_)
            loss.backward()
            model_optimizer.step()
            del trs, outputs

        model.eval()
        torch.cuda.empty_cache()

        # Update G for fixed DNNs
        G = update_G(data_tr, model, batch_size)

        # Collect the losses
        with torch.no_grad():
            tr_loss, mv, xrec = get_loss(data_tr, G, model, i_shape, batch_size, lambda_)
            val_loss, mv_val, xrec_val = get_loss2(data_val, model, i_shape, batch_size, lambda_)
            te_loss, mv_te, xrec_te = get_loss2(data_te, model, i_shape, batch_size, lambda_)
            loss_epochs[epoch, 0] = tr_loss
            loss_epochs[epoch, 1] = val_loss
            loss_epochs[epoch, 2] = te_loss
            print('EPOCH : {}'.format(epoch))
            print('  Training TOTAL LOSS   : {}'.format(loss_epochs[epoch, 0]))
            print('  Training MAXVAR LOSS   : {}'.format(mv))
            print('  Training REC LOSS   : {}'.format(xrec))
            print('  Validation TOTAL LOSS   : {}'.format(loss_epochs[epoch, 1]))
            print('  Validation MAXVAR LOSS   : {}'.format(mv_val))
            print('  Validation REC LOSS   : {}'.format(xrec_val))
            print('  Testing TOTAL LOSS   : {}'.format(loss_epochs[epoch, 2]))
            print('  Testing MAXVAR LOSS   : {}'.format(mv_te))
            print('  Testing REC LOSS   : {}'.format(xrec_te))
            print("  val. loss is : {:0.4f} & the min. loss is : {:0.4f}".format(val_loss, min_loss))
            print("  AND since, val_loss < min_loss is {}".format(val_loss < min_loss))

            if (val_loss < min_loss) & (epoch - last > 10):
                best_model = model
                best_epoch = epoch
                min_loss = val_loss

            # if (torch.mean(mv) > 0.45) & (epoch - last > 10):
            #    last = epoch
            #    min_loss = 10000
            #    model.apply(weight_reset)
            #    model_optimizer.state = collections.defaultdict(dict)
            #    print("Reset")

    return best_model, loss_epochs, best_epoch


def linear_cca(H1, H2, outdim_size):
    """
    An implementation of linear CCA  (equiv to MAXVAR  sum(||H_i*A_i-G||^2_F)
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o1 = H1.shape[1]
    o2 = H2.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - mean1
    H2bar = H2 - mean2

    if o1 > 1:
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        V1 = V1[:, np.nonzero(D1 > 1e-9)]
        D1 = D1[np.nonzero(D1 > 1e-9)]
        V1 = np.squeeze(V1)
        D1 = np.squeeze(D1)
        V2 = V2[:, np.nonzero(D2 > 1e-9)]
        D2 = D2[np.nonzero(D2 > 1e-9)]
        V2 = np.squeeze(V2)
        D2 = np.squeeze(D2)

        SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.matmul(np.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]
    else:
        A = 1 / np.linalg.norm(H1bar)
        H1bar = H1bar * A
        if o2 > 1:
            SigmaHat22 = np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)
            [DS, V] = np.linalg.eigh(SigmaHat22)
            V = V[:, np.nonzero(DS > 1e-9)]
            DS = DS[np.nonzero(DS > 1e-9)]
            V = np.squeeze(V)
            DS = np.squeeze(DS)
            U = np.matmul(np.matmul(H2bar, V), np.diag(DS ** -0.5))
            z = np.matmul(U.T, H1bar)
            z = z / np.linalg.norm(z)
            B = np.matmul(np.matmul(V, np.diag(DS ** -0.5)), z)
            D = np.matmul(np.matmul(B.T, H2bar.T), H1bar)
        else:
            B = 1 / np.linalg.norm(H2bar)
            H2bar = H2bar * B
            D = np.matmul(H2bar.T, H1bar)

    return A, B, mean1, mean2, D


def my_corr(X1, X2, K=None, rcov1=0, rcov2=0):
    """
    FINDS THE CORRELATION OF TWO D DIMENSIONAL X1 AND X2 VECTORS USING THE CCA LOSS.
    ARGUMENTS:
        X1 : T x D1
        X2 : T x D2
        K  : OUTPUT DIMENSION TO BE CONSIDERED

    RETURNS:
        corr: CORRELATION OF THE TWO VECTORS.
    """
    rcov1 = 1e-4
    rcov2 = 1e-4
    if K == 1:
        x1 = standardize_1(X1)
        x2 = standardize_1(X2)
        corr = np.dot(x1.T, x2)
        D = corr
    else:
        N, d1 = X1.shape
        d2 = X2.shape[1]
        if K is None:
            K = min(d1, d2)
        m1 = np.mean(X1, 0)
        X1c = X1 - m1
        m2 = np.mean(X2, 0)
        X2c = X2 - m2
        S11 = np.dot(X1c.T, X1c) / (N - 1) + rcov1 * np.eye(d1)
        S22 = np.dot(X2c.T, X2c) / (N - 1) + rcov2 * np.eye(d2)
        S12 = np.dot(X1c.T, X2c) / (N - 1)
        D1, V1 = np.linalg.eigh(S11)
        D2, V2 = np.linalg.eigh(S22)
        idx1 = np.nonzero(D1 > 1e-10)
        D1 = D1[idx1]
        idx2 = np.nonzero(D2 > 1e-10)
        D2 = D2[idx2]
        V1 = np.squeeze(V1[:, idx1])
        if np.size(idx1) == 1:
            V1 = np.expand_dims(V1, axis=1)

        V2 = np.squeeze(V2[:, idx2])
        if np.size(idx2) == 1:
            V2 = np.expand_dims(V2, axis=1)

        K11 = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        K22 = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)
        T = np.dot(np.dot(K11, S12), K22)
        [U, D, V] = np.linalg.svd(T)
        D = D[0:K]
        corr = np.mean(D)

    return corr, D


def compute_linear_cca(X1, X2, XV1, XV2, XTe1, XTe2, o_dim):
    print("Compute Linear CCA")
    A, B, mean1, mean2, corr_tr = linear_cca(X1.to('cpu').numpy(), X2.to('cpu').numpy(), o_dim)
    S1 = np.dot(X1.to('cpu').numpy() - mean1, A)
    S2 = np.dot(X2.to('cpu').numpy() - mean2, B)
    XV1_c = XV1.to('cpu').numpy() - mean1
    XV2_c = XV2.to('cpu').numpy() - mean2
    x = np.dot(XV1_c, A)
    y = np.dot(XV2_c, B)
    corr_val = my_corr(x, y, o_dim)
    XTe1_c = XTe1.to('cpu').numpy() - mean1
    XTe2_c = XTe2.to('cpu').numpy() - mean2
    x = np.dot(XTe1_c, A)
    y = np.dot(XTe2_c, B)
    corr_te = my_corr(x, y, o_dim)

    linear_cca_corr = np.zeros((3))
    linear_cca_corr[0] = np.mean(corr_tr)
    linear_cca_corr[1] = corr_val[0]
    linear_cca_corr[2] = corr_te[0]

    return linear_cca_corr, A, B, 0.5 * (S1 + S2), 0.5 * (x + y)


def lcca_method(V1, V2, o_dim):
    """
    CUSTOM LCCA METHOD
    """
    x1 = V1[2].to('cpu').numpy();
    x2 = V2[2].to('cpu').numpy()
    x3 = V1[1].to('cpu').numpy();
    x4 = V2[1].to('cpu').numpy()
    x5 = V1[0].to('cpu').numpy();
    x6 = V2[0].to('cpu').numpy()

    corr_tr, D_tr = np.squeeze(my_corr(x5, x6, o_dim))
    corr_val, D_val = np.squeeze(my_corr(x3, x4, o_dim))
    corr_te, D_te = np.squeeze(my_corr(x1, x2, o_dim))
    corr_l = [corr_tr, corr_val, corr_te]
    D_l = [D_tr, D_val, D_te]
    print(f'LCCA is : {corr_l}')
    print(f'LCCA is : {D_l}')

    return corr_l, D_l


def compute_corr_coef_DGCCA(data_tr, data_val, data_te, model, o_dim):
    # Get the encodings
    torch.cuda.empty_cache()
    with torch.no_grad():
        model_cp = copy.deepcopy(model)
        model_cp = model_cp.to('cpu')
        data = [data_tr, data_val, data_te]
        new_data = []
        join_enc = []
        for k in range(3):
            temp = data[k].to('cpu')
            pred_out = model_cp(temp)
            new_data.append(pred_out[0])
            join_enc.append(sum(pred_out[0]) / len(pred_out[0]))
            del temp, pred_out

        W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
        W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

        # Compute the
        corr_l, D_l = lcca_method(W1, W2, o_dim)

    return corr_l[2], D_l, join_enc


def compute_corr_coef_DGCCA_over_init(data_tr, data_val, data_te, model, o_dim):
    num_of_inits = len(model)
    corr_te = []
    Ds = []
    join_encs_over_init = []
    for s in range(num_of_inits):
        # Get the encodings
        torch.cuda.empty_cache()
        with torch.no_grad():
            model_cp = copy.deepcopy(model[s])
            model_cp = model_cp.to('cpu')
            data = [data_tr, data_val, data_te]
            new_data = []
            join_encs = []
            for k in range(3):
                temp = data[k].to('cpu')
                pred_out = model_cp(temp)
                new_data.append(pred_out[0])
                join_encs.append(sum(pred_out[0]) / len(pred_out[0]))
                del temp, pred_out

            W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
            W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

            # Compute the correlation coefficients
            corr_l, D_l = lcca_method(W1, W2, o_dim)
            corr_te.append(corr_l[2])
            Ds.append(D_l)
            join_encs_over_init.append(join_encs)

    return np.array(np.mean(corr_te)), np.array(np.std(corr_te)), Ds, join_encs_over_init


def compute_corr_coef_over_lambda(data_tr, data_val, data_te, model, o_dim):
    num_of_models = len(model)
    join_encs = []
    corrs_te = []
    Ds = []
    for l in range(num_of_models):
        # Get the encodings
        torch.cuda.empty_cache()
        with torch.no_grad():
            model_cp = copy.deepcopy(model[l])
            model_cp = model_cp.to('cpu')
            data = [data_tr, data_val, data_te]
            new_data = []
            join_enc = []
            for k in range(3):
                temp = data[k].to('cpu')
                pred_out = model_cp(temp)
                new_data.append(pred_out[0])
                join_enc.append(sum(pred_out[0]) / len(pred_out[0]))
                del temp, pred_out

            W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
            W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

            # Compute the
            corr_l, D_l = lcca_method(W1, W2, o_dim)

        join_encs.append(join_enc)
        Ds.append(D_l)
        corrs_te.append(corr_l[2])

    return corrs_te, Ds, join_encs


def compute_corr_coef_over_lambda_and_init(data_tr, data_val, data_te, model, o_dim):
    num_of_lambdas = len(model)
    num_of_inits = len(model[0])
    corrs_te = []
    std_corrs_te = []
    Ds = []
    join_encs_over_lambdas = []
    for l in range(num_of_lambdas):
        # Get the encodings
        corrs_over_init = []
        Ds_over_init = []
        join_encs_over_init = []
        for s in range(num_of_inits):
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_cp = copy.deepcopy(model[l][s])
                model_cp = model_cp.to('cpu')
                data = [data_tr, data_val, data_te]
                new_data = []
                join_encs = []
                for k in range(3):
                    temp = data[k].to('cpu')
                    pred_out = model_cp(temp)
                    new_data.append(pred_out[0])
                    join_encs.append(sum(pred_out[0]) / len(pred_out[0]))
                    del temp, pred_out

                W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
                W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

                # Compute the correlation coefficients
                corr_l, D_l = lcca_method(W1, W2, o_dim)
                corrs_over_init.append(corr_l[2])
                Ds_over_init.append(D_l)
                join_encs_over_init.append(join_encs)

        Ds.append(Ds_over_init)
        corrs_te.append(np.mean(corrs_over_init))
        std_corrs_te.append(np.std(corrs_over_init))
        join_encs_over_lambdas.append(join_encs_over_init)

    return np.array(corrs_te), np.array(std_corrs_te), Ds, join_encs_over_lambdas


def compute_corr_coef_over_lambda_for_Lyu(data_tr, data_val, data_te, model, o_dim, i_shape):
    num_of_models = len(model)
    join_encs = []
    corrs_te = []
    Ds = []
    for l in range(num_of_models):
        # Get the encodings
        torch.cuda.empty_cache()
        with torch.no_grad():
            model_cp = copy.deepcopy(model[l])
            model_cp = model_cp.to('cpu')
            data = [data_tr, data_val, data_te]
            new_data = []
            join_enc = []
            for k in range(3):
                temp = data[k].to('cpu')
                view1 = temp[:, :i_shape[0]]
                view2 = temp[:, i_shape[0]:]
                pred_out = model_cp.encode([view1, view2])
                new_data.append(pred_out[0])
                join_enc.append(sum(pred_out[0]) / len(pred_out[0]))
                del temp, pred_out

            W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
            W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

            # Compute the
            corr_l, D_l = lcca_method(W1, W2, o_dim)

        join_encs.append(join_enc)
        Ds.append(D_l)
        corrs_te.append(corr_l[2])

    return corrs_te, Ds, join_encs


def compute_corr_coef_over_lambda_and_init_for_Lyu(data_tr, data_val, data_te, model, o_dim, i_shape):
    num_of_lambdas = len(model)
    num_of_inits = len(model[0])
    corrs_te = []
    std_corrs_te = []
    Ds = []
    join_encs_over_lambdas = []
    for l in range(num_of_lambdas):
        # Get the encodings
        corrs_over_init = []
        Ds_over_init = []
        join_encs_over_init = []
        for s in range(num_of_inits):
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_cp = copy.deepcopy(model[l][s])
                model_cp = model_cp.to('cpu')
                data = [data_tr, data_val, data_te]
                new_data = []
                join_encs = []
                for k in range(3):
                    temp = data[k].to('cpu')
                    view1 = temp[:, :i_shape[0]]
                    view2 = temp[:, i_shape[0]:]
                    pred_out = model_cp.encode([view1, view2])
                    new_data.append(pred_out[0])
                    join_encs.append(sum(pred_out[0]) / len(pred_out[0]))
                    del temp, pred_out

                W1 = [new_data[0][0], new_data[1][0], new_data[2][0]]
                W2 = [new_data[0][1], new_data[1][1], new_data[2][1]]

                # Compute the correlation coefficients
                corr_l, D_l = lcca_method(W1, W2, o_dim)
                corrs_over_init.append(corr_l[2])
                Ds_over_init.append(D_l)
                join_encs_over_init.append(join_encs)

        Ds.append(Ds_over_init)
        corrs_te.append(np.mean(corrs_over_init))
        std_corrs_te.append(np.std(corrs_over_init))
        join_encs_over_lambdas.append(join_encs_over_init)

    return np.array(corrs_te), np.array(std_corrs_te), Ds, join_encs_over_lambdas


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    score = 0
    for i in range(D):
        score += w[ind[0][i], ind[1][i]]
    score = score / y_pred.size
    return score


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1)


class CEModel(nn.Module):
    def __init__(self, i_shape, o_dim, dataset, p=0):
        super(CEModel, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = 32
        self.mid_shape2 = 32
        self.o_dim = o_dim
        self.p = 0
        self.drp = nn.Dropout(p=p)
        self.dataset = dataset

        self.enc_net0 = enc_model(self.i_shape[0], self.mid_shape, self.o_dim, self.p)
        self.enc_net1 = enc_model(self.i_shape[1], self.mid_shape, self.o_dim, self.p)
        self.dec_net0 = dec_model(self.i_shape[0], self.mid_shape, self.mid_shape2, self.o_dim, self.p)
        self.dec_net1 = dec_model(self.i_shape[1], self.mid_shape, self.mid_shape2, self.o_dim, self.p)

    def forward(self, x):
        x1 = []
        y = []
        c = []
        for i in range(len(self.i_shape)):
            c.append(x[:, sum(self.i_shape[:i]): sum(self.i_shape[:i + 1])])

        if self.dataset == 1:
            x1.append(self.enc_net0(c[0]))
            x1.append(self.enc_net1(c[1]))
            x1.append(self.enc_net2(c[2]))
            r = []
            for i in range(len(self.i_shape)):
                r.append(self.dec_net0(x1[i]))
            y.append(r)
            r = []
            for i in range(len(self.i_shape)):
                r.append(self.dec_net1(x1[i]))
            y.append(r)
            r = []
            for i in range(len(self.i_shape)):
                r.append(self.dec_net2(x1[i]))
            y.append(r)
        else:
            x1.append(self.enc_net0(c[0]))
            x1.append(self.enc_net1(c[1]))
            y.append(self.dec_net0(x1[1]))
            y.append(self.dec_net1(x1[0]))

        z = [x1, y]
        return z

    def get_o_dim(self):
        return self.o_dim


class DCCAEModel(nn.Module):
    def __init__(self, i_shape, o_dim, dataset, p=0):
        super(DCCAEModel, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = 32
        self.mid_shape2 = 32
        self.o_dim = o_dim
        self.p = 0
        self.drp = nn.Dropout(p=p)
        self.dataset = dataset
        if dataset == 1:
            self.enc_net0 = enc_model(self.i_shape[0], self.mid_shape, self.o_dim, self.p)
            self.enc_net1 = enc_model(self.i_shape[1], self.mid_shape, self.o_dim, self.p)
            self.enc_net2 = enc_model(self.i_shape[2], self.mid_shape, self.o_dim, self.p)
            self.dec_net0 = dec_model(self.i_shape[0], self.mid_shape, self.mid_shape2, self.o_dim, self.p)
            self.dec_net1 = dec_model(self.i_shape[1], self.mid_shape, self.mid_shape2, self.o_dim, self.p)
            self.dec_net2 = dec_model(self.i_shape[2], self.mid_shape, self.mid_shape2, self.o_dim, self.p)
        else:
            self.enc_net0 = enc_model(self.i_shape[0], self.mid_shape, self.o_dim, self.p)
            self.enc_net1 = enc_model(self.i_shape[1], self.mid_shape, self.o_dim, self.p)
            self.dec_net0 = dec_model(self.i_shape[0], self.mid_shape, self.mid_shape2, self.o_dim, self.p)
            self.dec_net1 = dec_model(self.i_shape[1], self.mid_shape, self.mid_shape2, self.o_dim, self.p)

    def forward(self, x):
        x1 = []
        y = []
        c = []
        for i in range(len(self.i_shape)):
            c.append(x[:, sum(self.i_shape[:i]): sum(self.i_shape[:i + 1])])

        if self.dataset == 1:
            x1.append(self.enc_net0(c[0]))
            x1.append(self.enc_net1(c[1]))
            x1.append(self.enc_net2(c[2]))
            y.append(self.dec_net0(x1[0]))
            y.append(self.dec_net1(x1[1]))
            y.append(self.dec_net2(x1[2]))
        else:
            x1.append(self.enc_net0(c[0]))
            x1.append(self.enc_net1(c[1]))
            y.append(self.dec_net0(x1[0]))
            y.append(self.dec_net1(x1[1]))

        z = [x1, y]
        return z

    def get_o_dim(self):
        return self.o_dim


class GenModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GenModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_shape = 32
        self.mid_shape2 = 32
        self.dec_net = dec_model(self.out_dim, self.mid_shape, self.mid_shape2, self.in_dim, 0)

    def forward(self, x):
        z = self.dec_net(x)
        return z


class enc_model(nn.Module):
    def __init__(self, i_shape, mid_shape, o_dim, p):
        super(enc_model, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = mid_shape
        self.o_dim = o_dim
        self.act = nn.ReLU()
        self.drp = nn.Dropout(p=p)
        self.one = nn.Linear(self.i_shape, self.mid_shape)
        self.sec = nn.Linear(self.mid_shape, self.mid_shape)
        self.thr = nn.Linear(self.mid_shape, self.o_dim)

    def forward(self, x):
        x = self.sec(self.drp(self.act(self.one(x))))
        x = self.thr(self.drp(self.act(x)))
        return x


class dec_model(nn.Module):
    def __init__(self, i_shape, mid_shape, mid_shape2, o_dim, p):
        super(dec_model, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = mid_shape
        self.mid_shape2 = mid_shape2
        self.o_dim = o_dim
        self.act = nn.ReLU()
        self.drp = nn.Dropout(p=p)
        self.de1 = nn.Linear(self.o_dim, self.mid_shape)
        self.de2 = nn.Linear(self.mid_shape, self.mid_shape2)
        self.de3 = nn.Linear(self.mid_shape2, self.i_shape)

    def forward(self, y):
        y = self.drp(self.act(self.de1(y)))
        y = self.de3(self.drp(self.act(self.de2(y))))
        return y


class EmptyObject:
    pass


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
