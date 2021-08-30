import torch.optim as optim
from models.pcn_model import PCN
from models.full_model import FullModelPCN
from utils.utils import *


def train(args):
    # setup
    vis, log_model, log_train, log_path = log_setup(args)
    train_curve, val_curves = vis_curve_setup()
    best_val_losses, best_val_epochs = best_loss_setup()
    train_loss_meter, val_loss_meters = loss_average_meter_setup()
    dataset, dataset_test, dataloader, dataloader_test = data_setup(args, log_train)
    seed_setup(args, log_train)

    # model
    net = PCN(num_coarse=1024, num_fine=args.num_points).cuda()
    net = torch.nn.DataParallel(FullModelPCN(net))
    log_model.log_string(str(net.module.model) + '\n', stdout=False)

    # optim
    lrate = args.lr
    lr_clip = args.lr_clip
    optimizer = optim.Adam(net.module.model.parameters(), lr=lrate)
    global_step = 0
    load_model(args, net, optimizer, log_train)

    for epoch in range(args.resume_epoch, args.nepoch):
        train_loss_meter.reset()
        net.module.model.train()

        if epoch > 0 and epoch % 40 == 0:
            lrate = max(lrate * 0.7, lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate

        if epoch < 5:
            alpha = 0.01
        elif epoch < 15:
            alpha = 0.1
        elif epoch < 30:
            alpha = 0.5
        else:
            alpha = 1.0

        for i, data in enumerate(dataloader, 0):
            global_step += 1
            optimizer.zero_grad()
            _, inputs, gt = data
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            if args.loss == 'EMD':
                output1, output2, emd1, emd2, cd_p1, cd_p2, cd_t1, cd_t2 = net(inputs, gt.contiguous(), 0.005, 50, True,
                                                                               False)
                loss_net = emd1.mean() + emd2.mean() * alpha
                train_loss_meter.update(emd2.mean().item())
            else:
                output1, output2, emd1, emd2, cd_p1, cd_p2, cd_t1, cd_t2 = net(inputs, gt.contiguous(), 0.005, 50,
                                                                               False, True)
                loss_net = cd_p1.mean() + cd_p2.mean() * alpha
                train_loss_meter.update(cd_p2.mean().item())

            loss_net.backward()
            optimizer.step()

            if i % 600 == 0:
                log_train.log_string(
                    args.log_env + ' train [%d: %d/%d]  emd1: %f emd2: %f cd_p1: %f cd_p2: %f cd_t1: %f cd_t2: %f' % (
                    epoch, i, len(dataset) / args.batch_size, emd1.mean().item(), emd2.mean().item(),
                    cd_p1.mean().item(), cd_p2.mean().item(), cd_t1.mean().item(), cd_t2.mean().item()))

        train_curve.append(train_loss_meter.avg)
        save_model('%s/network.pth' % log_path, net, optimizer)
        log_train.log_string("saving net...")

        # VALIDATION
        if epoch % 5 == 0 or epoch == args.nepoch - 1:
            best_val_losses, best_val_epochs = val(args, net, epoch, dataloader_test, log_train,
                                                   best_val_losses, best_val_epochs, val_loss_meters)
            if best_val_epochs["best_emd_epoch"] == epoch:
                save_model('%s/best_emd_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best emd net...')
            if best_val_epochs["best_cd_p_epoch"] == epoch:
                save_model('%s/best_cd_p_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best cd_p net...')
            if best_val_epochs["best_cd_t_epoch"] == epoch:
                save_model('%s/best_cd_t_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best cd_t net...')

        val_curves["val_curve_emd"].append(val_loss_meters["val_loss_emd"].avg)
        val_curves["val_curve_cd_p"].append(val_loss_meters["val_loss_cd_p"].avg)
        val_curves["val_curve_cd_t"].append(val_loss_meters["val_loss_cd_t"].avg)
        vis_curve_plot(vis, args.log_env, train_curve, val_curves)
        epoch_log(args, log_model, train_loss_meter, val_loss_meters, best_val_losses, best_val_epochs)

    log_model.close()
    log_train.close()
