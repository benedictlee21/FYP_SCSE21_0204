import torch.optim as optim
from models.topnet_model import TopNet
from models.full_model import FullModelTopNet
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
    net = TopNet(args.num_points)
    net.cuda()
    net = torch.nn.DataParallel(FullModelTopNet(net))
    log_model.log_string(str(net.module.model) + '\n', stdout=False)

    # optim
    lr = args.lr
    optimizer = optim.Adagrad(net.parameters(), lr, initial_accumulator_value=1e-13)
    global_step = 0
    load_model(args, net, optimizer, log_train)

    for epoch in range(args.resume_epoch, args.nepoch):
        #TRAIN MODE
        train_loss_meter.reset()
        net.module.model.train()

        for i, data in enumerate(dataloader, 0):
            global_step += 1
            optimizer.zero_grad()
            _, inputs, gt = data

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            if args.loss == 'EMD':
                output, emd, cd_p, cd_t = net(inputs, gt.contiguous(), 0.005, 50, True, False)
                net_loss = emd.mean()
                train_loss_meter.update(emd.mean().item())
            else:
                output, emd, cd_p, cd_t = net(inputs, gt.contiguous(), 0.005, 50, False, True)
                net_loss = cd_p.mean()
                train_loss_meter.update(cd_p.mean().item())

            net_loss.backward()
            optimizer.step()

            if i % 600 == 0:
                log_train.log_string(args.log_env + ' train [%d: %d/%d] step: %d emd: %f cd_p: %f cd_t: %f ' % (epoch,
                                i, len(dataset) / args.batch_size, global_step, emd.mean().item(), cd_p.mean().item(),
                                cd_t.mean().item()))

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


