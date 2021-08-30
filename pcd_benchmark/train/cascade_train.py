from utils.utils import *
from models.full_model import FullModelCascade
from models.cascade_model import *
import torch.optim as optim


def train(args):
    # setup
    vis, log_model, log_train, log_path = log_setup(args)
    train_curve, val_curves = vis_curve_setup()
    best_val_losses, best_val_epochs = best_loss_setup()
    train_loss_meter, val_loss_meters = loss_average_meter_setup()
    dataset, dataset_test, dataloader, dataloader_test = data_setup(args, log_train)
    seed_setup(args, log_train)

    # model
    net_g = CascadeGenerator(step_ratio=args.step_ratio)
    net_g = torch.nn.DataParallel(FullModelCascade(net_g))
    net_d = torch.nn.DataParallel(PatchDection(args.num_points, divide_ratio=2))
    net_g.cuda()
    net_d.cuda()
    net_g.apply(weights_init_cascade)
    net_d.apply(weights_init_cascade)
    log_model.log_string(str(net_g.module.model) + '\n', stdout=False)
    log_model.log_string(str(net_d) + '\n', stdout=False)

    # optim
    lr_g = args.lr
    lr_d = lr_g / 2
    lr_clip = args.lr_clip
    optimizer_g = optim.Adam(net_g.module.model.parameters(), lr=lr_d, weight_decay=0.00001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr_g, weight_decay=0.00001, betas=(0.9, 0.999))
    decay_rate = 0.7
    global_step = 0
    lr_decay_step = int(len(dataset) / args.batch_size * 40)
    load_model(args, net_g, optimizer_g, log_train, net_d=net_d, optimizer_d=optimizer_d)

    for epoch in range(args.resume_epoch, args.nepoch):
        train_loss_meter.reset()
        net_g.module.model.train()
        net_d.train()

        if global_step % lr_decay_step == 0 and global_step > 0:
            lr_g_updated = lr_g * decay_rate ** (global_step // lr_decay_step)
            lr_d_updated = lr_d * decay_rate ** (global_step // lr_decay_step)
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = max(lr_g_updated, lr_clip)
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = max(lr_d_updated, lr_clip)

        if global_step * 2 < 10000:
            alpha = 0.01
        elif global_step * 2 < 20000:
            alpha = 0.1
        elif global_step * 2 < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        for i, data in enumerate(dataloader, 0):
            global_step += 1
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            if args.use_mean_feature == 0:
                _, inputs, gt = data
                mean_feature = None
            else:
                _, inputs, gt, mean_feature = data
                mean_feature = mean_feature.float().cuda()

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            if args.loss == 'EMD':
                output1, output2, emd1, emd2, cd1_p, cd2_p, cd1_t, cd2_t = net_g(inputs, gt.contiguous(), mean_feature,
                                                                                 0.005, 50, True, False)
                loss_rec = emd1.mean() + emd2.mean() * alpha
                train_loss_meter.update(emd2.mean().item())
            else:
                output1, output2, emd1, emd2, cd1_p, cd2_p, cd1_t, cd2_t = net_g(inputs, gt.contiguous(), mean_feature,
                                                                                 0.005, 50, False, True)
                loss_rec = cd1_p.mean() + cd2_p.mean() * alpha
                train_loss_meter.update(cd2_p.mean().item())

            set_requires_grad(net_d, False)
            d_fake = net_d(output2[:, 0:2048, :])
            errG_loss_batch = torch.mean((d_fake - 1) ** 2)
            total_gen_loss_batch = errG_loss_batch + loss_rec * 200
            total_gen_loss_batch.backward(retain_graph=True)
            optimizer_g.step()

            set_requires_grad(net_d, True)
            d_real = net_d(gt[:, 0:2048, :])
            d_loss_fake = torch.mean(d_fake ** 2)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
            total_dis_loss_batch = errD_loss_batch
            total_dis_loss_batch.backward()
            optimizer_d.step()

            if i % 600 == 0:
                log_train.log_string(
                    args.log_env + ' train [%d: %d/%d] step: %d G_loss: %f D_loss: %f emd1: %f emd2: %f cd1_p: %f cd2_p: %f cd1_t: %f cd2_t: %f' % (
                    epoch, i, len(dataset) / args.batch_size, global_step,
                    total_gen_loss_batch.mean().item(),
                    total_dis_loss_batch.mean().item(), emd1.mean().item(),
                    emd2.mean().item(), cd1_p.mean().item(), cd2_p.mean().item(),
                    cd1_t.mean().item(), cd2_t.mean().item()))

        train_curve.append(train_loss_meter.avg)
        save_model('%s/network.pth' % log_path, net_g, optimizer_g, net_d=net_d, optimizer_d=optimizer_d)
        log_train.log_string("saving net...")

        # VALIDATION
        if epoch % 5 == 0 or epoch == args.nepoch - 1:
            best_val_losses, best_val_epochs = val(args, net_g, epoch, dataloader_test, log_train,
                                                   best_val_losses, best_val_epochs, val_loss_meters)
            if best_val_epochs["best_emd_epoch"] == epoch:
                save_model('%s/best_emd_network.pth' % log_path, net_g, optimizer_g, net_d=net_d,
                           optimizer_d=optimizer_d)
                log_train.log_string('saving best emd net...')
            if best_val_epochs["best_cd_p_epoch"] == epoch:
                save_model('%s/best_cd_p_network.pth' % log_path, net_g, optimizer_g, net_d=net_d,
                           optimizer_d=optimizer_d)
                log_train.log_string('saving best cd_p net...')
            if best_val_epochs["best_cd_t_epoch"] == epoch:
                save_model('%s/best_cd_t_network.pth' % log_path, net_g, optimizer_g, net_d=net_d,
                           optimizer_d=optimizer_d)
                log_train.log_string('saving best cd_t net...')

        val_curves["val_curve_emd"].append(val_loss_meters["val_loss_emd"].avg)
        val_curves["val_curve_cd_p"].append(val_loss_meters["val_loss_cd_p"].avg)
        val_curves["val_curve_cd_t"].append(val_loss_meters["val_loss_cd_t"].avg)
        vis_curve_plot(vis, args.log_env, train_curve, val_curves)
        epoch_log(args, log_model, train_loss_meter, val_loss_meters, best_val_losses, best_val_epochs)

    log_model.close()
    log_train.close()
