import logging
import time
import typing
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, save_checkpoint
from .config import TrainingConfig
from .models import ModelType, setup_model
from .optimize import OptimizerType
from .utils import clip_grad_value_, duration_loss, mle_loss

try:
    from apex import amp
except ImportError:
    # apex not available (no fp16)
    amp = None


_LOGGER = logging.getLogger("glow_tts_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
    global_step: int = 1,
):
    """Run training for the specified number of epochs"""
    torch.manual_seed(config.seed)

    model, optimizer = setup_model(config, model=model, optimizer=optimizer)
    assert optimizer is not None

    amp_run = False
    if config.fp16_run and amp:
        # Use AMP for FP16 run
        amp_run = True
        model, optimizer._optim = amp.initialize(
            model, optimizer._optim, opt_level="O1"  # pylint: disable=protected-access
        )

    assert model is not None

    # Begin training
    for epoch in range(1, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s)", epoch, config.epochs, global_step
        )
        epoch_start_time = time.perf_counter()
        global_step = train_step(
            global_step, epoch, model, optimizer, train_loader, amp_run
        )

        # TODO: Do evalulation
        # evaluate(
        #     rank, epoch, hps, model, optimizer, val_loader, logger, writer_eval
        # )

        # Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
        _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
        save_checkpoint(
            Checkpoint(
                model=model,
                optimizer=optimizer,
                learning_rate=optimizer.cur_lr,
                global_step=global_step,
                version=config.version,
            ),
            checkpoint_path,
        )
        _LOGGER.info("Saved checkpoint to %s", checkpoint_path)

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "Epoch %s complete in %s second(s) (global step=%s)",
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    global_step: int,
    epoch: int,
    model: ModelType,
    optimizer: OptimizerType,
    train_loader: DataLoader,
    amp_run: bool,
):
    # train_loader.sampler.set_epoch(epoch)
    steps_per_epoch = len(train_loader)
    all_loss_g: typing.List[float] = []

    model.train()
    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = (x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True))
        y, y_lengths = (y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True))

        # Train model
        optimizer.zero_grad()

        (
            (z, z_m, z_logs, logdet, z_mask),
            (_x_m, _x_logs, _x_mask),
            (_attn, logw, logw_),
        ) = model(x, x_lengths, y, y_lengths, gen=False)

        # Compute loss
        l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = duration_loss(logw, logw_, x_lengths)

        # TODO: Weighted loss
        # loss_gs = [l_mle, l_length]
        loss_g = l_mle + l_length

        all_loss_g.append(loss_g.item())

        if amp_run:
            # Need to handle loss specially for FP16
            # pylint: disable=protected-access
            with amp.scale_loss(loss_g, optimizer._optim) as scaled_loss:
                scaled_loss.backward()
            clip_grad_value_(amp.master_params(optimizer._optim), 5)
        else:
            # FP32 run
            loss_g.backward()
            clip_grad_value_(model.parameters(), 5)

        optimizer.step()

        # TODO: Print alignment
        # if batch_idx % hps.train.log_interval == 0:
        #     (y_gen, *_), *_ = model.module(x[:1], x_lengths[:1], gen=True)
        #     logger.info(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch,
        #             batch_idx * len(x),
        #             len(train_loader.dataset),
        #             100.0 * batch_idx / len(train_loader),
        #             loss_g.item(),
        #         )
        #     )
        #     logger.info(
        #         [x.item() for x in loss_gs] + [global_step, optimizer.get_lr()]
        #     )

        #     scalar_dict = {
        #         "loss/g/total": loss_g,
        #         "learning_rate": optimizer.get_lr(),
        #         "grad_norm": grad_norm,
        #     }
        #     scalar_dict.update(
        #         {"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)}
        #     )
        #     utils.summarize(
        #         writer=writer,
        #         global_step=global_step,
        #         images={
        #             "y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
        #             "y_gen": utils.plot_spectrogram_to_numpy(
        #                 y_gen[0].data.cpu().numpy()
        #             ),
        #             "attn": utils.plot_alignment_to_numpy(
        #                 attn[0, 0].data.cpu().numpy()
        #             ),
        #         },
        #         scalars=scalar_dict,
        #     )

        _LOGGER.debug(
            "Loss: %s (step=%s/%s)", loss_g.item(), batch_idx + 1, steps_per_epoch
        )
        global_step += 1

    if all_loss_g:
        avg_loss_g = sum(all_loss_g) / len(all_loss_g)
        _LOGGER.info(
            "Avg. Loss for epoch %s: %s (global step=%s)",
            epoch,
            avg_loss_g,
            global_step,
        )

    return global_step


# def evaluate(rank, epoch, hps, model, optimizer, val_loader, logger, writer_eval):
#     if rank == 0:
#         global global_step
#         model.eval()
#         losses_tot = []
#         with torch.no_grad():
#             for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
#                 x, x_lengths = (
#                     x.cuda(rank, non_blocking=True),
#                     x_lengths.cuda(rank, non_blocking=True),
#                 )
#                 y, y_lengths = (
#                     y.cuda(rank, non_blocking=True),
#                     y_lengths.cuda(rank, non_blocking=True),
#                 )

#                 (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (
#                     attn,
#                     logw,
#                     logw_,
#                 ) = model(x, x_lengths, y, y_lengths, gen=False)
#                 l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
#                 l_length = commons.duration_loss(logw, logw_, x_lengths)

#                 loss_gs = [l_mle, l_length]
#                 loss_g = sum(loss_gs)

#                 if batch_idx == 0:
#                     losses_tot = loss_gs
#                 else:
#                     losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

#                 if batch_idx % hps.train.log_interval == 0:
#                     logger.info(
#                         "Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                             epoch,
#                             batch_idx * len(x),
#                             len(val_loader.dataset),
#                             100.0 * batch_idx / len(val_loader),
#                             loss_g.item(),
#                         )
#                     )
#                     logger.info([x.item() for x in loss_gs])

#         losses_tot = [x / len(val_loader) for x in losses_tot]
#         loss_tot = sum(losses_tot)
#         scalar_dict = {"loss/g/total": loss_tot}
#         scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
#         utils.summarize(
#             writer=writer_eval, global_step=global_step, scalars=scalar_dict
#         )
#         logger.info("====> Epoch: {}".format(epoch))
