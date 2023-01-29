import abc
from collections import OrderedDict
import gtimer as gt
from enlighten.agents.common.replay_buffer import EnvReplayBuffer
from enlighten.agents.common.data_collector import MdpPathCollector
from enlighten.agents.common.other import create_stats_ordered_dict
from enlighten.utils.path import * 
import torch
import os
import wandb

class TorchBatchRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            exploration_data_collector: MdpPathCollector,
            replay_buffer: EnvReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_expl_steps_before_training,
            num_trains_per_train_loop,
            checkpoint_folder_name,
            save_every_epochs,
            log_to_wandb,
            num_train_loops_per_epoch=1,
            start_epoch=0
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.expl_data_collector = exploration_data_collector
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_expl_steps_before_training = num_expl_steps_before_training
        self._start_epoch = start_epoch
        self.checkpoint_folder_name = checkpoint_folder_name
        self.save_every_epochs = save_every_epochs
        self.log_to_wandb = log_to_wandb

    def train(self):
        # collect k steps before training
        if self.num_expl_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                max_path_length=self.max_path_length,
                num_steps=self.num_expl_steps_before_training
            )
            
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        
        print("======> Initial collection done.")

        # train for n epochs
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print("========> Start epoch %d"%(self.epoch))
            #self._begin_epoch(self.epoch)
            self._train_one_epoch()
            self._end_epoch(self.epoch)

    def reset_stats_before_epoch(self):
        self.policy_loss = []
        self.q1_loss = []
        self.q2_loss = []
        self.alpha_loss = []
        self.alpha = []
    
    def update_stats_per_batch(self, current_update_stats):
        self.policy_loss.append(current_update_stats["policy_loss"])
        self.q1_loss.append(current_update_stats["q1_loss"])
        self.q2_loss.append(current_update_stats["q2_loss"])
        self.alpha_loss.append(current_update_stats["alpha_loss"])
        self.alpha.append(current_update_stats["alpha"])

    def _train_one_epoch(self):
        # reset stats of current epoch
        self.reset_stats_before_epoch()
        
        # loop n times
        # self.num_train_loops_per_epoch = 1
        for i in range(self.num_train_loops_per_epoch):
            # collect m steps
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                max_path_length=self.max_path_length,
                num_steps=self.num_expl_steps_per_train_loop
            )
            gt.stamp('exploration sampling', unique=False)
            print("=============> Collection done.")

            # add to replay buffer
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            # train for b batches
            self.training_mode(True)
            for j in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                stats = self.trainer.train(train_data)
                self.update_stats_per_batch(current_update_stats=stats)
                print("=============> Train batch %d done"%(j))
            gt.stamp('training', unique=False)
            self.training_mode(False)

            #print("Train update %d done."%(i))

    # compute stats at the end of each epoch
    def update_stats_after_epoch(self, logs):

        logs.update(create_stats_ordered_dict(
            "Train/policy_loss",
            self.policy_loss,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))

        logs.update(create_stats_ordered_dict(
            "Train/q1_loss",
            self.q1_loss,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))

        logs.update(create_stats_ordered_dict(
            "Train/q2_loss",
            self.q2_loss,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))

        logs.update(create_stats_ordered_dict(
            "Train/alpha_loss",
            self.alpha_loss,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))

        logs.update(create_stats_ordered_dict(
            "Train/alpha",
            self.alpha,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))

    # save checkpoint
    def save_checkpoint(self, checkpoint_number):
        # checkpoint folder
        folder_path = os.path.join(checkpoints_path, self.checkpoint_folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # get model snapshot
        checkpoint = self.trainer.get_snapshot()

        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        torch.save(checkpoint, checkpoint_path)

        print(f"========> Checkpoint {checkpoint_number} saved.")
    

    # log stats and save checkpoints at the end of each epoch
    def _end_epoch(self, epoch):
        # save checkpoints
        # do not save at epoch 0
        # checkpoint index starts from 0
        if (epoch+1) % self.save_every_epochs == 0:
            self.save_checkpoint(checkpoint_number = int((epoch+1) // self.save_every_epochs) - 1)
            
        gt.stamp('saving')

        # log stats
        self._log_stats(epoch)

        # reset buffers and train return 
        self.expl_data_collector.end_epoch(epoch)
        # self.replay_buffer.end_epoch(epoch)
        # self.trainer.end_epoch(epoch)


    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        

    # switch network mode to train or evaluation
    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
    
    # def _get_snapshot(self):
    #     snapshot = {}
    #     for k, v in self.trainer.get_snapshot().items():
    #         snapshot['trainer/' + k] = v
    #     # for k, v in self.expl_data_collector.get_snapshot().items():
    #     #     snapshot['exploration/' + k] = v
    #     # for k, v in self.replay_buffer.get_snapshot().items():
    #     #     snapshot['replay_buffer/' + k] = v
    #     return snapshot
    
    
    def _log_stats(self, epoch, print_logs=True):
        gt.stamp('logging')

        logs = dict()
        
        """
        Exploration (count after each epoch ends)
        """
        logs.update(self.expl_data_collector.get_diagnostics())
        # add return of exploration trajectories
        logs.update(create_stats_ordered_dict(
            "Train/return",
            self.expl_data_collector._epoch_returns,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))
        

        """
        Trainer (get mean/std/min/max of each epoch)
        """
        # add loss stats
        self.update_stats_after_epoch(logs)

        # add the number of train updates
        logs.update(self.trainer.get_diagnostics())

        """
        Epoch index
        """
        logs["Epoch"] = epoch

        """
        Replay Buffer (count after each epoch ends)
        """
        logs.update(self.replay_buffer.get_diagnostics())

        # print logs
        if print_logs:
            print("========> Epoch {} finished".format(epoch))
            print('=' * 80)
            for k, v in logs.items():
                print(f'{k}: {v}')
            print('=' * 80)
        
        # log to wandb at every epoch
        if self.log_to_wandb:
            wandb.log(logs, step=epoch)