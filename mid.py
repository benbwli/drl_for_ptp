import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation
import matplotlib.pyplot as plt
from evaluation.visualization.visualization import visualize_prediction

class MID():
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar:

                    self.optimizer.zero_grad()
                    train_loss = self.model.get_loss(batch, node_type)
                    pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    train_loss.backward()
                    self.optimizer.step()

            self.train_dataset.augment = False
            if epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                max_hl = self.hyperparams['maximum_history_length']


                for i, scene in enumerate(self.eval_scenes):
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t,t+10)
                        batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                       pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                       min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                                       max_ft=12, hyperparams=self.hyperparams)
                        if batch is None:
                            continue
                        test_batch = batch[0]
                        nodes = batch[1]
                        timesteps_o = batch[2]
                        traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True) # B * 20 * 12 * 2

                        predictions = traj_pred
                        predictions_dict = {}
                        for i, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

                        batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                               scene.dt,
                                                                               max_hl=max_hl,
                                                                               ph=ph,
                                                                               node_type_enum=self.eval_env.NodeType,
                                                                               kde=False,
                                                                               map=None,
                                                                               best_of=True,
                                                                               prune_ph_to_future=True)

                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)

                if self.config.dataset == "eth":
                    ade = ade/0.6
                    fde = fde/0.6
                elif self.config.dataset == "sdd":
                    ade = ade * 50
                    fde = fde * 50


                print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
                self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

                # Saving model
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                 }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

                self.model.train()


    def eval(self):
        epoch = self.config.eval_at

        # --- Deterministic seeding for reproducible evaluation ---
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']

        # Options: "TRUE_CP", "VARIANCE", "NONE"
        POST_PROCESSING_METHOD = "VARIANCE"
        CP_GUIDED_DIFFUSION = False  # Scale diffusion noise based on CP radius
        prev_predictions = {}
        calibration_scores = [0.1]
        cp_radius_history = []  # Track CP radius over time for line graph

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                # Compute CP-guided noise scale for diffusion
                cp_noise_scale = None
                if CP_GUIDED_DIFFUSION and len(calibration_scores) > 1:
                    cp_radius = np.percentile(calibration_scores, 90)
                    cp_noise_scale = float(np.clip(1.0 / (1.0 + cp_radius), 0.3, 1.0))
                
                traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20, bestof=True, cp_noise_scale=cp_noise_scale) # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                cp_radius_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                        cp_radius_dict[ts] = dict()
                    predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

                # --- CP WEIGHTING ALGORITHM & CALIBRATION SCORE TRACKING ---
                if POST_PROCESSING_METHOD in ["TRUE_CP", "VARIANCE"]:
                    for ts in predictions_dict.keys():
                        for node in predictions_dict[ts].keys():
                            new_traj = predictions_dict[ts][node] # shape: (1, 20, 12, 2)
                            
                            # Default weight fallback
                            weight_new = 1.0
                            
                            if node in prev_predictions:
                                last_ts = max(prev_predictions[node].keys())
                                time_diff = ts - last_ts
                                
                                # If there is a valid temporal overlap
                                if time_diff < 12 and time_diff > 0:
                                    old_traj = prev_predictions[node][last_ts]
                                    aligned_old = old_traj[:, :, time_diff:, :]
                                    
                                    # Always calculate the CP error on the previous trajectory to update calibration_scores
                                    ground_truth = node.get(np.array([last_ts + 1, ts]), {'position': ['x', 'y']})
                                    
                                    # Handle potential missing frames by masking NaNs
                                    valid_mask = ~np.isnan(ground_truth.sum(axis=1))
                                    
                                    if valid_mask.any():
                                        valid_gt = ground_truth[valid_mask]  # shape: (valid_frames, 2)
                                        # old_traj shape: (1, 20, 12, 2) — first time_diff steps overlap with GT
                                        valid_old = old_traj[0, :, 0:time_diff, :][:, valid_mask, :]  # (20, valid_frames, 2)
                                        
                                        # Calculate ADE of the Best Path (best of 20)
                                        error = np.linalg.norm(valid_old - valid_gt, axis=-1)  # (20, valid_frames)
                                        best_of_error = error.mean(axis=1).min()  # scalar
                                        calibration_scores.append(best_of_error)
                                        
                                    if POST_PROCESSING_METHOD == "TRUE_CP":
                                        # Compute CP radius (90th percentile)
                                        cp_radius = np.percentile(calibration_scores, 90)
                                        weight_new = np.clip(1.0 / (1.0 + cp_radius), 0.1, 0.9)
                                        
                                    elif POST_PROCESSING_METHOD == "VARIANCE":
                                        variance = np.var(new_traj, axis=1).mean()
                                        weight_new = np.clip(1.0 / (1.0 + variance), 0.1, 0.9)
                                        
                                    # Pad end of old trajectory to maintain prediction horizon of 12
                                    pad_len = 12 - aligned_old.shape[2]
                                    if pad_len > 0:
                                        padding = np.repeat(aligned_old[:, :, -1:, :], pad_len, axis=2)
                                        aligned_old = np.concatenate([aligned_old, padding], axis=2)
                                        
                                    weight_old = 1.0 - weight_new
                                    predictions_dict[ts][node] = (new_traj * weight_new) + (aligned_old * weight_old)
                                    
                            # Compute CP radius (90th percentile) for safety shield plotting
                            cp_radius = np.percentile(calibration_scores, 90)
                            cp_radius_dict[ts][node] = cp_radius
                            cp_radius_history.append((ts, cp_radius))

                            # Save blended/new prediction back into memory
                            if node not in prev_predictions:
                                prev_predictions[node] = {}
                            prev_predictions[node][ts] = predictions_dict[ts][node]

                # --- ADD THIS TO PLOT THE PREDICTIONS ---
                for ts in predictions_dict.keys():
                    # Isolate a single timestep
                    single_ts_dict = {ts: predictions_dict[ts]}
                    single_cp_dict = cp_radius_dict.get(ts, {})
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    visualize_prediction(
                        ax=ax,
                        prediction_output_dict=single_ts_dict,
                        dt=scene.dt,
                        max_hl=max_hl,
                        ph=ph,
                        map=None,  # Or pass the environment map if you have it
                        cp_radius_dict=single_cp_dict
                    )
                    
                    ax.set_title(f"Diffusion Trajectories at Timestep {ts}")
                    
                    # Save the plot
                    plt.savefig(f"trajectory_plot_ts{ts}.png")
                    plt.close(fig)
                # ----------------------------------------

                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50

        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")

        # --- Plot CP Radius over time ---
        if len(cp_radius_history) > 0:
            from collections import defaultdict
            ts_to_radii = defaultdict(list)
            for ts, r in cp_radius_history:
                ts_to_radii[ts].append(r)
            
            sorted_ts = sorted(ts_to_radii.keys())
            mean_radii = [np.mean(ts_to_radii[ts]) for ts in sorted_ts]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(sorted_ts, mean_radii, color='red', linewidth=1.5, alpha=0.8, label='Mean CP Radius')
            ax.fill_between(sorted_ts, 0, mean_radii, color='red', alpha=0.1)
            ax.set_xlabel('Timestep', fontsize=12)
            ax.set_ylabel('CP Radius (meters)', fontsize=12)
            ax.set_title('Conformal Prediction Safety Shield Radius Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('cp_radius_over_time.png', dpi=150)
            plt.close(fig)
            print(f"CP Radius plot saved to cp_radius_over_time.png")
            print(f"CP Radius — Mean: {np.mean(mean_radii):.4f}m, Max: {np.max(mean_radii):.4f}m, Min: {np.min(mean_radii):.4f}m")


    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                        lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        device = torch.device(self.config.device)
        self.registrar = ModelRegistrar(self.model_dir, device)

        # Always load the supervised baseline first to populate the encoder!
        epoch = self.config.eval_at
        baseline_path = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
        print(f"Loading supervised baseline encoder from: {baseline_path}")
        baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
        self.registrar.load_models(baseline_checkpoint['encoder'])

        if self.config.eval_mode:
            rl_ckpt_path = self.config.get('rl_checkpoint_path', "")
            if rl_ckpt_path:
                print(f"Loading RL checkpoint from: {rl_ckpt_path}")
                self.checkpoint = torch.load(rl_ckpt_path, map_location="cpu", weights_only=False)
            else:
                self.checkpoint = baseline_checkpoint


        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        device = torch.device(self.config.device)
        self.encoder = Trajectron(self.registrar, self.hyperparams, device)


        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

        device = torch.device(self.config.device)
        self.model = model.to(device)

        if self.config.eval_mode:
            rl_ckpt_path = self.config.get('rl_checkpoint_path', "")
            if rl_ckpt_path:
                if 'ddpm' in self.checkpoint and 'encoder' in self.checkpoint:
                    # In case the user points to a nested baseline checkpoint here by mistake
                    new_state_dict = {}
                    for k, v in self.checkpoint['encoder'].items():
                        new_state_dict[f'encoder.{k}'] = v
                    for k, v in self.checkpoint['ddpm'].items():
                        new_state_dict[f'diffusion.{k}'] = v
                    self.model.load_state_dict(new_state_dict, strict=False)
                else:
                    # Standard RL flattened checkpoint
                    self.model.load_state_dict(self.checkpoint, strict=False)
            else:
                self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'],
                                           self.hyperparams['pred_state'],
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=1,
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.config.preprocess_workers)
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")
