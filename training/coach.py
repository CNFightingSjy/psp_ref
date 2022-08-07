import os
import matplotlib
import matplotlib.pyplot as plt
from yaml import load

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, adv_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
# from models.psp import pSp
from models.psp_ref import pSp
from ranger import Ranger
from refDataLoader import refDataset
from models.stylegan2.model import Discriminator
import pdb


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network
		self.size = self.opts.output_size
		self.channel_multiplier = self.opts.channel_multiplier
		self.net = pSp(self.opts).to(self.device)
		self.discriminator = Discriminator(self.size, self.channel_multiplier).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
		if self.opts.adv_lambda > 0:
			# self.d_logistic_loss = adv_loss.d_logistic_loss().to(self.device).eval()
			# self.g_nonsaturating_loss = adv_loss.g_nonsaturating_loss().to(self.device).eval()
			self.adv_loss = adv_loss.AdvLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		# 在此增加对于reference的dataloader
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def requires_grad(model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
    		# 此处的dataloader包含灰度图x，色块图r，目标图像y
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, r, y = batch
				x, r, y = x.to(self.device).float(), r.to(self.device).float(), y.to(self.device).float()
				# 训练判别器
				for p in self.net.parameters():
					p.requires_grad = False
				for p in self.discriminator.parameters():
					p.requires_grad = True
				y_hat, latent = self.net.forward(x, r, return_latents=True)
				# 此处添加生成图片和参考图片cat操作，变为4维
				# print(y_hat[0].shape)
				# print(r[0].shape)
				# print(torch.cat((y_hat[0], r[0]), dim=0).shape)
				real_pred = []
				fake_pred = []
				for i in range(self.opts.batch_size):
					fake = self.discriminator(torch.cat((y_hat[0], r[0]), dim=0).unsqueeze(0))
					real = self.discriminator(torch.cat((y[0], r[0]), dim=0).unsqueeze(0))# 修改x为转换为lab空间的原图y
					real_pred.append(real)
					fake_pred.append(fake)
				# fake = torch.cat((y_hat, r),dim=0)
				# real = torch.cat((x, r), dim=0)
				# fake_pred = self.discriminator(fake)
				# real_pred = self.discriminator(real)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, real_pred, fake_pred, False, latent)
				# 将生成器梯度清零
				self.discriminator.zero_grad()
				loss.backward()
				self.optimizer.step()

				# 训练生成器，经过encoder和合成网络返回生成的图片和latent code
				for p in self.net.parameters():
					p.requires_grad = True
				for p in self.discriminator.parameters():
					p.requires_grad = False
				y_hat, latent = self.net.forward(x, r, return_latents=True)
				fake_pred_g = []
				for i in range(self.opts.batch_size):
					fake_g = self.discriminator(torch.cat((y_hat[0], r[0]), dim=0).unsqueeze(0))
					fake_pred_g.append(fake_g)
				# fake = torch.cat((y_hat, r), dim=0)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, None, fake_pred_g, True, latent)
				self.net.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, r, y, y_hat, title='images/train/clothes')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.opts.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, r, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1


	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, r, y = batch
			# print(batch_idx)
			with torch.no_grad():
				# 增减输入reference
				x, r, y = x.to(self.device).float(), r.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, r, return_latents=True)
				real_pred = []
				fake_pred = []
				for i in range(self.opts.batch_size):
					fake = self.discriminator(torch.cat((y_hat[0], r[0]), dim=0).unsqueeze(0))
					real = self.discriminator(torch.cat((y[0], r[0]), dim=0).unsqueeze(0))# 修改x为转换为lab空间的原图y
					real_pred.append(real)
					fake_pred.append(fake)
				# fake = torch.cat((y_hat, r),dim=0)
				# real = torch.cat((x, r), dim=0)# 修改x为真实的图像
				# fake_pred = self.discriminator(fake)
				# real_pred = self.discriminator(real)
				loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, real_pred , fake_pred, False, latent) #real fake train_g
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, r, y, y_hat,
									  title='images/test/clothes',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.opts.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, r, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict		
	
	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	# 在此增加对于判别器的优化器
	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
			params += list(self.discriminator.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	# 更换ImagesDataset为自定义的Dataset
	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		# 添加了两个ref参数
		train_dataset = refDataset(source_root=dataset_args['train_source_root'],
									  ref_root=dataset_args['train_ref_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  ref_transform=transforms_dict['transform_ref'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = refDataset(source_root=dataset_args['test_source_root'],
									 ref_root=dataset_args['test_ref_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 ref_transform=transforms_dict['transform_ref'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		if self.opts.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset
	
	# 添加参数判别器输出real, fake对应真实和虚假预测值，是否计算判别器Loss标志位
	def calc_loss(self, x, y, y_hat, real, fake, train_g, latent):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		# 修改loss，修改loss_id为计算颜色的loss
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda
		# 此处添加对抗损失
		if self.opts.adv_lambda > 0:
			loss_adv = self.adv_loss(real, fake, train_g)
			loss_dict['loss_adv'] = float(loss_adv)
			loss += loss_adv * self.opts.adv_lambda
			# if not train_g:
			# 	loss_adv = self.d_logistic_loss(real, fake)
			# 	loss_dict['loss_adv'] = float(loss_adv)
			# 	loss += loss_adv * self.opts.adv_lambda
			# elif train_g:
			# 	loss_adv = self.g_nonsaturating_loss(fake)
			# 	loss_dict['loss_adv'] = float(loss_adv)
			# 	loss += loss_adv * self.opts.adv_lambda
				

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	# 增加参考图像reference
	def parse_and_log_images(self, id_logs, x, r, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		# print("x:",x[0].shape,"r:",r[0].shape,"y:",y[0].shape,"y_hat:",y_hat[0].shape)
		for i in range(display_count):
			# x[i] = torch.squeeze(x[i])
			# print(x[i].shape)
			cur_im_data = {
				'input_cloth': common.tensor2gray(x[i]),
				'reference': common.tensor2im(r[i]),
				'target_cloth': common.tensor2im(y[i]),
				'output_cloth': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
