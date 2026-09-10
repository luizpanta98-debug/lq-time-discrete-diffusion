from unet import UNet
from data import load_imagedataset
from dm import DM
from seed import set_seed
from tqdm import tqdm
import numpy as np
import json
import os 
import torch as th
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

def train(net, dm, train_loader, val_loader, config):
	with open(f"{config['save_checkpoint']}/../config.json", "w") as f:
		json.dump(config, f, indent=4)

	if config['ema']:
		ema = AveragedModel(net,
		multi_avg_fn=get_ema_multi_avg_fn(config['ema_decay']))
		if config['load_checkpoint']:
			ema.load_state_dict(th.load(f"{config['save_checkpoint']}/ema.pt", weights_only=False))

	opt = th.optim.Adam(net.parameters(), lr=config['lr'])
	acc_steps = config['batch_size'] // config['virtual_batch_size']
	for epoch in range(1,config['epochs']+1):
		# EPOCH IN
		loss_epoch=[]
		val_epoch=[]
		for i, images in tqdm(list(enumerate(train_loader))):
			#opt.zero_grad()
			images = images.to(config['device'])
			labels=None
			# labels = labels.to(config['device'])
			ts = th.randint(0, dm.T, (images.shape[0], ), device=config['device'])
			xts, eps = dm.q(images, ts)
			eps_pred = net(xts, ts, labels)
			loss = F.mse_loss(eps_pred, eps)
			loss_epoch.append(loss.item())
			(loss / acc_steps).backward()
			#opt.step()
			if (i+1)%acc_steps == 0:
				opt.step()
				opt.zero_grad()
				if config['ema']:
					ema.update_parameters(net)

		with th.inference_mode():
			for i, images in tqdm(list(enumerate(val_loader))):
				images = images.to(config['device'])
				labels=None
				# labels = labels.to(config['device'])
				ts = th.randint(0, dm.T, (images.shape[0], ), device=config['device'])
				xts, eps = dm.q(images, ts)
				eps_pred = net(xts, ts, labels)
				val_loss = F.mse_loss(eps_pred, eps)
				val_epoch.append(val_loss.item())

		if epoch % config['save_freq'] == 0:
			th.save(net.state_dict(), f"{config['save_checkpoint']}/last_weights.pt")
			print(f"model weights saved -> {config['save_checkpoint']}/last_weights.pt")
			if config['ema']:
				th.save(ema.state_dict(), f"{config['save_checkpoint']}/ema.pt")
				print(f"ema saved -> {config['save_checkpoint']}/ema.pt")
				th.save(ema.module.state_dict(), f"{config['save_checkpoint']}/ema_weights.pt")
				print(f"ema weights saved -> {config['save_checkpoint']}/ema_weights.pt")
			
		print(f'\nepoch {epoch}\nloss {round(np.mean(loss_epoch),4)}\nval {round(np.mean(val_epoch),4)}\n')
		# EPOCH OUT

if __name__ == "__main__":
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	with open("config.json", "r") as f:
		config = json.load(f)

	set_seed(config['seed'])
	os.makedirs(config['save_checkpoint'], exist_ok=True)

	train_dataset, val_dataset = load_imagedataset(config['data_fp'], config['img_resolution'], config['split'])
	train_loader = DataLoader(train_dataset, batch_size=config['virtual_batch_size'], shuffle=True, drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=config['virtual_batch_size'], shuffle=True)
	dm = DM()
	net = UNet(
		img_resolution=config["img_resolution"],
		in_channels=config["in_channels"], 
		out_channels=config["out_channels"], 
		label_dim=config["label_dim"],
		model_channels=config["model_channels"], 
		channel_mult=config["channel_mult"],
		channel_mult_emb=config["channel_mult_emb"],
		num_res_blocks=config["num_res_blocks"],
		attn_resolutions=config["attn_resolutions"],
		).to(config['device'])

	if config['load_checkpoint']:	
		net.load_state_dict(th.load(
				f"{config['save_checkpoint']}/last_weights.pt",
				weights_only=False))
		print('last weights loaded.')

	print(f'train data size: {len(train_dataset)}\nval data size: {len(val_dataset)}')
	train(net, dm, train_loader, val_loader, config)
