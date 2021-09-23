import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "data")
parser.add_argument("density", help="density")
parser.add_argument("alpha_meta", type=float, help="learning rate of metagraph")
parser.add_argument("margin_meta", type=float, help="margin of metagraph")
parser.add_argument("adv_meta", type=float, help="adv rate of metagraph")
parser.add_argument("alpha", type=float, help="learning rate of original graph")
parser.add_argument("margin", type=float, help="margin of original graph")
parser.add_argument("adv", type=float, help="adv of original graph")
args = parser.parse_args()

data = args.data
density = args.density
alpha_meta = args.alpha_meta
margin_meta = args.margin_meta
adv_meta = args.adv_meta
alpha = args.alpha
margin = args.margin
adv_temperature = args.adv

path_meta = './result_square/' + density + '/' + data + '_meta/'
path_data = "./benchmarks/" + data + '/'

train_dataloader = TrainDataLoader(
	in_path = path_meta, 
	batch_size = 1000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

test_dataloader = TestDataLoader(path_meta, "link", False)

rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 512,
	margin = margin_meta,
	epsilon = 2,
)

model_meta = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = adv_meta),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

trainer = Trainer(model = model_meta, data_loader = train_dataloader, train_times = 200, alpha = alpha_meta, use_gpu = True, opt_method = "adam")
trainer.run()

f = open('./result_square/' + density + '/' + data + '_meta/labels_' + data + '_' + density + '.txt', 'r')
cluster = []
for line in f.readlines():
	cluster.append(int(line.strip()))
f.close()

a = torch.tensor(cluster, device = 'cuda:0')
ent_embeddings = rotate.ent_embeddings(a)
rel_embeddings = rotate.rel_embeddings

train_dataloader = TrainDataLoader(
	in_path = path_data, 
	batch_size = 1000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

test_dataloader = TestDataLoader(path_data, "link", False)

rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	ent_embeddings = torch.nn.Embedding.from_pretrained(ent_embeddings.detach().clone(), freeze = False),
	rel_embeddings = torch.nn.Embedding.from_pretrained(rel_embeddings.weight.data.detach().clone(), freeze = False),
	dim = 512,
	margin = margin,
	epsilon = 2,
)

model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = adv_temperature),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 500, alpha = alpha, use_gpu = True, opt_method = "adam")
trainer.run()

tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)