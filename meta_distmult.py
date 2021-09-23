import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "data")
parser.add_argument("density", help="density")
parser.add_argument("alpha_meta", type=float, help="learning rate of metagraph")
parser.add_argument("regul_meta", type=float, help="regularization rate of metagraph")
parser.add_argument("alpha", type=float, help="learning rate of original graph")
parser.add_argument("regul", type=float, help="regularization rate of original graph")
args = parser.parse_args()

data = args.data
density = args.density
alpha_meta = args.alpha_meta
regul_meta = args.regul_meta
alpha = args.alpha
regul_rate = args.regul

path_meta = './result_square/' + density + '/' + data + '_meta/'
path_data = "./benchmarks/" + data + '/'

train_dataloader_meta = TrainDataLoader(
	in_path = path_meta, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

test_dataloader_meta = TestDataLoader(path_meta, "link", False)

distmult_meta = DistMult(
	ent_tot = train_dataloader_meta.get_ent_tot(),
	rel_tot = train_dataloader_meta.get_rel_tot(),
	dim = 200
)

model_meta = NegativeSampling(
	model = distmult_meta, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader_meta.get_batch_size(), 
	regul_rate = regul_meta
)

trainer_meta = Trainer(model = model_meta, data_loader = train_dataloader_meta, train_times = 200, alpha = alpha_meta, use_gpu = True, opt_method = "adagrad")
trainer_meta.run()

f = open('./result_square/' + density + '/' + data + '_meta/labels_' + data + '_' + density + '.txt', 'r')
cluster = []
for line in f.readlines():
	cluster.append(int(line.strip()))
f.close()

a = torch.tensor(cluster, device = 'cuda:0')
ent_embeddings = distmult_meta.ent_embeddings(a)
rel_embeddings = distmult_meta.rel_embeddings

train_dataloader = TrainDataLoader(
	in_path = path_data, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

test_dataloader = TestDataLoader(path_data, "link", False)

distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	ent_embeddings = torch.nn.Embedding.from_pretrained(ent_embeddings.detach().clone(), freeze = False),
	rel_embeddings = torch.nn.Embedding.from_pretrained(rel_embeddings.weight.data.detach().clone(), freeze = False),
	dim = 200
)

model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = regul_rate
)

trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 500, alpha = alpha, use_gpu = True, opt_method = "adagrad")
trainer.run()

tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)