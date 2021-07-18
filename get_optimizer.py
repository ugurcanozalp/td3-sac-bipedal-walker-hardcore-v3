import torch

def get_optimizer(model, lr, wd):
	no_decay = ['bias']
	optimizer_grouped_parameters = [
	        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'learning_rate': lr},
	        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'learning_rate': lr}
	]
	opt = torch.optim.AdamW(optimizer_grouped_parameters)
	return opt