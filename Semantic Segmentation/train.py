import datetime
import os
import random
import time
from math import sqrt
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
# from tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import voc
from models import *

from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from tqdm import tqdm as tqdm
cudnn.benchmark = True
from torchvision.transforms import *
from torchvision.transforms import ToTensor, ToPILImage
ckpt_path = './ckpt'

exp_name = 'RSPPNET'

args = {
	'epoch_num': 200,
	'lr': 0.0001,
	'weight_decay': 0.0005,
	'momentum': 0.9,
	'lr_patience': 100,  # large patience denotes fixed lr
	'snapshot': '',  # empty string denotes learning from scratch
	'print_freq': 1,
	'val_save_to_img_file': False,
	'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}


def lr_poly(base_lr, iter,max_iter=200,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter, net, train_args):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(0.0001, i_iter)
    print('current lr:', lr)
    
    # optimizer.step()
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=train_args['weight_decay'], momentum=0.9, centered=False)
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=train_args['weight_decay'])
    optimizer = optim.SGD(net.parameters(),lr=lr, momentum=train_args['momentum'],weight_decay=train_args['weight_decay'])
    # optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(net), 'lr': lr }, {'params': get_10x_lr_params(net), 'lr': 10*lr} ], lr = lr, momentum = train_args['momentum'],weight_decay = train_args['weight_decay'])
    # optimizer.zero_grad()
    # optimizer.param_groups[1]['lr'] = lr * 10

max_label = 20

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
    
        
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 
    return Aiou

def main(train_args):
	net = PSPNet(num_classes=voc.num_classes).cuda()
	if len(train_args['snapshot']) == 0:
		curr_epoch = 1
		train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
	else:
		print('training resumes from ' + train_args['snapshot'])
		net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
		split_snapshot = train_args['snapshot'].split('_')
		curr_epoch = int(split_snapshot[1]) + 1
		train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
									 'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
									 'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}


	mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	input_transform = standard_transforms.Compose([
		ToTensor(),
		Normalize([.485, .456, .406], [.229, .224, .225]),
		
	])
	joint_transform = joint_transforms.Compose([
		joint_transforms.CenterCrop(224),
		# joint_transforms.Scale(2),
        joint_transforms.RandomHorizontallyFlip(),
    ])
	target_transform = standard_transforms.Compose([
		extended_transforms.MaskToTensor(),
		
	])
	restore_transform = standard_transforms.Compose([
		extended_transforms.DeNormalize(*mean_std),
		standard_transforms.ToPILImage(),
	])
	visualize = standard_transforms.Compose([
		standard_transforms.Scale(400),
		standard_transforms.CenterCrop(400),
		standard_transforms.ToTensor()
	])
	val_input_transform = standard_transforms.Compose([
		CenterCrop(224),
		ToTensor(),
		Normalize([.485, .456, .406], [.229, .224, .225]),
	])
	val_target_transform = standard_transforms.Compose([
		CenterCrop(224),
		extended_transforms.MaskToTensor(),
	])
	train_set = voc.VOC('train', transform=input_transform, target_transform=target_transform, joint_transform=joint_transform)
	train_loader = DataLoader(train_set, batch_size=4, num_workers=4, shuffle=True)
	val_set = voc.VOC('val', transform=val_input_transform, target_transform=val_target_transform)
	val_loader = DataLoader(val_set, batch_size=4, num_workers=4, shuffle=False)

	# criterion = CrossEntropyLoss2d(size_average=True, ignore_index=voc.ignore_label).cuda()
	criterion = torch.nn.CrossEntropyLoss(ignore_index=voc.ignore_label).cuda()
	optimizer = optim.SGD(net.parameters(),lr=train_args['lr'], momentum=train_args['momentum'],weight_decay=train_args['weight_decay'])

	check_mkdir(ckpt_path)
	check_mkdir(os.path.join(ckpt_path, exp_name))
	# open(os.path.join(ckpt_path, exp_name, 'loss_001_aux_SGD_momentum_95_random_lr_001.txt'), 'w').write(str(train_args) + '\n\n')

	for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
		# adjust_learning_rate(optimizer,epoch,net,train_args)
		train(train_loader, net, criterion, optimizer, epoch, train_args)
		validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)
		adjust_learning_rate(optimizer,epoch,net,train_args)
		# scheduler.step(val_loss)
    
def train(train_loader, net, criterion, optimizer, epoch, train_args):
	# interp = nn.Upsample(size=256, mode='bilinear')
	net.train()
	train_loss = AverageMeter()
	curr_iter = (epoch - 1) * len(train_loader)
	for i, data in enumerate(train_loader):
		inputs, labels = data
		assert inputs.size()[2:] == labels.size()[1:]
		N = inputs.size(0)
		inputs = Variable(inputs).cuda()
		labels = Variable(labels).cuda()
		random_number = random.random()
		if random_number > 0.5:
			optimizer.zero_grad()
			outputs,aux_logits = net(inputs)
			assert outputs.size()[2:] == labels.size()[1:]
			assert outputs.size()[1] == voc.num_classes
			loss_1 = criterion(outputs, labels)
			loss_2 = criterion(aux_logits, labels)
			loss = (loss_1 + 0.4*loss_2)*random.random()
			loss.backward()
			optimizer.step()
			train_loss.update(loss.data[0], N)
		else:
			optimizer.zero_grad()
			outputs,aux_logits = net(inputs)
			assert outputs.size()[2:] == labels.size()[1:]
			assert outputs.size()[1] == voc.num_classes
			loss_1 = criterion(outputs, labels)
			loss_2 = criterion(aux_logits, labels)
			loss = loss_1 + 0.4*loss_2
			loss.backward()
			optimizer.step()
			train_loss.update(loss.data[0], N)

		curr_iter += 1
		# writer.add_scalar('train_loss', train_loss.avg, curr_iter)

		if i % train_args['print_freq'] == 0:
			print('[epoch %d], [iter %d / %d], [train loss %.5f],[N: %d]' % (
				epoch, i + 1, len(train_loader), train_loss.avg, N
				# , loss_1.data[0], loss_2.data[0],[loss %.3f],[loss2 %.3f]
			))


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
	net.eval()
	global best_acc
	val_loss = AverageMeter()
	inputs_all, gts_all, predictions_all = [], [], []

	for vi, data in tqdm(enumerate(val_loader)):
		inputs, gts = data
		N = inputs.size(0)
		inputs = Variable(inputs, volatile=True).cuda()
		gts = Variable(gts, volatile=True).cuda()

		outputs = net(inputs)
		# interp = nn.Upsample(size=256, mode='bilinear')
		# outputs = interp(net(inputs))
		predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

		val_loss.update(criterion(outputs, gts).data[0], N)

		# if random.random() > train_args['val_img_sample_rate']:
		# 	inputs_all.append(None)
		# else:
		# 	inputs_all.append(inputs.data.squeeze_(0).cpu())
		gts_all.append(gts.data.squeeze_(0).cpu().numpy())
		predictions_all.append(predictions)
		# IOU.append(get_iou(outputs,gts))

	acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, voc.num_classes)
	
	if mean_iu > train_args['best_record']['mean_iu']:
		train_args['best_record']['val_loss'] = val_loss.avg
		train_args['best_record']['epoch'] = epoch
		train_args['best_record']['acc'] = acc
		train_args['best_record']['acc_cls'] = acc_cls
		train_args['best_record']['mean_iu'] = mean_iu
		train_args['best_record']['fwavacc'] = fwavacc
		snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f' % (
			epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc
		)
		open(os.path.join(ckpt_path, exp_name, 'loss_0001_dilation_aux_SGD_momentum_090_PSPNet_L3.txt'), 'a').write(str(epoch) + '_' + str(mean_iu) + ',')
		# torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))

	print('--------------------------------------------------------------------')
	print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
		epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

	print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
		train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
		train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

	print('--------------------------------------------------------------------')



if __name__ == '__main__':
	main(args)
