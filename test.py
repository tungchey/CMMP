import numpy as np, argparse, time, pickle, random
import torch
from torch.utils.data import DataLoader
from dataloader import IEMOCAPDataset, MELDDataset
from sklearn.metrics import f1_score, accuracy_score
from opts import parse_opts

from CMMP import CMMP




def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    testset = MELDDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0, num_workers=0, pin_memory=False):
    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return test_loader



def eval_model(model, dataloader, epoch, cuda):
    losses, preds, labels = [], [], []
    model.eval()
    for data in dataloader:
        textf, _, _, _, visuf, acouf, qmask, umask, label = [d.cuda() for d in
                                                             data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        log_prob = model(textf, acouf, visuf, qmask, umask, lengths, epoch)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_accuracy, preds, avg_fscore


if __name__ == '__main__':

    args = parse_opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda = args.cuda
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 768, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 512}  # meld_audio 300 denseface 342  IS10:768
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = 768  # 1024
    D_text = 1024  # feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        D_m = 768  # 1024
    D_g = 768 if args.Dataset == 'IEMOCAP' else 768  # IEMOCAP 384
    D_e = 100
    g_h = 384
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    seed = 67137 if args.Dataset == 'MELD' else 1475


    visual_prompt, audio_prompt, text_prompt = torch.load('prompts.pkl')

    cmmp = CMMP(base_model=args.base_model, dataset=args.Dataset, multi_attn_flag=True, roberta_dim=768,
                    hidden_dim=768, dropout=0, num_layers_t=2, num_layers_av=2,
                    model_dim=768, num_heads=4, D_m_audio=512, D_m_visual=1000, D_m=D_m, D_g=D_g, D_e=D_e, g_h=g_h,
                    n_speakers=n_speakers, dropout2=args.dropout, no_cuda=args.no_cuda, model_type='hyper',
                    use_residue=args.use_residue, D_m_v=D_visual, D_m_a=D_audio, modals=args.modals,
                    n_classes=n_classes, device='cuda',
                    att_type=args.fusion, use_speaker=args.use_speaker, use_modal=args.use_modal, norm=args.norm,
                    visual_prompt=visual_prompt, audio_prompt=audio_prompt, text_prompt=text_prompt)


    if cuda:
        cmmp.cuda()


    if args.Dataset == 'MELD': # num_layers_av=1
        test_loader = get_MELD_loaders(valid=0.0, batch_size=batch_size, num_workers=2) 
    elif args.Dataset == 'IEMOCAP':  # num_layers_av=2
        test_loader = get_IEMOCAP_loaders(valid=0.0, batch_size=batch_size, num_workers=2)
    else:
        print("There is no such dataset")


    # state = torch.load("")
    state = torch.load("")
    cmmp.load_state_dict(state)
    print('testing loaded Model')
    test_acc, test_pred, test_fscore = eval_model(cmmp, test_loader, 0, cuda)
    print('test_acc:', test_acc, 'test_fscore:', test_fscore)
