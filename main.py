# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter
import logging
import argparse
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from code.common import *

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

from code.KEMP import KEMP
from code.dataloader import prepare_data_seq

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def load_params():
    if (os.cpu_count() > 8):
        USE_CUDA = True
    else:
        USE_CUDA = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="data/kemp_dataset_preproc.json", help='processed EmpatheticDialogue dataset')
    parser.add_argument("--save_path", type=str, default="save/test/", help='path to save the training files')
    parser.add_argument("--resume_path", type=str, default="result/", help='path to save the checkpoint file')
    parser.add_argument("--tokenizer_dir", type=str, default="data/", help='path to tokenization file')
    parser.add_argument("--emb_file", type=str, default='', help='path to glove embedding file')

    ## training
    parser.add_argument("--model", type=str, default="seq2seq", help='model name, [KEMP, wo_ECE, wo_EDD]')
    parser.add_argument("--use_cuda", type=bool, default=True, help='gpu is available or not')
    parser.add_argument("--cuda", action="store_true", help='use gpu or not')
    parser.add_argument('--device_id', dest='device_id', type=str, default="0", help='gpu device id')
    parser.add_argument('--eps', type=float, default=1e-9, help='arg in NoamOpt')
    parser.add_argument('--epochs', type=int, default=10000, help='training iterations')
    parser.add_argument('--check_iter', type=int, default=2000, help='validation iterations')
    parser.add_argument("--noam", action="store_true", help='NoamOpt')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2, help='dropout')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument("--plm", action="store_true", help='use pretraining model or not')
    parser.add_argument("--use_oov_emb", action="store_true", help='')
    parser.add_argument("--pretrain_emb", action="store_true", help='use pretrained embedding (glove) or not')
    parser.add_argument("--weight_sharing", action="store_true", help='sharing params between input embedding and output proj')
    parser.add_argument("--label_smoothing", action="store_true", help='label smoothing loss')
    parser.add_argument("--universal", action="store_true", help='universal transformer')
    parser.add_argument("--act", action="store_true", help='arg in universal transformer, adaptive computation time')
    parser.add_argument("--act_loss_weight", type=float, default=0.001, help='arg in universal transformer')
    parser.add_argument("--specify_model", action="store_true", help='arg for resuming training')


    ## testing
    parser.add_argument("--test", action="store_true", help='true for inference, false for training')
    parser.add_argument("--train_then_test", action="store_true", help='test model if the training finishes')
    parser.add_argument("--beam_search", action="store_true", help='beam decoding')
    parser.add_argument("--beam_size", type=int, default=5, help='beam size')
    parser.add_argument("--topk", type=int, default=0, help='topk sampling')

    ## transformer
    parser.add_argument("--hidden_dim", type=int, default=100, help='hidden size')
    parser.add_argument("--emb_dim", type=int, default=100, help='embedding dimension')
    parser.add_argument("--hop", type=int, default=6, help='number of transformer layers')
    parser.add_argument("--heads", type=int, default=1, help='number of attention heads')
    parser.add_argument("--depth", type=int, default=40, help='size of last dimension of keys/values. Must be divisible by number of heads')
    parser.add_argument("--filter", type=int, default=50, help='hidden size of the middle layer in FFN.')
    parser.add_argument("--project", action="store_true", help='project the input of decoder from embedding dimension to hidden dimension')
    parser.add_argument("--concept_num", type=int, default=3, help='the maximum number of external concepts injection for a word.')
    parser.add_argument("--total_concept_num", type=int, default=10, help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--max_seq_length", type=int, default=1000, help='max sequence length (required for timing signal)')
    parser.add_argument("--pointer_gen", action="store_true", help='copy mechanism')
    parser.add_argument("--attn_loss", action="store_true", help="emotional attention loss")
    parser.add_argument("--emotion_feature", action="store_true", help="emotional feature")

    args = parser.parse_args()
    print_opts(args)

    args.emb_file = args.emb_file or "data/glove.6B.{}d.txt".format(str(args.emb_dim))
    if (not args.test):
        args.save_path_dataset = args.save_path

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')  # ,filename='save/logs/{}.log'.format(str(name)))
    args.collect_stats = False

    args.UNK_idx = 0
    args.PAD_idx = 1
    args.EOS_idx = 2
    args.SOS_idx = 3
    args.USR_idx = 4  # speak state
    args.SYS_idx = 5  # listener state
    args.KG_idx = 6  # concept state
    args.CLS_idx = 7
    args.SEP_idx = 8
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.USE_CUDA = USE_CUDA
    return args

args = load_params()

os.environ["CUDA_VISOBLE_DEVICES"] = args.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device_id))


if __name__ == '__main__':
    args = load_params()
    print_file = None
    eval_file = open(args.model+'_eval.txt', 'w')
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(args, batch_size=args.batch_size)
    print('-----finish loading data--------')

    model = KEMP(args, vocab, decoder_number=program_number)

    model_save_path = os.path.join(args.save_path, 'result')
    if os.path.exists(model_save_path) is False: os.makedirs(model_save_path)
    log_save_path = os.path.join(args.save_path, 'save')
    if os.path.exists(log_save_path) is False: os.makedirs(log_save_path)

    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and args.pretrain_emb):
            xavier_uniform_(p)

    print("MODEL USED", args.model, file=print_file)
    print("TRAINABLE PARAMETERS", count_parameters(model), file=print_file)

    if args.test is False:
        try:
            model = model.to(args.device)
            model = model.train()
            best_ppl = 1000
            patient = 0
            writer = SummaryWriter(log_dir=args.save_path)
            weights_best = deepcopy(model.state_dict())
            data_iter = make_infinite(data_loader_tra)

            for n_iter in tqdm(range(1000000)):
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)
                writer.add_scalars('loss', {'loss_train': loss}, n_iter)
                writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
                writer.add_scalars('bce', {'bce_train': bce}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
                if args.noam:
                    writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)

                if (n_iter + 1) % args.check_iter == 0:
                    model = model.eval()
                    model.epoch = n_iter
                    model.__id__logger = 0
                    loss_val, ppl_val, bce_val, acc_val = evaluate(args, model, data_loader_val, ty="valid", max_dec_step=50, print_file=eval_file)
                    writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                    writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                    writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                    writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
                    model = model.train()

                    if n_iter < 13000:
                        continue
                    if ppl_val <= best_ppl:
                        best_ppl = ppl_val
                        patient = 0
                        ## SAVE MODEL
                        torch.save({"model": model.state_dict(),
                                    "result": [loss_val, ppl_val, bce_val, acc_val]},
                                   os.path.join(model_save_path, 'model_{}_{:.4f}.tar'.format(n_iter, best_ppl)))
                        weights_best = deepcopy(model.state_dict())
                        print("best_ppl: {}; patient: {}".format(best_ppl, patient), file=print_file)
                    else:
                        patient += 1
                        print("patient is: {} now".format(patient), file=print_file)
                    if patient > 2: break
        except KeyboardInterrupt:
            print('-' * 89, file=print_file)
            print('Exiting from training early', file=print_file)

        ## SAVE THE BEST
        torch.save({'models': weights_best,
                    'result': [loss_val, ppl_val, bce_val, acc_val], },
                   os.path.join(model_save_path, args.model+'_best.tar'))
        print('Saving the best checkpoint in {}'.format(os.path.join(model_save_path, args.model+'_best.tar')))

        ## TESTING
        if args.train_then_test:
            model.load_state_dict({name: weights_best[name] for name in weights_best})
            model.eval()
            model.epoch = 100
            with torch.no_grad():
                loss_test, ppl_test, bce_test, acc_test = evaluate(args, model, data_loader_tst, ty="test", max_dec_step=50, print_file=print_file)
        else:
            loss_test, ppl_test, bce_test, acc_test = 0,0,0,0
    else:  # test
        print("TESTING !!!", file=print_file)
        model = model.to(args.device)
        model = model.eval()

        if args.specify_model:
            checkpoint = torch.load(args.resume_path, map_location=lambda storage, location: storage)
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(os.path.join(model_save_path, args.model+'_best.tar'),
                                    map_location=lambda storage, location: storage)
            weights_best = checkpoint['models']
            model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        loss_test, ppl_test, bce_test, acc_test = evaluate(args, model, data_loader_tst, ty="test", max_dec_step=30, print_file=eval_file)

    print("model: ", args.model, "End .", file=print_file)
    if args.test or args.train_then_test:
        if args.specify_model:
            file_summary = "_summary.txt"
        else:
            file_summary = os.path.join(model_save_path,'summary.txt')
        with open(file_summary, 'w') as the_file:
            the_file.write("EVAL\tLoss\tPPL\tAccuracy\n")
            the_file.write(
                "{}\t{:.4f}\t{:.4f}\t{:.4f}".format("test", loss_test, ppl_test, acc_test))

