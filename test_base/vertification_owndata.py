from data_pipe import de_preprocess, get_train_loader, get_val_data,get_val_data_own
import torch
import numpy as np
import os
from pathlib import Path
from verifacation import evaluate
import face_model
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model',
                        default=u"models/model_2020-04-23-01-13_accuracy_0.9352857142857143_step_189440_None.pth",
						#default=u"models/model_2020-04-23-01-13_accuracy_0.9352857142857143_step_189440_None.pth",
                        help='path to load model.')
    parser.add_argument('--device_ids', default=[4], type=list, help='gpu id')
    parser.add_argument('--net_depth', default=50, type=int, help='network depth')
    parser.add_argument('--drop_ratio', default=0.4, type=float, help='network drop ratio')
    parser.add_argument('--net_mode', default='ir', help='network mode,ir or ir_se')
    parser.add_argument('--test_path', default='/ssd_data/face_ms1s', help='test db path')
    parser.add_argument('--embedding_size', default=256, type=int, help='network embedding size')
    args = parser.parse_args()
    return args
def  recalculate(best_threshold):
    return (2.0-best_threshold)/2
def learner_evaluate(model,carray, issame,embedding_size=512, batch_size=100, nrof_folds = 5, tta = False):

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            #print(batch_size)
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device)) + model(fliped.to(device))
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                embeddings[idx:idx + batch_size] = model(batch.to(device)).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device)) + model(fliped.to(device))
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                embeddings[idx:] = model(batch.to(device)).cpu()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    best_threshold_ret=recalculate(best_thresholds.mean())
    return accuracy.mean(), best_threshold_ret
if __name__ =="__main__":
    args = parse_args()
    print(os.path.basename(args.model))
    model = face_model.FaceModel(args)

    print('learner loaded')
    use_gpu = model.use_gpu
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_path = Path(args.test_path)
    agedb_30,cfp_fp, lfw,agedb_30_issame,cfp_fp_issame,lfw_issame = get_val_data(data_path)
    accuracy1, best_threshold  = learner_evaluate(model.model, agedb_30,agedb_30_issame,embedding_size=args.embedding_size)
    print ("agedb_30",accuracy1, best_threshold)
    accuracy2, best_threshold  = learner_evaluate(model.model, lfw,lfw_issame,embedding_size=args.embedding_size)
    print ("lfw",accuracy2, best_threshold)
    accuracy3, best_threshold  = learner_evaluate(model.model, cfp_fp, cfp_fp_issame,embedding_size=args.embedding_size)
    print ("cfp_fp",accuracy3, best_threshold)
   