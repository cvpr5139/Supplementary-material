import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from utils import normalize
from utils import novelty_score
from models.Loss import TestMethod, NLL
from datasets.transforms import Static_intensity

class ResultsAccumulator:
    def __init__(self, time_steps):
        self._buffer = np.zeros(shape=(time_steps,), dtype=np.float32)
        self._counts = np.zeros(shape=(time_steps,))

    def push(self, score):
        self._buffer += score
        self._counts += 1

    def get_next(self):
        ret = self._buffer[0] / self._counts[0]

        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        self._buffer[-1] = 0
        self._counts[-1] = 0

        return ret

    @property
    def results_left(self):
        return np.sum(self._counts != 0).astype(np.int32)

class VideoAnomalyDetectionResultHelper(object):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.videoshape = args.videoshape
        self.multi_gpus = False
        self.cuda = torch.device("cuda:%d" % int(args.GPU) if torch.cuda.is_available() else "cpu")
        self.test_method = TestMethod(args.dataset, self.cuda)
        s = 'final'
        if args.checkpoint is not None:
            if (args.checkpoint).find('Step') >= 0:
                name, step = (args.checkpoint).split('Step_')
                s, _ = step.split('_checkpoint')
        if args.flow:
            self.output_file_val = os.path.join(args.modelsave, s + '_results_flow_val.txt')
            self.output_file = os.path.join(args.modelsave, s + '_results_flow.txt')
        else:
            self.output_file_val = os.path.join(args.modelsave, s + '_results_val.txt')
            self.output_file = os.path.join(args.modelsave, s + '_results.txt')
        self.outputdir = args.modelsave

        self.workers = args.workers
        self.dataname = args.dataset
        if self.dataname == 'shanghaitech':
            self.lamb = 1
        elif self.dataname == 'Avenue':
            self.lamb = 0.5
        elif self.dataname == 'Ped2':
            self.lamb = 0.1

    @torch.no_grad()
    def test_video_anomaly_detection(self, model, flow = None, val=False):
        _, t, _, _ = self.videoshape

        test_loglikelihood = NLL().to(self.cuda)
        test_loglikelihood.eval()
        model.eval()
        if flow is not None:
            flow.eval()
            vad_table = self.empty_table(True)
            flow_input = Static_intensity(self.cuda)
        else:
            vad_table = self.empty_table(False)

        global_rec = []
        global_f = []
        global_y = []

        results_accumulator_rec = ResultsAccumulator(time_steps=t)
        results_accumulator_nll = ResultsAccumulator(time_steps=t)

        if val == True:
            video_list = self.dataset.val_videos
        else:
            video_list = self.dataset.test_videos

        for cl_idx, video_id in enumerate(video_list):
            self.dataset.test(video_id)
            loader = DataLoader(self.dataset, num_workers= self.workers, collate_fn=self.dataset.collate_fn)
            sample_nll = np.zeros(shape=(len(loader) + t - 1,))
            sample_rec = np.zeros(shape=(len(loader) + t - 1,))
            sample_y = self.dataset.load_test_sequence_gt(video_id)

            for i, input in tqdm(enumerate(loader)):
                x = input[0]
                x = x.to(self.cuda)
                if flow is not None:
                    x_r, flow_z = model(x)
                    intensity = flow_input(input[0])
                    flows = torch.cat((flow_z[0], intensity), 1)
                    _, nll, _ = flow(flows.to(self.cuda))
                    # _, nll, _ = flow(flow_z[1].to(self.cuda))
                else:
                    x_r, _ = model(x)
                    nll = torch.Tensor([0.])
                if self.dataname == 'shanghaitech':
                    score = self.test_method(x, input[1], x_r)
                else:
                    score = self.test_method(x, None, x_r)
                results_accumulator_rec.push(self.test_method.reconstruction_loss)
                sample_rec[i] = results_accumulator_rec.get_next()

                test_nll = test_loglikelihood(nll)
                results_accumulator_nll.push(test_nll.item())
                sample_nll[i] = results_accumulator_nll.get_next()

            # Get last results
            while results_accumulator_rec.results_left != 0:
                index = (- results_accumulator_rec.results_left)
                sample_rec[index] = results_accumulator_rec.get_next()
                sample_nll[index] = results_accumulator_nll.get_next()

            min_nll, max_nll, min_rec, max_rec = self.compute_normalizing_coefficients(sample_nll, sample_rec)

            # Compute the normalized scores and novelty score
            sample_rec = normalize(sample_rec, min_rec, max_rec)
            if flow is not None:
                sample_nll = normalize(sample_nll, min_nll, max_nll)
                sample_f = novelty_score(sample_nll, sample_rec, self.lamb)
                global_f.append(sample_f)
            global_rec.append(sample_rec)
            global_y.append(sample_y)

            try:
                if flow is not None:
                    # Compute AUROC for this video
                    this_video_metrics = [
                        roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                        roc_auc_score(sample_y, sample_f),  # likelihood metric
                    ]
                else:
                    this_video_metrics = [
                        roc_auc_score(sample_y, sample_rec) ]
                vad_table.add_row([video_id] + list(this_video_metrics))
            except ValueError:
                # This happens for sequences in which all frames are abnormal
                # Skipping this row in the table (the sequence will still count for global metrics)
                continue

        # Compute global AUROC and print table
        global_rec = np.concatenate(global_rec)
        global_y = np.concatenate(global_y)
        if flow is not None:
            global_f = np.concatenate(global_f)

        if flow is not None:
            global_metrics = [
                roc_auc_score(global_y, global_rec),  # reconstruction metric
                roc_auc_score(global_y, global_f),  # likelihood metric
            ]
        else:
            global_metrics = [
                roc_auc_score(global_y, global_rec)
            ]
        vad_table.add_row(['avg'] + list(global_metrics))
        print(vad_table)
        # Save table
        if val:
            with open(self.output_file_val, mode='a+') as f:
                f.write(str(vad_table))
                f.write('\n')
            return list(global_metrics)
        else:
            with open(self.output_file, mode='a+') as f:
                f.write(str(vad_table))
                f.write('\n')
            return list(global_metrics)

    @torch.no_grad()
    def test_video_anomaly_detection_multiflow(self, model, flow1, flow2, val=False):
        _, t, _, _ = self.videoshape
        flow_input = Static_intensity(self.cuda)
        test_loglikelihood = NLL().to(self.cuda)
        test_loglikelihood.eval()
        model.eval()
        flow1.eval()
        flow2.eval()
        vad_table = PrettyTable()
        vad_table.field_names = ['VIDEO-ID', 'AUROC-Recon', 'AUROC-FlowS', 'AUROC-FlowD', 'AUROC-Total']
        vad_table.float_format = '0.4'

        global_rec = []
        global_fa = []
        global_fm = []
        global_ns = []
        global_y = []

        results_accumulator_rec = ResultsAccumulator(time_steps=t)
        results_accumulator_nll1 = ResultsAccumulator(time_steps=t)
        results_accumulator_nll2 = ResultsAccumulator(time_steps=t)

        if val == True:
            video_list = self.dataset.val_videos
        else:
            video_list = self.dataset.test_videos

        for cl_idx, video_id in enumerate(video_list):
            self.dataset.test(video_id)
            loader = DataLoader(self.dataset, num_workers=self.workers, collate_fn=self.dataset.collate_fn)
            sample_nll1 = np.zeros(shape=(len(loader) + t - 1,))
            sample_nll2 = np.zeros(shape=(len(loader) + t - 1,))
            sample_rec = np.zeros(shape=(len(loader) + t - 1,))
            sample_y = self.dataset.load_test_sequence_gt(video_id)

            for i, input in tqdm(enumerate(loader)):
                x = input[0]
                x = x.to(self.cuda)
                intensity = flow_input(input[0])

                x_r, flow_z = model(x)
                flows = torch.cat((flow_z[0], intensity), 1)
                _, nll1, _ = flow1(flows.to(self.cuda))
                _, nll2, _ = flow2(flow_z[1].to(self.cuda))

                if self.dataname == 'shanghaitech':
                    score = self.test_method(x, input[1], x_r)
                else:
                    score= self.test_method(x, None, x_r)
                results_accumulator_rec.push(self.test_method.reconstruction_loss)
                sample_rec[i] = results_accumulator_rec.get_next()

                test_nll1 = test_loglikelihood(nll1)
                results_accumulator_nll1.push(test_nll1.item())
                sample_nll1[i] = results_accumulator_nll1.get_next()

                test_nll2 = test_loglikelihood(nll2)
                results_accumulator_nll2.push(test_nll2.item())
                sample_nll2[i] = results_accumulator_nll2.get_next()

            # Get last results
            while results_accumulator_rec.results_left != 0:
                index = (- results_accumulator_rec.results_left)
                sample_rec[index] = results_accumulator_rec.get_next()
                sample_nll1[index] = results_accumulator_nll1.get_next()
                sample_nll2[index] = results_accumulator_nll2.get_next()

            min_nll1, max_nll1, min_nll2, max_nll2 = self.compute_normalizing_coefficients(sample_nll1, sample_nll2)
            sample_nll = ((max_nll1-min_nll1)/(max_nll1-min_nll1+max_nll2-min_nll2))*sample_nll1 + ((max_nll2-min_nll2)/(max_nll1-min_nll1+max_nll2-min_nll2))*sample_nll2
            # sample_nll = sample_nll1+sample_nll2
            min_nll, max_nll, min_rec, max_rec = self.compute_normalizing_coefficients(sample_nll, sample_rec)

            # Compute the normalized scores and novelty score
            sample_rec = normalize(sample_rec, min_rec, max_rec)
            sample_nll1 = normalize(sample_nll1, min_nll1, max_nll1)
            sample_nll2 = normalize(sample_nll2, min_nll2, max_nll2)
            sample_nll = normalize(sample_nll, min_nll, max_nll)
            sample_ns = novelty_score(sample_nll, sample_rec, self.lamb)
            sample_fa = novelty_score(sample_nll1, sample_rec, self.lamb)
            sample_fm = novelty_score(sample_nll2, sample_rec, self.lamb)
            global_ns.append(sample_ns)
            global_rec.append(sample_rec)
            global_y.append(sample_y)
            global_fa.append(sample_fa)
            global_fm.append(sample_fm)

            try:
                this_video_metrics = [
                    roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                    roc_auc_score(sample_y, sample_fa),  # likelihood metric
                    roc_auc_score(sample_y, sample_fm),  # likelihood metric
                    roc_auc_score(sample_y, sample_ns)  # novelty score
                ]
                vad_table.add_row([video_id] + list(this_video_metrics))
            except ValueError:
                # This happens for sequences in which all frames are abnormal
                # Skipping this row in the table (the sequence will still count for global metrics)
                continue
        # Compute global AUROC and print table
        global_ns = np.concatenate(global_ns)
        global_fa = np.concatenate(global_fa)
        global_fm = np.concatenate(global_fm)
        global_rec = np.concatenate(global_rec)
        global_y = np.concatenate(global_y)
        global_metrics = [
            roc_auc_score(global_y, global_rec),  # reconstruction metric
            roc_auc_score(global_y, global_fa),  # likelihood metric
            roc_auc_score(global_y, global_fm),  # likelihood metric
            roc_auc_score(global_y, global_ns),  # novelty score
        ]
        vad_table.add_row(['avg'] + list(global_metrics))
        print(vad_table)

        # Save table
        if val:
            with open(self.output_file_val, mode='a+') as f:
                f.write(str(vad_table))
                f.write('\n')
            return list(global_metrics)
        else:
            with open(self.output_file, mode='a+') as f:
                f.write(str(vad_table))
                f.write('\n')
            return list(global_metrics)

    @staticmethod
    def compute_normalizing_coefficients(sample_llk, sample_rec):
        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    # @property
    def empty_table(self, flow):
        table = PrettyTable()
        if flow:
            table.field_names = ['VIDEO-ID', 'AUROC-REC', 'AUROC-F']
        else:
            table.field_names = ['VIDEO-ID', 'AUROC-REC']
        table.float_format = '0.4'
        return table