import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ...models.subNets.Statical import StatisticNetwork, shuffle_batch, create_fake_samples, div_metric
from ...models.subNets.Decouple import alignment_loss, orth_loss, rec_loss, Decouple
from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class MULT():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        self.stat_net = StatisticNetwork(input_dim=combined_dim, hidden_dim=128)
        self.lambda_ = args.get("lambda", 1e-3)
        self.decouple_net = Decouple(dst_feature_dims * 2) # NOTE: define decouple net
        
        
    def train_stat(self, dataloader, train_steps=None):
        loss_hist = []
        self.model.eval()
        current_step = 0
        # check if self.model will not be updated
        print(f"check if self.model will not be updated: {self.model.training}")
        
        with tqdm(dataloader) as td:
            if train_steps is not None:
                td.total = train_steps
                
            for batch_data in td:
                if train_steps is not None and current_step >= train_steps:
                    break
                current_step += 1
                
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device)
                if self.args.get("data_missing"):
                    text_m = batch_data['text_m'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    
                    # simply replace input with corrupted data
                    text = text_m
                    audio = audio_m
                    vision = vision_m
                    
                elif self.args.get("data_dropping"):
                    text_d = batch_data['text_d'].to(self.args.device)
                    text_dropping_mask = batch_data['text_dropping_mask'].to(self.args.device)
                    vision_d = batch_data['vision_d'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    vision_dropping_mask = batch_data['vision_dropping_mask'].to(self.args.device)
                    audio_d = batch_data['audio_d'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)
                    audio_dropping_mask = batch_data['audio_dropping_mask'].to(self.args.device)
                    
                    
                    text = text_d
                    audio = audio_d
                    vision = vision_d
                # forward
                text_neg, audio_neg, vision_neg = create_fake_samples(text, audio, vision)
                text_neg, audio_neg, vision_neg = text_neg.to(self.args.device), audio_neg.to(self.args.device), vision_neg.to(self.args.device)
                # with torch.no_grad():
                pos_outputs = self.model(text, audio, vision)
                
                t_pos, v_pos, a_pos = pos_outputs['Feature_t'], pos_outputs['Feature_v'], pos_outputs['Feature_a']
                
                comb_pos = torch.cat([t_pos, a_pos, v_pos], dim=1)
                scores_pos = self.stat_net(comb_pos)
                neg_outputs = self.model(text_neg, audio_neg, vision_neg)
                t_neg, v_neg, a_neg = neg_outputs['Feature_t'], neg_outputs['Feature_v'], neg_outputs['Feature_a']
                comb_neg = torch.cat([t_neg, a_neg, v_neg], dim=1)
                # NOTE: Add decouple Network
                t_s, a_s, v_s, t_p, a_p, v_p = self.decouple_net(t_pos, a_pos, v_pos)
                
                nt_s, na_s, nv_s, nt_p, na_p, nv_p = self.decouple_net(t_neg, a_neg, v_neg)
                
                
                # NOTE loss: 
                l_align = alignment_loss(t_s, a_s, v_s)
                l_orth = ( orth_loss(t_s, t_p) + orth_loss(a_s, a_p) + orth_loss(v_s, v_p) ) / 3.
                
                l_rec = (rec_loss(t_s, t_p, t_pos) + rec_loss(a_s, a_p, a_pos) + rec_loss(v_s, v_p, v_pos) ) / 3.
                
                n_l_align = alignment_loss(nt_s, na_s, nv_s)
                n_l_orth = ( orth_loss(nt_s, nt_p) + orth_loss(na_s, na_p) + orth_loss(nv_s, nv_p) ) / 3.
                n_l_rec = (rec_loss(nt_s, nt_p, t_neg) + rec_loss(na_s, na_p, a_neg) + rec_loss(nv_s, nv_p, v_neg) ) / 3.
                
                dec_loss = (l_align + l_orth + l_rec) - (n_l_align + n_l_orth + n_l_rec) * 0.001 # hard coded weight
                
                
                # reconstrct comb_neg
                comb_neg = torch.cat([nt_s + nt_p, na_s + na_p, nv_s + nv_p], dim=1)
                
                comb_pos = torch.cat([t_s + t_p, a_s + a_p, v_s + v_p], dim=1)
                
                
                scores_neg = self.stat_net(comb_neg)
                w_div = -div_metric(scores_pos, scores_neg, mode="w") 
                self.stat_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()
                # w_div.backward()
                total = w_div + dec_loss
                total.backward()
                # NOTE not sure how dec net should be updated
                self.stat_optimizer.step()
                self.dec_optimizer.step()
                loss_hist.append(w_div.item())
                
                
                
            
        return torch.mean(torch.tensor(loss_hist)) if len(loss_hist) > 0 else 0.0
            
            


    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) # NOTE: for simplicity, add decouple network's parameter here
        self.stat_optimizer = optim.SGD(self.stat_net.parameters(), lr=self.args.learning_rate)
        self.dec_optimizer = optim.SGD(self.decouple_net.parameters(), lr=self.args.learning_rate)
        self.decouple_net.to(self.args.device)
        self.stat_net.to(self.args.device)
        
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:  
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            
            self.stat_net.train()
            stat_loss = self.train_stat( dataloader['train'], train_steps= 0.3 * len(dataloader['train']))
            self.model.train()
            logger.info(f"STAT LOSS: {stat_loss}")
            self.train_text_list = []
            self.train_audio_list = []
            self.train_vision_list = []
            
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        self.optimizer.zero_grad()
                        # NOTE not sure how to update dec 
                        self.dec_optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    self.train_audio_list.extend(audio)
                    self.train_text_list.extend(text)
                    self.train_vision_list.extend(vision)
                    if self.args.get("data_missing"):
                        text_m = batch_data['text_m'].to(self.args.device)
                        text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                        audio_m = batch_data['audio_m'].to(self.args.device)
                        audio_mask = batch_data['audio_mask'].to(self.args.device)
                        audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                        vision_m = batch_data['vision_m'].to(self.args.device)
                        vision_mask = batch_data['vision_mask'].to(self.args.device)
                        vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device) 
                        # simply replace input with corrupted data
                        text = text_m
                        audio = audio_m
                        vision = vision_m
                        
                    elif self.args.get("data_dropping"):
                        text_d = batch_data['text_d'].to(self.args.device)
                        text_dropping_mask = batch_data['text_dropping_mask'].to(self.args.device)
                        vision_d = batch_data['vision_d'].to(self.args.device)
                        vision_mask = batch_data['vision_mask'].to(self.args.device)
                        vision_dropping_mask = batch_data['vision_dropping_mask'].to(self.args.device)
                        audio_d = batch_data['audio_d'].to(self.args.device)
                        audio_mask = batch_data['audio_mask'].to(self.args.device)
                        audio_dropping_mask = batch_data['audio_dropping_mask'].to(self.args.device)
                        text = text_d
                        audio = audio_d
                        vision = vision_d
       
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                # forward
                    text_neg, audio_neg, vision_neg = create_fake_samples(text, audio, vision)
                    text_neg, audio_neg, vision_neg = text_neg.to(self.args.device), audio_neg.to(self.args.device), vision_neg.to(self.args.device)
                    # with torch.no_grad():
                    pos_outputs = self.model(text, audio, vision)
                    
                    t_pos, v_pos, a_pos = pos_outputs['Feature_t'], pos_outputs['Feature_v'], pos_outputs['Feature_a']
                    
                    comb_pos = torch.cat([t_pos, a_pos, v_pos], dim=1)
                    neg_outputs = self.model(text_neg, audio_neg, vision_neg)
                    t_neg, v_neg, a_neg = neg_outputs['Feature_t'], neg_outputs['Feature_v'], neg_outputs['Feature_a']
                    
                    
                    
                    # NOTE: Add decouple Network
                    t_s, a_s, v_s, t_p, a_p, v_p = self.decouple_net(t_pos, a_pos, v_pos)
                    
                    nt_s, na_s, nv_s, nt_p, na_p, nv_p = self.decouple_net(t_neg, a_neg, v_neg)
                    
                    
                    # NOTE loss: 
                    l_align = alignment_loss(t_s, a_s, v_s)
                    l_orth = ( orth_loss(t_s, t_p) + orth_loss(a_s, a_p) + orth_loss(v_s, v_p) ) / 3.
                    
                    l_rec = (rec_loss(t_s, t_p, t_pos) + rec_loss(a_s, a_p, a_pos) + rec_loss(v_s, v_p, v_pos) ) / 3.
                    
                    n_l_align = alignment_loss(nt_s, na_s, nv_s)
                    n_l_orth = ( orth_loss(nt_s, nt_p) + orth_loss(na_s, na_p) + orth_loss(nv_s, nv_p) ) / 3.
                    n_l_rec = (rec_loss(nt_s, nt_p, t_neg) + rec_loss(na_s, na_p, a_neg) + rec_loss(nv_s, nv_p, v_neg) ) / 3.
                    
                    dec_loss = (l_align + l_orth + l_rec) - (n_l_align + n_l_orth + n_l_rec) * 0.001 # hard coded weight
                    
                    
                    # reconstrct comb_neg
                    comb_neg = torch.cat([nt_s + nt_p, na_s + na_p, nv_s + nv_p], dim=1)
                    
                    comb_pos = torch.cat([t_s + t_p, a_s + a_p, v_s + v_p], dim=1)
                    
                    
                    
                    
                    scores_pos = self.stat_net(comb_pos)

                    comb_neg = torch.cat([t_neg, a_neg, v_neg], dim=1)
                    scores_neg = self.stat_net(comb_neg)
                    w_div = div_metric(scores_pos, scores_neg, mode="w") * (-self.lambda_)
                    # wrap these shits to a function in stat ne
                    outputs = pos_outputs['M']
                    # compute loss
                    loss = self.criterion(outputs, labels)  + w_div # make it a hyperparameter later # NOTE: add decouple loss
                    
                    # backward
                    loss.backward()
                    dec_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.args.grad_clip)
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        self.optimizer.step()
                        self.dec_optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    self.optimizer.step()
                    self.dec_optimizer.step()
                    
            # convert train_*_list to tensor
            self.train_text_list = torch.stack(self.train_text_list, dim=0).to(self.args.device)
            self.train_audio_list = torch.stack(self.train_audio_list, dim=0).to(self.args.device)
            self.train_vision_list = torch.stack(self.train_vision_list, dim=0).to(self.args.device)
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(self.model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(self.model.cpu().state_dict(), self.args.model_save_path)
                self.model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(self.model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            
            break

    def detect_missing_modality_by_zero_out(
            self,
            model, stat_net,
            text_tokens, audio_tokens, vision_tokens,
            device
    ):
        """
        返回: missing_modality (str), 可能是 'text'/'audio'/'vision'
        思路:
          1) 跑一次全模态 => 得到 baseline scores
          2) 分别将 text/audio/vision 置零 => 跑模型 => score
             下降最小 => 说明该模态贡献最少 => 视为缺失模态
        """
        # baseline forward
        with torch.no_grad():
            pos_outputs = model(text_tokens, audio_tokens, vision_tokens)
            # 拼接 (Feature_t, Feature_a, Feature_v) NOTE: modifies
            rep_t = pos_outputs['Feature_t']
            rep_a = pos_outputs['Feature_a']
            rep_v = pos_outputs['Feature_v']
            t_s, a_s, v_s, t_p, a_p, v_p = self.decouple_net(rep_t, rep_a, rep_v)
            comb_full = torch.cat([t_s + t_p , a_s + a_p, v_s + v_p], dim=1)
            
            # comb_full = torch.cat([rep_t, rep_a, rep_v], dim=1)
            score_full = stat_net(comb_full).mean().item()  # 也可 batch内逐条对比

        # 函数: 用某模态置零再跑
        def forward_with_zero(modality: str):
            if modality == "text":
                z_text = torch.zeros_like(text_tokens).to(device)
                out = model(z_text, audio_tokens, vision_tokens)
            elif modality == "audio":
                z_audio = torch.zeros_like(audio_tokens).to(device)
                out = model(text_tokens, z_audio, vision_tokens)
            else:  # "vision"
                z_vision = torch.zeros_like(vision_tokens).to(device)
                out = model(text_tokens, audio_tokens, z_vision)
                
            # NOTE modified
            rep_t2 = out['Feature_t']
            rep_a2 = out['Feature_a']
            rep_v2 = out['Feature_v']
            t_s2, a_s2, v_s2, t_p2, a_p2, v_p2 = self.decouple_net(rep_t2, rep_a2, rep_v2)
            comb2 = torch.cat([t_s2 + t_p2 , a_s2 + a_p2, v_s2 + v_p2], dim=1)
            
            
            # comb2 = torch.cat([rep_t2, rep_a2, rep_v2], dim=1)
            score2 = stat_net(comb2).mean().item()
            return score2

        # 分别置零 text/audio/vision
        score_text_zero = forward_with_zero("text")
        score_audio_zero = forward_with_zero("audio")
        score_vision_zero = forward_with_zero("vision")

        drop_text = score_full - score_text_zero
        drop_audio = score_full - score_audio_zero
        drop_vision = score_full - score_vision_zero

        # 哪个drop最小 => 贡献度最小 => 缺失最严重
        deltas = {"text": drop_text, "audio": drop_audio, "vision": drop_vision}
        missing_modality = min(deltas, key=deltas.get)
        return missing_modality

    def compute_cosine_sim_get_k(self, query: torch.Tensor, train_tokens_list: torch.Tensor,K=10, device='cpu'):
        """
        对 query与train_tokens_list 的余弦相似度,
        都做 mean-pooling + 归一化 => 返回 Top——K
        """
        query = query.to(device)  # [1, seq_len, emb_dim] or shape
        train_tokens_list = train_tokens_list.to(device)  # [N, seq_len, emb_dim]

        # mean pooling
        q_vec = query.mean(dim=1, keepdim=False)  # [1, emb_dim]
        t_vecs = train_tokens_list.mean(dim=1)  # [N, emb_dim]

        q_norm = q_vec / (q_vec.norm(dim=1, keepdim=True) + 1e-9)  # [1, emb_dim]
        t_norm = t_vecs / (t_vecs.norm(dim=1, keepdim=True) + 1e-9)  # [N, emb_dim]

        # cos sim
        cos_sim = (t_norm @ q_norm.transpose(0, 1)).squeeze(-1)  # [N]
        # (3) 取 topK
        topk_vals, topk_indices = torch.topk(cos_sim, K, largest=True)
        return topk_indices.cpu().tolist(),cos_sim



    def retrieve_topk_for_missing_modality(
            self,
            missing_modality: str,
            test_audio_tokens: torch.Tensor,
            test_vision_tokens: torch.Tensor,
            train_audio_list: torch.Tensor,
            train_vision_list: torch.Tensor,
            K=10,
            device='cpu'
    ):
        """
        若 missing_modality == 'text':
           1) 分别计算 test_audio_tokens 与 train_audio_list 的相似度
           2) 分别计算 test_vision_tokens 与 train_vision_list 的相似度
           3) 将两者加和后 取 topK

        若 missing_modality == 'audio':

           只做 test_audio_tokens 与 train_audio_list 的对比, 取 topK
        若 missing_modality == 'vision':

           只做 test_vision_tokens 与 train_vision_list 的对比, 取 topK

        返回: topk_indices (list[int]) => 最相似的训练集索引
        """

        # 如果不是 'text' 缺失, 只做单一路 token 相似度
        if missing_modality == 'vision':
            topk_idx, sim_a = self.compute_cosine_sim_get_k(test_audio_tokens, train_audio_list, K=K, device=device)
            return topk_idx

        if missing_modality == 'audio':
            topk_idx, sim_v = self.compute_cosine_sim_get_k(test_vision_tokens, train_vision_list, K=K, device=device)
            return topk_idx

        # 如果 missing_modality == 'text'
        # => 先分别对 audio, vision 做相似度, 再加和
        with torch.no_grad():
            idx_a, audio_sim = self.compute_cosine_sim_get_k(test_audio_tokens, train_audio_list, device=device)  # shape [N]
            idx_v, vision_sim = self.compute_cosine_sim_get_k(test_vision_tokens, train_vision_list, device=device)  # shape [N]
            combined_sim = audio_sim + vision_sim  # [N]
            topk_vals, topk_indices = torch.topk(combined_sim, k=K, largest=True)
            return topk_indices.cpu().tolist()


    def replace_missing_modality_and_score_topk(
            self,
            model, stat_net,
            missing_modality: str,
            text_tokens, audio_tokens, vision_tokens,
            train_text_list, train_audio_list, train_vision_list,
            device,
            K=10
    ):
        """
        1) retrieve_topk_for_missing_modality => indices
        2) 遍历 indices => 替换 => forward => score
        3) 返回 best_text, best_audio, best_vision
        """
        best_score = -float("inf")

        # baseline: 保持原 token
        best_text = text_tokens.clone()
        best_audio = audio_tokens.clone()
        best_vision = vision_tokens.clone()

        # step1: topK candidate
        topk_indices = self.retrieve_topk_for_missing_modality(
            missing_modality,
            audio_tokens, vision_tokens,
            train_audio_list, train_vision_list,
            K=K, device='cpu'
        )

        # step2: 遍历 topK
        for i in topk_indices:
            cand_text = train_text_list[i].to(device).unsqueeze(0)  # shape [seq_len, emb_dim]
            cand_audio = train_audio_list[i].to(device).unsqueeze(0)
            cand_vision = train_vision_list[i].to(device).unsqueeze(0)

            if missing_modality == "text":
                tmp_text = cand_text
                tmp_audio = audio_tokens
                tmp_vision = vision_tokens
            elif missing_modality == "audio":
                tmp_text = text_tokens
                tmp_audio = cand_audio
                tmp_vision = vision_tokens
            else:  # missing_modality == "vision"
                tmp_text = text_tokens
                tmp_audio = audio_tokens
                tmp_vision = cand_vision

            # forward => stat_net
            with torch.no_grad():
                out = model(tmp_text, tmp_audio, tmp_vision)
                rep_t = out['Feature_t']
                rep_a = out['Feature_a']
                rep_v = out['Feature_v']
                # NOTE modified
                t_s, a_s, v_s, t_p, a_p, v_p = self.decouple_net(rep_t, rep_a, rep_v)
                comb = torch.cat([t_s + t_p, a_s + a_p, v_s + v_p], dim=1)
                # comb = torch.cat([rep_t, rep_a, rep_v], dim=1)
                score_val = stat_net(comb).mean().item()

            if score_val > best_score:
                best_score = score_val
                if missing_modality == "text":
                    best_text = tmp_text.clone()
                    best_audio = audio_tokens
                    best_vision = vision_tokens
                elif missing_modality == "audio":
                    best_text = text_tokens
                    best_audio = tmp_audio.clone()
                    best_vision = vision_tokens
                else:
                    best_text = text_tokens
                    best_audio = audio_tokens
                    best_vision = tmp_vision.clone()

        return best_text, best_audio, best_vision, best_score

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        self.model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        # 事先加载 训练集 tokens list
        train_text_list = self.train_text_list
        train_audio_list = self.train_audio_list
        train_vision_list = self.train_vision_list

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text_tokens = batch_data['text'].to(self.args.device)  # shape [batch_size, ...]
                    audio_tokens = batch_data['audio'].to(self.args.device)
                    vision_tokens = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)

                    batch_size = labels.size(0)
                    # 针对 batch 内每个样本分别处理
                    for i in range(batch_size):
                        # (1) 拿到单条样本 tokens
                        text_i = text_tokens[i].unsqueeze(0)  # shape [seq_len, emb_dim]
                        audio_i = audio_tokens[i].unsqueeze(0)
                        vision_i = vision_tokens[i].unsqueeze(0)
                        label_i = labels[i].unsqueeze(0)  # shape [1] or [1, num_classes]

                        # step1: 用 zero_out方式检测得到缺失模态
                        missing_modality = self.detect_missing_modality_by_zero_out(
                            self.model, self.stat_net,
                            text_i, audio_i, vision_i,
                            self.args.device
                        )
                        # step2: Top_k 检索替换
                        best_text, best_audio, best_vision, best_score = self.replace_missing_modality_and_score_topk(
                            self.model, self.stat_net,
                            missing_modality,
                            text_i, audio_i, vision_i,
                            train_text_list, train_audio_list, train_vision_list,
                            self.args.device
                        )

                        # step3: 用替换后的 tokens 做下游预测
                        #    这里再跑一次 forward
                        out_best = self.model(best_text, best_audio, best_vision)
                        pred_best = out_best['M']  # [1, 1 or #class]
                        loss_i = self.criterion(pred_best, label_i)
                        eval_loss += loss_i.item()

                        y_pred.append(pred_best.cpu())
                        y_true.append(label_i.cpu())

        eval_loss /= len(dataloader)
        pred_tensor = torch.cat(y_pred, dim=0)
        true_tensor = torch.cat(y_true, dim=0)

        eval_results = self.metrics(pred_tensor, true_tensor)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {eval_results}")

        return eval_results
