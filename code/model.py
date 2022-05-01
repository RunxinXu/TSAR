import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers.modeling_utils import PreTrainedModel
import dgl
import dgl.nn.pytorch as dglnn


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)
        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class MyBertmodel(BertPreTrainedModel):
    def __init__(self, config, gcn_layers=3, lambda_boundary=0, event_embedding_size=200):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.len_embedding = nn.Embedding(config.len_size, config.len_dim)
        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim + event_embedding_size, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        self.global_gate = nn.Linear(config.hidden_size, 1)
        self.local_gate = nn.Linear(config.hidden_size, 1)

        # GRAPH
        self.gcn_layers = gcn_layers
        self.rel_name_lists = [str(i) for i in range(13)]
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(config.hidden_size, config.hidden_size, self.rel_name_lists,
                                num_bases=len(self.rel_name_lists), activation=activation_func, self_loop=True, dropout=config.hidden_dropout_prob * 3)
                                for i in range(self.gcn_layers)])
        self.middle_layer = nn.Sequential(
            nn.Linear(config.hidden_size * (self.gcn_layers+1), config.hidden_size),
            activation_func,
            nn.Dropout(config.hidden_dropout_prob)    
        )

        # boundary
        self.lambda_boundary = lambda_boundary
        if self.lambda_boundary > 0:
            self.start_classifier = nn.Linear(config.hidden_size, 2)
            self.end_classifier = nn.Linear(config.hidden_size, 2)

        # positive weight
        pos_loss_weight = getattr(config, 'pos_loss_weight', None)
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)])
        self.pos_loss_weight[0] = 1

        self.init_weights()

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        event_ids=None,
        labels=None,
        spans=None,
        span_lens=None,
        label_masks=None,
        trigger_index=None,
        subwords_snt2spans=None,
        subwords_span2snts=None,
        trigger_snt_ids=None,
        belongingsnts=None,
        graphs=None,
        start_labels=None,
        end_labels=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # ================= GLOBAL =================
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        bsz, seq_len, hidsize = last_hidden_state.size()
        span_num = spans.size(1)
        # ================= GLOBAL =================

        # ============== GLOBAL GRAPH ===============
        all_graphs = []
        all_span_infos = []
        all_node_features = []
        SNT_EDGE_TYPE='6'

        for example_idx, graph_list in enumerate(graphs):
            span_info = [] # node_num * 2
            cur_big_graph = []
            LL = [0]
            for g in graph_list:
                g = g.to(last_hidden_state.device)
                cur_big_graph.append(g)
                span_info.append(g.ndata['span'])
                LL.append(g.ndata['span'].size(0)+LL[-1])
            span_info = torch.cat(span_info, dim=0)
            node_num = span_info.size(0)
            all_span_infos.append(span_info)

            # fully connect root node with SNT_EDGE_TYPE edge
            cur_big_graph = dgl.batch(cur_big_graph)
            LL = LL[:-1]
            for root_i in LL:
                for root_j in LL:
                    if root_i != root_j:
                        cur_big_graph.add_edges(u=root_i, v=root_j, etype=SNT_EDGE_TYPE)
            all_graphs.append(cur_big_graph)

            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1).to(last_hidden_state)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask, last_hidden_state[example_idx]) / graph_span_mask_num   # node_num * hid_size
            all_node_features.append(node_feature)
        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.middle_layer(feature_bank) # all_node_num * hidden_size

        # put them back
        cur_bias = 0
        all_global_graph_feature = []
        for cur_span_info in all_span_infos:
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias+cur_node_num] # node_num * hidden_size
            cur_bias += cur_node_num
            # cur_span_info: node_num * 2
            # cur_features_bank: node_num * hidden_size
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(cur_node_num, 1).to(last_hidden_state)
            graph_span_mask = (graph_span_mask >= cur_span_info[:, 0:1]) & (graph_span_mask <= cur_span_info[:, 1:])
            graph_span_mask = graph_span_mask.t()
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            global_graph_feature = torch.mm(graph_span_mask, cur_features_bank) / graph_span_mask_num
            all_global_graph_feature.append(global_graph_feature.unsqueeze(0))
        final_global_graph_feature = torch.cat(all_global_graph_feature, dim=0)
        # ============== GLOBAL GRAPH ===============

        # ================= LOCAL ==================
        token2sentspan = torch.gather(input=subwords_snt2spans, dim=1, index=belongingsnts.unsqueeze(-1).expand(-1, -1, 2))
        x = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, seq_len, 1).to(last_hidden_state)
        tokenmask = (x>=token2sentspan[:,:,0:1]) & (x<=token2sentspan[:,:,1:])
        trigger2sentspan = torch.gather(input=subwords_snt2spans, dim=1, index=trigger_snt_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
        x = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1).to(last_hidden_state)
        triggermask = (x>=trigger2sentspan[:,0:1]) & (x<=trigger2sentspan[:,1:])
        triggermask = triggermask.unsqueeze(-1).expand(-1, seq_len, -1)
        focusmask = tokenmask | triggermask
        focusmask[:, 0, :] = attention_mask
        focus = self.bert(
            input_ids,
            attention_mask=focusmask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        focus = focus[0]
        focus = self.dropout(focus)
        # ================= LOCAL ==================

        # ============== LOCAL GRAPH ===============
        all_graphs = []
        all_span_infos = []
        all_node_features = []

        for example_idx, graph_list in enumerate(graphs):
            span_info = [] # node_num * 2
            for g in graph_list:
                g = g.to(focus.device)
                all_graphs.append(g)
                span_info.append(g.ndata['span'])
            span_info = torch.cat(span_info, dim=0)
            node_num = span_info.size(0)
            all_span_infos.append(span_info)
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1).to(focus)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask, focus[example_idx]) / graph_span_mask_num
            all_node_features.append(node_feature)
        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.middle_layer(feature_bank)

        cur_bias = 0
        all_local_graph_feature = []
        for cur_span_info in all_span_infos:
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias+cur_node_num] # node_num * hidden_size
            cur_bias += cur_node_num
            # cur_span_info: node_num * 2
            # cur_features_bank: node_num * hidden_size
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(cur_node_num, 1).to(focus)
            graph_span_mask = (graph_span_mask >= cur_span_info[:, 0:1]) & (graph_span_mask <= cur_span_info[:, 1:])
            graph_span_mask = graph_span_mask.t() # seq_len * node_num
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            local_graph_feature = torch.mm(graph_span_mask, cur_features_bank) / graph_span_mask_num
            all_local_graph_feature.append(local_graph_feature.unsqueeze(0))
        final_local_graph_feature = torch.cat(all_local_graph_feature, dim=0)
        # ============== LOCAL GRAPH ===============

        # ================= FUSION =================
        loss = None

        global_feature = last_hidden_state + final_global_graph_feature
        local_feature = focus + final_local_graph_feature
        final_gate = torch.nn.functional.sigmoid(self.global_gate(global_feature) + self.local_gate(local_feature)) # bsz * seq_len * 1
        final = final_gate * global_feature + (1-final_gate) * local_feature
        start_feature = self.transform_start(final)
        end_feature = self.transform_end(final)
        trigger_feature = self.select_single_token_rep(final, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        len_state = self.len_embedding(span_lens) # bsz * span_num * pos_size

        # span loss
        b_feature =  self.select_rep(start_feature, spans[:,:,0])
        e_feature =  self.select_rep(end_feature, spans[:,:,1])
        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(final)
        context_mask = (context>=spans[:,:,0:1]) & (context<=spans[:,:,1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)
        context_feature = torch.bmm(context_mask, final) # bsz * span_num * hidsize
        span_feature = torch.cat((b_feature, e_feature, context_feature), dim=-1)
        span_feature = self.transform_span(span_feature)

        if self.event_embedding is not None:
            logits = torch.cat((
                span_feature, trigger_feature, 
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature, 
                len_state, self.event_embedding(event_ids).unsqueeze(1).expand(-1, span_num, -1)), dim=-1
            )
        else:
            logits = torch.cat((
                span_feature, trigger_feature, 
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature, 
                len_state), dim=-1
            )
        logits = self.classifier(logits)  # bsz * span_num * num_labels
        label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1) 
        logits = logits.masked_fill(label_masks_expand==0, -1e4)
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight.to(final))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.contiguous().view(-1))

        # start/end boundary loss
        if self.lambda_boundary > 0:
            start_logits = self.start_classifier(start_feature)
            end_logits = self.end_classifier(end_feature)
            if start_labels is not None and end_labels is not None:
                loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight[:2].to(final))
                loss += self.lambda_boundary * (loss_fct(start_logits.view(-1, 2), start_labels.contiguous().view(-1)) \
                    + loss_fct(end_logits.view(-1, 2), end_labels.contiguous().view(-1))
                )

        return {
            'loss': loss,
            'logits': logits,
            'spans': spans,
        }

class MyRobertamodel(RobertaPreTrainedModel):
    def __init__(self, config, gcn_layers=3, lambda_boundary=0, event_embedding_size=200):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.len_embedding = nn.Embedding(config.len_size, config.len_dim)
        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim + event_embedding_size, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )

        self.global_gate = nn.Linear(config.hidden_size, 1)
        self.local_gate = nn.Linear(config.hidden_size, 1)

        # GRAPH
        self.gcn_layers = gcn_layers
        self.rel_name_lists = [str(i) for i in range(13)]
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(config.hidden_size, config.hidden_size, self.rel_name_lists,
                                num_bases=len(self.rel_name_lists), activation=activation_func, self_loop=True, dropout=config.hidden_dropout_prob * 3)
                                for i in range(self.gcn_layers)])
        self.middle_layer = nn.Sequential(
            nn.Linear(config.hidden_size * (self.gcn_layers+1), config.hidden_size),
            activation_func,
            nn.Dropout(config.hidden_dropout_prob)    
        )

        # boundary
        self.lambda_boundary = lambda_boundary
        if self.lambda_boundary > 0:
            self.start_classifier = nn.Linear(config.hidden_size, 2)
            self.end_classifier = nn.Linear(config.hidden_size, 2)

        # positive weight
        pos_loss_weight = getattr(config, 'pos_loss_weight', None)
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)])
        self.pos_loss_weight[0] = 1
        self.init_weights()

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        event_ids=None,
        labels=None,
        spans=None,
        span_lens=None,
        label_masks=None,
        trigger_index=None,
        subwords_snt2spans=None,
        subwords_span2snts=None,
        trigger_snt_ids=None,
        belongingsnts=None,
        graphs=None,
        start_labels=None,
        end_labels=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # ================= GLOBAL =================
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)  # bsz * seq_len * hidsize
        bsz, seq_len, hidsize = last_hidden_state.size()
        span_num = spans.size(1)
        # ================= GLOBAL =================

        # ============== GLOBAL GRAPH ===============
        all_graphs = []
        all_span_infos = []
        all_node_features = []
        SNT_EDGE_TYPE='6'

        for example_idx, graph_list in enumerate(graphs):
            span_info = [] # node_num * 2
            cur_big_graph = []
            LL = [0]
            for g in graph_list:
                g = g.to(last_hidden_state.device)
                cur_big_graph.append(g)
                span_info.append(g.ndata['span'])
                LL.append(g.ndata['span'].size(0)+LL[-1])
            span_info = torch.cat(span_info, dim=0)
            node_num = span_info.size(0)
            all_span_infos.append(span_info)

            cur_big_graph = dgl.batch(cur_big_graph)
            LL = LL[:-1]
            for root_i in LL:
                for root_j in LL:
                    if root_i != root_j:
                        cur_big_graph.add_edges(u=root_i, v=root_j, etype=SNT_EDGE_TYPE)
            all_graphs.append(cur_big_graph)

            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1).to(last_hidden_state)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask, last_hidden_state[example_idx]) / graph_span_mask_num
            all_node_features.append(node_feature)
        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.middle_layer(feature_bank)

        cur_bias = 0
        all_global_graph_feature = []
        for cur_span_info in all_span_infos:
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias+cur_node_num]
            cur_bias += cur_node_num
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(cur_node_num, 1).to(last_hidden_state)
            graph_span_mask = (graph_span_mask >= cur_span_info[:, 0:1]) & (graph_span_mask <= cur_span_info[:, 1:])
            graph_span_mask = graph_span_mask.t()
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            global_graph_feature = torch.mm(graph_span_mask, cur_features_bank) / graph_span_mask_num
            all_global_graph_feature.append(global_graph_feature.unsqueeze(0))
        final_global_graph_feature = torch.cat(all_global_graph_feature, dim=0)
        # ============== GLOBAL GRAPH ===============

        # ================= LOCAL ==================
        token2sentspan = torch.gather(input=subwords_snt2spans, dim=1, index=belongingsnts.unsqueeze(-1).expand(-1, -1, 2))
        x = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, seq_len, 1).to(last_hidden_state)
        tokenmask = (x>=token2sentspan[:,:,0:1]) & (x<=token2sentspan[:,:,1:])
        trigger2sentspan = torch.gather(input=subwords_snt2spans, dim=1, index=trigger_snt_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
        x = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1).to(last_hidden_state)
        triggermask = (x>=trigger2sentspan[:,0:1]) & (x<=trigger2sentspan[:,1:])
        triggermask = triggermask.unsqueeze(-1).expand(-1, seq_len, -1)
        focusmask = tokenmask | triggermask
        focusmask[:, 0, :] = attention_mask

        focus = self.roberta(
            input_ids,
            attention_mask=focusmask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        focus = focus[0]
        focus = self.dropout(focus)  # bsz * seq_len * hidsize
        # ================= LOCAL ==================

        # ============== LOCAL GRAPH ===============
        all_graphs = []
        all_span_infos = []
        all_node_features = []

        for example_idx, graph_list in enumerate(graphs):
            span_info = [] # node_num * 2
            for g in graph_list:
                g = g.to(focus.device)
                all_graphs.append(g)
                span_info.append(g.ndata['span'])
            span_info = torch.cat(span_info, dim=0)
            node_num = span_info.size(0)
            all_span_infos.append(span_info)
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1).to(focus)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask, focus[example_idx]) / graph_span_mask_num
            all_node_features.append(node_feature)
        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.middle_layer(feature_bank) # all_node_num * hidden_size

        cur_bias = 0
        all_local_graph_feature = []
        for cur_span_info in all_span_infos:
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias+cur_node_num]
            cur_bias += cur_node_num
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(cur_node_num, 1).to(focus)
            graph_span_mask = (graph_span_mask >= cur_span_info[:, 0:1]) & (graph_span_mask <= cur_span_info[:, 1:])
            graph_span_mask = graph_span_mask.t()
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            local_graph_feature = torch.mm(graph_span_mask, cur_features_bank) / graph_span_mask_num
            all_local_graph_feature.append(local_graph_feature.unsqueeze(0))
        final_local_graph_feature = torch.cat(all_local_graph_feature, dim=0)
        # ============== LOCAL GRAPH ===============

        # ================= FUSION =================
        loss = None

        global_feature = last_hidden_state + final_global_graph_feature
        local_feature = focus + final_local_graph_feature
        final_gate = torch.nn.functional.sigmoid(self.global_gate(global_feature) + self.local_gate(local_feature)) # bsz * seq_len * 1
        final = final_gate * global_feature + (1-final_gate) * local_feature
        start_feature = self.transform_start(final)
        end_feature = self.transform_end(final)
        trigger_feature = self.select_single_token_rep(final, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        len_state = self.len_embedding(span_lens)

        # span loss
        b_feature =  self.select_rep(start_feature, spans[:,:,0])
        e_feature =  self.select_rep(end_feature, spans[:,:,1])
        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(final)
        context_mask = (context>=spans[:,:,0:1]) & (context<=spans[:,:,1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)
        context_feature = torch.bmm(context_mask, final)
        span_feature = torch.cat((b_feature, e_feature, context_feature), dim=-1)
        span_feature = self.transform_span(span_feature)

        if self.event_embedding is not None:
            logits = torch.cat((
                span_feature, trigger_feature, 
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature, 
                len_state, self.event_embedding(event_ids).unsqueeze(1).expand(-1, span_num, -1)), dim=-1
            )
        else:
            logits = torch.cat((
                span_feature, trigger_feature, 
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature, 
                len_state), dim=-1
            )
        logits = self.classifier(logits)  # bsz * span_num * num_labels
        label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1) 
        logits = logits.masked_fill(label_masks_expand==0, -1e4)
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight.to(final))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.contiguous().view(-1))

        # startend loss
        if self.lambda_boundary > 0:
            start_logits = self.start_classifier(start_feature)
            end_logits = self.end_classifier(end_feature)
            if start_labels is not None and end_labels is not None:
                loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight[:2].to(final))
                loss += self.lambda_boundary * (loss_fct(start_logits.view(-1, 2), start_labels.contiguous().view(-1)) \
                    + loss_fct(end_logits.view(-1, 2), end_labels.contiguous().view(-1))
                )

        return {
            'loss': loss,
            'logits': logits,
            'spans': spans,
        }
        