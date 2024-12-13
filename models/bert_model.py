import numpy as np
import torch
import os

import torchvision
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertTokenizer, AutoModel, AutoConfig

from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x, aux_imgs=None):
        # full image prompt
        prompt_guids = self.get_resnet_prompt(x)  # 4x[bsz, 256, 7, 7]

        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224
        if aux_imgs is not None:
            aux_prompt_guids = []  # goal: 3 x (4 x [bsz, 256, 7, 7])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i])  # 4 x [bsz, 256, 7, 7]
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)  # (bsz, 256, 56, 56)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)  # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)  # conv2: (bsz, 256, 7, 7)
        return prompt_guids


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x


class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.tokenizer = BertTokenizer.from_pretrained("D:/zyk/pretrainModel/bert-base-uncased",
        #                                                do_lower_case=True)
        self.max_seq = args.max_seq
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len

        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=3840, out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=4 * 2 * 768)
            )
            self.gates = nn.ModuleList([nn.Linear(4 * 768 * 2, 4) for i in range(12)])

        self.num_labels = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.change_img = nn.Linear(3 * 224 * 224, self.max_seq)
        self.fc_1 = nn.Linear(self.bert.config.hidden_size + 200, self.num_labels)
        self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.fc_3 = nn.Linear(177 * self.bert.config.hidden_size, 1062)
        self.fc_fusion = nn.Linear(self.bert.config.hidden_size * 2, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size + 200, self.bert.config.hidden_size + 200)

        self.encoder_t = AutoModel.from_pretrained(args.bert_name)
        self.hid_dim_t = AutoConfig.from_pretrained(args.bert_name).hidden_size

        self.encoder_v = getattr(torchvision.models, args.encoder_v)()
        self.encoder_v.load_state_dict(torch.load(args.v_models_path))
        self.hid_dim_v = self.encoder_v.fc.in_features

        self.proj = nn.Linear(self.hid_dim_v, self.hid_dim_t)
        self.aux_head = nn.Linear(self.hid_dim_t, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                aug_tokens=None):
        no_prompt_text = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   past_key_values=None,
                                   return_dict=True)  # (bsz, max_seq, 768)
        bsz = attention_mask.size(0)
        if self.args.use_prompt:
            visual_prompt, addFusion_prompt, text_prompt = self.get_visual_prompt(images, aux_imgs, input_ids,
                                                                                  attention_mask,
                                                                                  token_type_ids,
                                                                                  no_prompt_text['last_hidden_state'])
            prompt_guids_length = visual_prompt[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

            addprompt_guids_length = addFusion_prompt[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, addprompt_guids_length)).to(self.args.device)
            addprompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            addprompt_attention_mask = attention_mask
            visual_prompt = None
            text_prompt = None
            addFusion_prompt = None

        visual_prompt_text = self.bert(input_ids=input_ids,
                                       attention_mask=prompt_attention_mask,
                                       token_type_ids=token_type_ids,
                                       past_key_values=visual_prompt,
                                       return_dict=True)

        # addFusion_prompt_text = self.bert(input_ids=input_ids,
        #                                   attention_mask=addprompt_attention_mask,
        #                                   token_type_ids=token_type_ids,
        #                                   past_key_values=addFusion_prompt,
        #                                   return_dict=True)
        if self.args.cross_fusion:
            visual_embeds = resnet_encode(self.encoder_v, images)  # images (bsz, 3, 224, 224)
            visual_embeds = self.proj(visual_embeds)  # (bsz, 49, 768)
            textprompt_guids_length = text_prompt[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, textprompt_guids_length)).to(self.args.device)
            visual_mask = torch.ones((bsz, visual_embeds.shape[1]), dtype=attention_mask.dtype, device=self.device)
            textprompt_attention_mask = torch.cat((prompt_guids_mask, visual_mask), dim=1)
            visual_type_ids = torch.ones((bsz, visual_embeds.shape[1]), dtype=token_type_ids.dtype, device=self.device)

            text_prompt_visual = self.bert(inputs_embeds=visual_embeds,  # (8, 49, 768)
                                           attention_mask=textprompt_attention_mask,
                                           token_type_ids=visual_type_ids,
                                           past_key_values=text_prompt,
                                           return_dict=True)
            text_prompt_visual_features = text_prompt_visual['last_hidden_state'].repeat_interleave(3, 1)

            # cross_fusion
            sequence_output = torch.cat(
                [visual_prompt_text['last_hidden_state'], text_prompt_visual_features],
                dim=-1)  # bsz, len, hidden
        else:
            # HVPNeT
            sequence_output = visual_prompt_text['last_hidden_state']

        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden

        if self.args.cross_fusion:
            emissions = self.fc_fusion(sequence_output)  # bsz, len, labels
        else:
            # HVPNeT
            emissions = self.fc_2(sequence_output)

        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, images, aux_imgs, input_ids, attention_mask, token_type_ids, encoded_text):

        # similar
        if self.args.use_similar:
            first_outputs = self._bert_forward_with_image(images, input_ids, attention_mask, token_type_ids)
            feats = first_outputs.last_hidden_state[:, 0]
            logits = self.aux_head(feats)  # 分类器 self.aux_head = nn.Linear(hid_dim_t, 2)
            # get R关系矩阵
            gate_signal = F.softmax(logits, dim=1)[:, 1].view(images.size(0), 1, 1, 1)

            images = images * gate_signal  # images.shape([8, 3, 224, 224])     gate.shape([8, 1, 1, 1])

        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)  # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in
                            aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in
                            aux_prompt_guids]  # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768 * 2, dim=-1)  # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768 * 2, dim=-1) for aux_prompt_guid in
                                  aux_prompt_guids]  # 3x [4 x [bsz, 4, 768*2]]

        replace_fusion_result = []
        text_result = []
        add_fusion_result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4  # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []  # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4  # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1),
                                                             split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)

            replace_key_val = aux_key_vals  # K'  [ (bsz, 4, 768*2) * 3 ]
            replace_key_val = torch.cat(replace_key_val, dim=1)
            replace_key_val = replace_key_val.split(768, dim=-1)
            key, value = replace_key_val[0].reshape(bsz, 12, -1, 64).contiguous(), replace_key_val[1].reshape(bsz, 12,
                                                                                                              -1,
                                                                                                              64).contiguous()  # bsz, 12, -1, 64
            temp_dict = (key, value)
            replace_fusion_result.append(temp_dict)

            add_key_val = [key_val] + aux_key_vals  # K' + K    [ (bsz, 4, 768*2) * 4 ]
            add_key_val = torch.cat(add_key_val, dim=1)
            add_key_val = add_key_val.split(768, dim=-1)
            key, value = add_key_val[0].reshape(bsz, 12, -1, 64).contiguous(), add_key_val[1].reshape(bsz, 12, -1,
                                                                                                      64).contiguous()
            temp_dict = (key, value)
            add_fusion_result.append(temp_dict)

            # 构造文本K、V矩阵
            # encoded_text.shape (bsz, max_seq, 768)
            text_key_val = encoded_text.split(int(self.max_seq / 2), dim=1)
            text_key, text_value = text_key_val[0].reshape(bsz, 12, -1, 64).contiguous(), text_key_val[1].reshape(bsz,
                                                                                                                  12,
                                                                                                                  -1,
                                                                                                                  64).contiguous()
            temp_dict = (text_key, text_value)
            text_result.append(temp_dict)
        return replace_fusion_result, add_fusion_result, text_result

    def _bert_forward_with_image(self, images, input_ids, attention_mask, token_type_ids, gate_signal=None):
        # 得到第一次联合编码后的输出
        textual_embeds = self.encoder_t.embeddings.word_embeddings(input_ids)
        visual_embeds = resnet_encode(self.encoder_v, images)  # images (bsz, 3, 224, 224)
        visual_embeds = self.proj(visual_embeds)

        if gate_signal is not None:
            visual_embeds *= gate_signal  # R * V

        inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)  # 拼接文本和图片

        batch_size = visual_embeds.size()[0]
        visual_length = visual_embeds.size()[1]

        attention_mask = attention_mask
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = token_type_ids
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
