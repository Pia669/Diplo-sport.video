from transformers import SiglipVisionModel
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as checkpoint

import Dataset

SEGMENT_SIZE = 20
SPATIAL_EMBEDDING_OUT_DIM = 128
NUM_HEADS = 8
NUM_SPATIOTEMP_BLOCK = 16
CUDA = False


class NeuralNet(nn.Module):
    def __init__(self, keywords=None, feature_dim=768, num_transformer_layers=2,
                  transformer_heads=8, classifier_transformer_type = "avg_pool",
                 vision_encoder_type = "spatial_and_temporal", use_transformer = True,
                 model_name = "google/siglip-base-patch16-224"):
        super(NeuralNet, self).__init__()

        self.keywords = keywords

        self.siglip_model = VisionTimesformer(patch_size=16, model_name=model_name, width=768, layers=12, heads=12,
                                              output_dim=feature_dim, input_resolution=224,
                                              encoder_type=vision_encoder_type)
        self.classifier_ln1 = nn.LayerNorm(feature_dim)
        self.classifier_ln2 = nn.LayerNorm(feature_dim)
        self.classifier_transformer_type = classifier_transformer_type
        self.use_transformer = use_transformer

        if self.classifier_transformer_type == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        if self.use_transformer:
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=transformer_heads,
                dim_feedforward=feature_dim * 4,
                dropout=0.,
                activation='relu'
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer,
                num_layers=num_transformer_layers
            )

        self.classifier = nn.Linear(feature_dim, len(self.keywords))

    def forward(self, x):
        B, _, _, _, _ = x.shape
        x = self.siglip_model(x)
        x = self.classifier_ln1(x)

        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_transformer:
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)
            if self.classifier_transformer_type == "cls_token":
                x = x[0, :, :]  # cls_token是第一个元素
            elif self.classifier_transformer_type == "avg_pool":
                x = x.mean(dim=0)
        else:
            x = x.mean(dim=1)

        x = self.classifier_ln2(x)
        logits = self.classifier(x)
        return logits


class ResidualAttentionBlock(nn.Module):
    def __init__(self, res_idx, d_model=768, n_head=12, drop_path=0., attn_mask=None, dropout=0.,
                 attention_type='divided_space_time', model_name="google/siglip-base-patch16-224"):
        super().__init__()
        model = SiglipVisionModel.from_pretrained(model_name)
        vision_model = model.vision_model

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')

        # Temporal Attention Parameters
        if attention_type == 'divided_space_time':
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.temporal_fc = nn.Linear(d_model, d_model)
            self.register_parameter('temporal_alpha_attn', nn.Parameter(torch.tensor(0.)))

        self.encoder = vision_model.encoder.layers[res_idx]
        self.attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x)[0]

    def temporal_attention(self, x):
        return self.temporal_attn(x, x, x)[0]

    def forward(self, x, B, T):
        # divided_space_time

        ## Temporal
        xt = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        res_temporal = self.drop_path(self.temporal_attention(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + self.temporal_alpha_attn.tanh() * res_temporal  # 180 196 768

        ## Spatial
        xs = xt  # always 180 196 768
        res_spatial = self.encoder(xs, self.attn_mask)[0]

        return res_spatial


class Timesformer(nn.Module):
    def __init__(self, width, layers, heads, model_name, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(
                ResidualAttentionBlock(d_model=width, n_head=heads, res_idx=idx, drop_path=dpr[idx], dropout=dropout,
                                       model_name=model_name))
        self.checkpoint_num = checkpoint_num

    def forward(self, x, B, T):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, B, T)
        return x


class VisionTimesformer(nn.Module):
    def __init__(
            self, output_dim=768, num_frames=20,
            input_resolution=224, patch_size=16, width=768,
            layers=12, heads=12,
            encoder_type="spatial_and_temporal",
            model_name="google/siglip-base-patch16-224"
    ):
        super().__init__()

        self.num_frames = num_frames
        model = SiglipVisionModel.from_pretrained(model_name)

        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.encoder_type = encoder_type
        self.patch_size = patch_size
        self.width = width

        if self.encoder_type == "spatial_only":
            self.vision_model = model

        elif self.encoder_type == "spatial_and_temporal":
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

            vision_model = model.vision_model
            self.vision_model_embedding = vision_model.embeddings
            self.timesformer = Timesformer(width=width, layers=layers, heads=heads, model_name=model_name)
            self.post_layernorm = vision_model.post_layernorm
            self.head = vision_model.head

    def get_num_layers(self):
        return len(self.timesformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_positional_embedding'}

    def forward(self, x):
        B, T, _, _, _ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        if self.encoder_type == "spatial_only":
            x = self.vision_model(x)['pooler_output']
            x = rearrange(x, "(b t) m -> b t m", b=B)  # 6 30 768

        elif self.encoder_type == "spatial_and_temporal":
            x = self.vision_model_embedding(x)  # 180 196 768
            x = rearrange(x, "(b t) n m -> b n t m", b=B, t=T)
            x = x + self.temporal_positional_embedding
            x = rearrange(x, "b n t m -> (b t) n m")  # 180 196 768
            x = self.timesformer(x, B, T)  # 180 196 768
            x = self.post_layernorm(x)
            x = self.head(x)  # 180 768
            x = rearrange(x, "(b t) m -> b t m", b=B, t=T)  # 6 30 768

        return x


if __name__ == '__main__':
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    net = NeuralNet(
        keywords=['1goal', '2chance', '3special']
        )
    print('NN')

    test_batch_size = 1
    dataset = Dataset.VideoDataset('test.json')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    print("dataset")

    for images, labels in test_loader:
        output = net(images)

        print('output type: {}, expected: {}'.format(output, labels))

        break
