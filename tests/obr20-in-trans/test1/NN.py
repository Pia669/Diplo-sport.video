import torch
import torch.nn as nn
import torchvision.models as models

import Dataset

SEGMENT_SIZE = 20
SPATIAL_EMBEDDING_OUT_DIM = 256
NUM_HEADS = 8
NUM_SPATIOTEMP_BLOCK = 8
CUDA = False


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        self.embedding = VideoEmbedding()

        self.spatioTemp = nn.Sequential()
        for i in range(NUM_SPATIOTEMP_BLOCK):
            self.spatioTemp.append(SpatialAttention(NUM_HEADS))
            self.spatioTemp.append(TempAttention(NUM_HEADS))

        self.agr1 = MultiHeadAttentionPooling(SPATIAL_EMBEDDING_OUT_DIM, 16)
        self.agr2 = MultiHeadAttentionPooling(SPATIAL_EMBEDDING_OUT_DIM, 1)

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        self.fc1 = nn.Linear(5120, 2560)
        self.fc2 = nn.Linear(2560, 1280)
        self.fc3 = nn.Linear(1280, 512)
        self.fc4 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.embedding(x)

        #encoding end x.shape = (batch, 20, 145, 256)

        x = self.spatioTemp(x)

        #attention end x.shape = (batch, 20, 145, 256)

        batch_size, temp_size, spatial_size, embed_size = x.shape
        x = x.reshape(-1, spatial_size, SPATIAL_EMBEDDING_OUT_DIM)
        x = self.agr1(x)
        x = x.reshape(-1, 16, SPATIAL_EMBEDDING_OUT_DIM)
        x = self.agr2(x)
        x = x.view(batch_size, -1)

        #aggregation end x.shape = (batch, 5120)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.soft(self.fc4(x))

        return x


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.attn_heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_heads)])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        pooled_outputs = []
        for attn_head in self.attn_heads:
            attn_weights = self.softmax(attn_head(x))
            pooled_output = torch.mean(attn_weights * x, dim=1)
            pooled_outputs.append(pooled_output)
        return torch.cat(pooled_outputs, dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, heads):
        super(SpatialAttention, self).__init__()
        self.spatial = nn.MultiheadAttention(SPATIAL_EMBEDDING_OUT_DIM, heads, batch_first=True)

    def forward(self, x):
        batch_size, temporal_size, spatial_size, segment_size = x.shape
        out = x.reshape(-1, spatial_size, segment_size)
        out, n = self.spatial(out, out, out, need_weights=False)
        out = out.view(batch_size, temporal_size, spatial_size, segment_size)
        return out + x


class TempAttention(nn.Module):
    def __init__(self, heads):
        super(TempAttention, self).__init__()
        self.temp = nn.MultiheadAttention(SPATIAL_EMBEDDING_OUT_DIM, heads, batch_first=True)

    def forward(self, x):
        batch_size, temporal_size, spatial_size, segment_size = x.shape
        out = torch.swapaxes(x, 1, 2)
        out = out.reshape(-1, temporal_size, segment_size)
        out, n = self.temp(out, out, out, need_weights=False)
        out = out.view(batch_size, spatial_size, temporal_size, segment_size)
        out = torch.swapaxes(out, 1, 2)
        return out + x


class VideoEmbedding(nn.Module):
    def __init__(self):
        super(VideoEmbedding, self).__init__()

        self.embedding = nn.Embedding(256, SPATIAL_EMBEDDING_OUT_DIM)
        self.position_embedding = nn.Embedding(144, SPATIAL_EMBEDDING_OUT_DIM)
        self.temporal_embedding = nn.Embedding(20, SPATIAL_EMBEDDING_OUT_DIM)

    def forward(self, x):
        device = self.embedding.weight.get_device()

        x = self.segment_image(x, SEGMENT_SIZE)

        batch_size, temporal_size, spatial_size, channels, height, width = x.shape
        segment_size = channels * SEGMENT_SIZE * SEGMENT_SIZE

        x = x.view(batch_size, temporal_size, spatial_size, segment_size)

        #print(x.get_device())
        #print(self.embedding.weight.device)

        x = x * 255
        x = x.int()

        #print(x.get_device())
        #print(self.embedding.weight.device)

        x = self.embedding(x)  #(b, f, 144, 1200, 256)
        x = x.mean(dim=3)  #(b, f, 144, 256)

        pos_emb = torch.arange(0, 144)
        pos_emb = self.position_embedding(pos_emb.to(device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, temporal_size, spatial_size, SPATIAL_EMBEDDING_OUT_DIM)

        x += pos_emb

        cls = torch.zeros((batch_size, temporal_size, 1, SPATIAL_EMBEDDING_OUT_DIM))
        x = torch.cat((cls.to(device), x), dim=2)

        temp_emb = torch.arange(0, temporal_size)
        temp_emb = self.temporal_embedding(temp_emb.to(device))
        temp_emb = temp_emb.unsqueeze(1).unsqueeze(0).expand(batch_size, temporal_size, spatial_size + 1,
                                                             SPATIAL_EMBEDDING_OUT_DIM)

        x += temp_emb

        return x.reshape((batch_size, temporal_size, spatial_size+1, SPATIAL_EMBEDDING_OUT_DIM))

    def segment_image(self, image, segment_size):
        batch_size, frames, channels, height, width = image.shape

        batch = []
        for b in range(batch_size):

            clip_frames = []
            for f in range(frames):

                frame_segments = []
                for y in range(0, height, segment_size):
                    for x in range(0, width, segment_size):
                        segment = image[b, f, :, y:y + segment_size, x:x + segment_size]
                        frame_segments.append(segment)
                clip_frames.append(torch.stack(frame_segments))

            batch.append(torch.stack(clip_frames))

        return torch.stack(batch)


if __name__ == '__main__':
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    net = NeuralNet(3)
    print('NN')

    test_batch_size = 1
    dataset = Dataset.VideoDataset('test.json')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    print("dataset")

    for images, labels in test_loader:
        output = net(images)

        print('output type: {}, expected: {}'.format(output, labels))

        break
