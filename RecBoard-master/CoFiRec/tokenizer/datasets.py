import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dataset_name = data_path.split("/")[2]
        if self.dataset_name == "Instruments":
            self.collab_embeddings = np.load(f"./data/{self.dataset_name}/embedding.npy")[1:]
        elif self.dataset_name == "Beauty":
            self.collab_embeddings = torch.load(f"./data/{self.dataset_name}/embedding.pt")[1:]
        else:
            self.collab_embeddings = torch.load(f"./data/{self.dataset_name}/embedding.pt")[1:]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        if self.dataset_name == "Instruments":
            emb = self.embeddings[index, :-1, :]
        else:
            emb = self.embeddings[index, :, :]
        collab_emb = self.collab_embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        collab_emb=torch.FloatTensor(collab_emb)
        return (tensor_emb, collab_emb), index

    def __len__(self):
        return len(self.embeddings)


class PairedEmbDataset(data.Dataset):

    def __init__(self, data_path):
        title_emb_path = data_path.replace(".emb", ".title-emb")
        desc_emb_path = data_path.replace(".emb", ".description-emb")
        self.title_embeddings = np.load(title_emb_path)
        self.desc_embeddings = np.load(desc_emb_path)
        
        if len(self.title_embeddings) != len(self.desc_embeddings):
            raise ValueError("Title and description embeddings must have the same number of items: {} != {}".format(len(self.title_embeddings), len(self.desc_embeddings)))
        
        self.dim = self.title_embeddings.shape[-1]

    def __getitem__(self, index):
        title_emb = self.title_embeddings[index]
        desc_emb = self.desc_embeddings[index]
        tensor_title_emb = torch.FloatTensor(title_emb)
        tensor_desc_emb = torch.FloatTensor(desc_emb)
        return (tensor_title_emb, tensor_desc_emb), index

    def __len__(self):
        return len(self.title_embeddings)