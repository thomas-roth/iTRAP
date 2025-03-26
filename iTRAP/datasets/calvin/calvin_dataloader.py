import os
import numpy as np
from torch.utils.data import Dataset


class CalvinDataLoader(Dataset):
    def __init__(self, calvin_root, dataset_path, annotations_folder="lang_annotations"):
        self.calvin_root = calvin_root
        self.dataset_path = os.path.join(self.calvin_root, dataset_path)

        self.annotations = np.load(os.path.join(self.dataset_path, annotations_folder, "auto_lang_ann.npy"), allow_pickle=True).item()


    def __len__(self):
        return len(self.annotations["info"]["indx"])


    def __getitem__(self, index):
        data = self.get_single_seq(index)
        return {**data} # shallow copy


    def get_single_seq(self, seq_index):
        anno_seq_start_index, anno_seq_stop_index = self.annotations["info"]["indx"][seq_index]
        anno_seq = self.annotations["language"]["task"][seq_index]
        anno_text_seq = self.annotations["language"]["ann"][seq_index]

        obs_seq = []
        for index in range(anno_seq_start_index, anno_seq_stop_index):
            try:
                frame = np.load(os.path.join(self.dataset_path, f"episode_{index:07d}.npz"), allow_pickle=True)
                obs_seq.append(dict(frame))
            except FileNotFoundError as e:
                print(e)
                pass
        
        obs_seq = {key: [dic[key] for dic in obs_seq] for key in obs_seq[0]}

        return {"obs": obs_seq, "anno": anno_seq, "anno_text": anno_text_seq}
