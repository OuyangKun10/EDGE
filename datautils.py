from torch.utils.data import DataLoader, TensorDataset
from main import prepare_dataset,preprocess_dataset
import yaml
path='./conf/config.yaml'
f=open(path,'r',encoding='utf-8')
cfg=f.read()
cfg=yaml.load(cfg,Loader=yaml.FullLoader)
BATCH_SIZE=cfg['BATCH_SIZE']
def set_up_data_loader(text_path: str,
                       acosutic_path: str,
                       visual_path: str,
                       lowercase_utterances: bool=False,
                       unfolded_dialogue: bool=True,):
    dataset = preprocess_dataset(prepare_dataset(text_path=text_path,
                                                 acosutic_path=acosutic_path,
                                                 visual_path=visual_pasth,
                                                 lowercase_utterances=lowercase_utterances,
                                                 unfolded_dialogue=unfolded_dialogue),
                                unfolded_dialogue=unfolded_dialogue)
    # print(dataset.keys())
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'],
                            dataset['acoustic_input'],
                            dataset['visual_input'],
                            dataset['labels'])
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

