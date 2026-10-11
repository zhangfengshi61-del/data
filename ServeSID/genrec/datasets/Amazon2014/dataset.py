import json
import os

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Amazon2014(AbstractDataset):
    """Amazon2014 datasets in the RecBoard-master protocol (550_LOU leave-one-out).

    Reads the processed RecBoard layout:

        <amazon2014_root>/<category>/
            user_item_sequences.jsonl   {"user": int, "items": [int, ...]}
            item_text.jsonl              {"item": int, "asin": str, "title": str, "description": str}

    Item and user identifiers are the 1-based integer ids used by RecBoard.
    """

    def __init__(self, config: dict):
        super(Amazon2014, self).__init__(config)

        self.category = config['category']
        assert self.category, 'Amazon2014 requires --category (e.g. Beauty / Toys)'
        self.data_root = config.get(
            'amazon2014_root',
            '/data/fszhang/RecBoard-master/data/processed',
        )
        self._category_dir = os.path.join(self.data_root, self.category)
        assert os.path.isdir(self._category_dir), f'Missing RecBoard processed dir: {self._category_dir}'

        self._load_item_seqs()
        self._build_id_mapping()
        self._load_item_meta()

        self.log(
            f'[DATASET] Amazon2014/{self.category}: '
            f'{self.n_users} users, {self.n_items - 1} items'
        )

    def _load_item_seqs(self):
        seq_file = os.path.join(self._category_dir, 'user_item_sequences.jsonl')
        if not os.path.exists(seq_file):
            raise FileNotFoundError(f'Missing user_item_sequences.jsonl: {seq_file}')
        self.all_item_seqs = {}
        with open(seq_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                user = str(obj['user'])
                items = [str(int(i)) for i in obj['items']]
                if len(items) > 0:
                    self.all_item_seqs[user] = items

    def _build_id_mapping(self):
        item_ids = set()
        user_ids = set(self.all_item_seqs.keys())
        for seq in self.all_item_seqs.values():
            item_ids.update(seq)
        sorted_items = sorted(item_ids, key=int)
        self.id_mapping['item2id'] = {'[PAD]': 0}
        self.id_mapping['id2item'] = ['[PAD]']
        for item in sorted_items:
            self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
            self.id_mapping['id2item'].append(item)
        self.id_mapping['user2id'] = {'[PAD]': 0}
        self.id_mapping['id2user'] = ['[PAD]']
        for user in sorted(user_ids, key=int):
            self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
            self.id_mapping['id2user'].append(user)

    def _load_item_meta(self):
        meta_file = os.path.join(self._category_dir, 'item_text.jsonl')
        self.item2meta = {}
        if not os.path.exists(meta_file):
            self.log('[DATASET] No item_text.jsonl; item2meta left empty.')
            return
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                item = str(int(obj['item']))
                title = clean_text(obj.get('title') or '')
                desc = clean_text(obj.get('description') or '')
                self.item2meta[item] = f'{title} {desc}'.strip()
