import os.path as osp

from .bases import BaseImageDataset


class Naic(BaseImageDataset):
    """
    Naic

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    def __init__(self, root='root', verbose=True, **kwargs):
        super(Naic, self).__init__()
        self.dataset_dir = osp.join(root, 'naic', "naic-train")
        self.train_list_txt = osp.join(self.dataset_dir, 'train_list.txt')
        self.query_list_txt = osp.join(self.dataset_dir, 'query_list.txt')
        self.gallery_list_txt = osp.join(self.dataset_dir, 'gallery_list.txt')

        self._check_before_run()

        train = self._process_dir(self.train_list_txt)
        query = self._process_dir(self.query_list_txt)
        gallery = self._process_dir(self.gallery_list_txt)

        if verbose:
            print("=> naic loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_list_txt):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_list_txt):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_list_txt):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, list_txt_path):
        dataset = []
        with open(list_txt_path, "r+") as f:
            lines = f.readlines()
        label_image = {}
        for line in lines:
            relate_path, label = line.split(" ")
            label = label[:-1]
            dataset.append((osp.join(self.dataset_dir, relate_path), int(label), 1))

        return dataset
