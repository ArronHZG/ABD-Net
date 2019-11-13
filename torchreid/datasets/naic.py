import glob
import os.path as osp
import re

from torchreid.datasets.bases import BaseImageDataset


# class Naic(BaseImageDataset):
#     """
#     Naic
#
#     Dataset statistics:
#     # identities: 1501 (+1 for background)
#     # images: 12936 (train) + 3368 (query) + 15913 (gallery)
#     """
#
#     def __init__(self, root='root', verbose=True, **kwargs):
#         super(Naic, self).__init__()
#         self.dataset_dir = osp.join(root, 'naic', "naic-train")
#         self.train_list_txt = 'train_list.txt'
#         self.query_list_txt = 'query_list.txt'
#         self.gallery_list_txt = 'train_list.txt'
#
#         self._check_before_run()
#
#         train = self._process_dir(self.train_list_txt)
#         query = self._process_dir(self.query_list_txt)
#         gallery = self._process_dir(self.gallery_list_txt)
#
#         if verbose:
#             print("=> naic loaded")
#             self.print_dataset_statistics(train, query, gallery)
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_list_txt):
#             raise RuntimeError("'{}' is not available".format(self.train_list_txt))
#         if not osp.exists(self.query_list_txt):
#             raise RuntimeError("'{}' is not available".format(self.query_list_txt))
#         if not osp.exists(self.gallery_list_txt):
#             raise RuntimeError("'{}' is not available".format(self.gallery_list_txt))
#
#     def _process_dir(self, list_txt_path):
#         dataset = []
#         with open(list_txt_path, "r+") as f:
#             lines = f.readlines()
#         label_image = {}
#         for line in lines:
#             relate_path, label = line.split(" ")
#             label = label[:-1]
#             dataset.append((osp.join(self.dataset_dir, relate_path), int(label), 1))
#
#         return dataset


class Naic(BaseImageDataset):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'naic/NAICReID'

    def __init__(self, root='root', verbose=True, **kwargs):
        super(Naic, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'clean_bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Naic loaded")
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
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            assert -1 <= pid <= 6000  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
