import os.path
import tempfile
import shutil
import time
from typing import Sequence
from torchvision.utils import save_image

import logging
logger = logging.getLogger(__name__)

class ImageSaver:
    def __init__(self, target_dir, rank, max_save_num=50000, compressed=True):
        self.target_dir = target_dir
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_name = self.tmp_dir.name
        self.max_save_num = max_save_num
        self._have_saved_num = 0
        self.compressed = compressed
        self.rank = rank
        self.max_upload_num = 500
        self._have_uploaded_num = 0

        while True:
            time.sleep(1.0)
            if not os.path.exists(self.target_dir):
                try:
                    os.makedirs(self.target_dir, exist_ok=True)
                except:
                    logger.warning(f'{self.target_dir} does not exist, trying again...')
            else:
                break

    def save_image(self, images, filenames):
       for sample, filename in zip(images, filenames):
            if isinstance(filename, Sequence):
                filename = filename[0]
            path = f'{self.tmp_dir_name}/{filename}'
            if self._have_saved_num >= self.max_save_num:
               break
            save_image(sample, path, nrow=4, normalize=True, value_range=(-1, 1))
            self._have_saved_num += 1
    def upload_image(self, images, filenames):
        for sample, filename in zip(images, filenames):
            if isinstance(filename, Sequence):
                filename = filename[0]
            path = f'{self.target_dir}/{filename}'
            if self._have_uploaded_num >= self.max_upload_num:
               break
            save_image(sample, path, nrow=4, normalize=True, value_range=(-1, 1))
            self._have_uploaded_num += 1


    def upload_all(self, prefix=""):
        rank = self.rank
        if self.compressed:
            # zip the files in tmp dir
            shutil.make_archive(f"{rank}", 'zip', self.tmp_dir_name+"/")
            # copy to target dir
            os.system(f'cp {rank}.zip {self.target_dir}/{prefix}_{rank}.zip')
        else:
            raise NotImplementedError
            # os.system(f'cp -r {self.tmp_dir_name} {self.target_dir}/{rank}')

        # clear tmp dir
        self.tmp_dir.cleanup()
