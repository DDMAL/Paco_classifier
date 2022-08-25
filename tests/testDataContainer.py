import numpy as np
from Paco_classifier.preprocess import bytes2Gb
import Paco_classifier.data_loader as DataLoader

import unittest
from unittest import mock

import glob

class TestDataContainer(unittest.TestCase):
    def setUp(self) -> None:
        self.bool_byte = np.dtype(bool).itemsize
        self.float_byte = np.dtype(float).itemsize
        self.uint8_byte = np.dtype(np.uint8).itemsize

        self.images = [p for p in sorted(glob.glob("./dataset/MS73/training/images/*.png"))]
        self.bg = [p for p in sorted(glob.glob("./dataset/MS73/training/layers/bg/*.png"))]
        self.neume = [p for p in sorted(glob.glob("./dataset/MS73/training/layers/neume/*.png"))]
        self.staff = [p for p in sorted(glob.glob("./dataset/MS73/training/layers/staff/*.png"))]
        self.text = [p for p in sorted(glob.glob("./dataset/MS73/training/layers/text/*.png"))]

    def createContainer(self, ram_limit:int = 4, dum_size:int = 1):
        container = DataLoader.DataContainer(ram_limit)
        for img_path, bg_path, neu_path, sta_path, text_path in zip(self.images,
                                                                    self.bg,
                                                                    self.neume,
                                                                    self.staff,
                                                                    self.text):
            x_name = img_path.split("/")[-1]                                                        

            img_data = DataLoader.Data(x_name, img_path, size=dum_size)
            bg_data = DataLoader.Data(x_name, bg_path, size=dum_size)
            neu_data = DataLoader.Data(x_name, neu_path, size=dum_size)
            sta_data = DataLoader.Data(x_name, sta_path, size=dum_size)
            text_data = DataLoader.Data(x_name, text_path, size=dum_size)

            container.addXYPair(x_name, "bg", img_data, bg_data)
            container.addXYPair(x_name, "neu", img_data, neu_data)
            container.addXYPair(x_name, "staff", img_data, sta_data)
            container.addXYPair(x_name, "text", img_data, text_data)
        return container

    def test_data(self):
        ram_limit = 4
        dum_size = 1
        container = self.createContainer(ram_limit, dum_size)

    def checkPendingWorking(self, container:DataLoader.DataContainer, layer_name:str, 
                            gt_working_len:int, gt_pending_len:int, gt_total_len: int, prefix:str=""):
        pending_len = len(container.meta[layer_name]['pending'])
        working_len = len(container.meta[layer_name]['working'])
        ram_limit = container.ram_limit

        self.assertEqual(working_len, gt_working_len, f"{prefix}Failed to clear working list with len: {working_len}")
        self.assertEqual(pending_len, gt_pending_len, f"{prefix}Failed to restore pending list with len: {pending_len}")
        self.assertEqual(working_len + pending_len, gt_total_len, f"{prefix}pending/working/total: {pending_len}/{working_len}/{gt_total_len}")

        used_ram = 0
        for working_y in container.meta[layer_name]['working']:
            self.assertIsNotNone(working_y.img, f"{prefix}Failed to load {working_y.path} in working list.")
            self.assertNotIn(working_y, container.meta[layer_name]['pending'], f"{prefix}Failed to remove {working_y.path} from pending list.")

            working_x = container.meta['Image'][working_y.x_name]
            self.assertIsNotNone(working_x.img, f"{prefix}Failed to load {working_x.path} in working list.")

            used_ram += working_y.size
            used_ram += working_x.size
        self.assertTrue(used_ram <= ram_limit, f"{prefix}Required RAM {used_ram} exceed limit {ram_limit}")

        for pending_y in container.meta[layer_name]['pending']:
            self.assertIsNone(pending_y.img, f"{prefix}Failed to del img of {pending_y.path} in pending list.")

            pending_x = container.meta['Image'][pending_y.x_name]
            self.assertIsNone(pending_x.img, f"{prefix}Failed to del img og {pending_x.path} in pending list.")

    @mock.patch('numpy.load')
    def test_init_ram_exceed_limit(self, mock_np_load):
        """
        All X/Y pair exceed the RAM limit
        """
        mock_np_load.return_value = np.zeros((256, 256, 3))
        # The RAM limit is 2G, but each X/Y pair requires 2*3G = 6G
        ram_limit = 2
        dum_size = 3
        container = self.createContainer(ram_limit, dum_size)
        total_layer_len = len(container.meta['bg']['pending'])
        for layer_name in ["bg", "neu", "staff", "text"]:
            self.assertRaises(ValueError, container.initWorkingList, layer_name)

    @mock.patch('numpy.load')
    def test_reload_ram_exceed_limit(self, mock_np_load):
        """
        When one of the X/Y pair exceed the RAM limit. Reload the prvious working list.
        """
        mock_np_load.return_value = np.zeros((256, 256, 3))

        container = DataLoader.DataContainer(ram_limit=11)
        for idx, dum_size in enumerate([2, 10, 2, 1]):
        #for idx, dum_size in enumerate([2, 2, 1, 10]):
            x_name = f"{idx}_{dum_size}"
            img_data = DataLoader.Data(x_name, f"img_path/{x_name}", size=dum_size)
            bg_data = DataLoader.Data(x_name, f"bg_path/{x_name}", size=dum_size)
            neu_data = DataLoader.Data(x_name, f"neu_path/{x_name}", size=dum_size)
            sta_data = DataLoader.Data(x_name, f"sta_path/{x_name}", size=dum_size)
            text_data = DataLoader.Data(x_name, f"text_path/{x_name}", size=dum_size)

            container.addXYPair(x_name, "bg", img_data, bg_data)
            container.addXYPair(x_name, "neu", img_data, neu_data)
            container.addXYPair(x_name, "staff", img_data, sta_data)
            container.addXYPair(x_name, "text", img_data, text_data)

        total_layer_len = len(container.meta['bg']['pending'])
        for layer_name in ["bg", "neu", "staff", "text"]:
            container.initWorkingList(layer_name)
            pending_len = len(container.meta[layer_name]['pending'])
            working_len = len(container.meta[layer_name]['working'])
            # [2*2G, 2*2G, 2*1G] should be loaded in the working list
            # [2*10G] should stay in the pending list
            self.checkPendingWorking(container, layer_name, 3, 1, total_layer_len, "After Init: ")

            # Trying to lad 10*2 Gb into 11G RAM will raise ValueError
            self.assertRaises(ValueError, container.reloadPendingList, layer_name)

            container.delWorkingList(layer_name)
            self.checkPendingWorking(container, layer_name, 0, total_layer_len, total_layer_len, "After delete: ")

    @mock.patch('numpy.load')
    def test_reload_and_fill_ram(self, mock_np_load):
        """
        Load more images to fill the RAM.
        """
        mock_np_load.return_value = np.zeros((256, 256, 3))

        container = DataLoader.DataContainer(ram_limit=21)
        for idx, dum_size in enumerate([2, 2, 1, 10]):
            x_name = f"{idx}_{dum_size}"
            img_data = DataLoader.Data(x_name, f"img_path/{x_name}", size=dum_size)
            bg_data = DataLoader.Data(x_name, f"bg_path/{x_name}", size=dum_size)
            neu_data = DataLoader.Data(x_name, f"neu_path/{x_name}", size=dum_size)
            sta_data = DataLoader.Data(x_name, f"sta_path/{x_name}", size=dum_size)
            text_data = DataLoader.Data(x_name, f"text_path/{x_name}", size=dum_size)

            container.addXYPair(x_name, "bg", img_data, bg_data)
            container.addXYPair(x_name, "neu", img_data, neu_data)
            container.addXYPair(x_name, "staff", img_data, sta_data)
            container.addXYPair(x_name, "text", img_data, text_data)

        total_layer_len = len(container.meta['bg']['pending'])
        for layer_name in ["bg", "neu", "staff", "text"]:
            # Load 2 * 10G int 21G RAM
            container.initWorkingList(layer_name)
            pending_len = len(container.meta[layer_name]['pending'])
            working_len = len(container.meta[layer_name]['working'])
            # [2*20G] should be loaded in the working list
            # [2*1G, 2*2G, 2*2G] should stay in the pending list
            self.checkPendingWorking(container, layer_name, 1, 3, total_layer_len, "After Init: ")

            # Trying to lad 10*2 Gb into 11G RAM will raise ValueError
            # Remove 2*20G from the working list, now
            # [2*1G, 2*2G, 2*2G] should stay in the working list
            # [2*20G] should be loaded in the pending list
            container.reloadPendingList(layer_name)
            self.checkPendingWorking(container, layer_name, 3, 1, total_layer_len, "After Reload: ")

            container.delWorkingList(layer_name)
            self.checkPendingWorking(container, layer_name, 0, total_layer_len, total_layer_len, "After delete: ")

    @mock.patch('numpy.load')
    def test_container_init_del(self, mock_np_load):
        mock_np_load.return_value = np.zeros((256, 256, 3))
        ram_limit = 5
        dum_size = 1
        container = self.createContainer(ram_limit, dum_size)

        total_layer_len = len(container.meta['bg']['pending'])
        for layer_name in ["bg", "neu", "staff", "text"]:
            container.initWorkingList(layer_name)
            pending_len = len(container.meta[layer_name]['pending'])
            working_len = len(container.meta[layer_name]['working'])
            self.checkPendingWorking(container, layer_name, working_len, pending_len, total_layer_len, "After Init: ")

            for i in range(5):
                container.reloadPendingList(layer_name)
                pending_len = len(container.meta[layer_name]['pending'])
                working_len = len(container.meta[layer_name]['working'])
                self.checkPendingWorking(container, layer_name, working_len, pending_len, total_layer_len, "After Init: ")

            container.delWorkingList(layer_name)
            self.checkPendingWorking(container, layer_name, 0, total_layer_len, total_layer_len, "After delete: ")

    @mock.patch('numpy.load')
    def test_reload_stuff(self, mock_np_load):
        """
        """
        mock_np_load.return_value = np.zeros((256, 256, 3))

        container = DataLoader.DataContainer(ram_limit=21)
        random_ram_size = np.random.randint(high=11, low=1, size=20)
        for idx, dum_size in enumerate(random_ram_size.tolist()):
            x_name = f"{idx}_{dum_size}"
            img_data = DataLoader.Data(x_name, f"img_path/{x_name}", size=dum_size)
            bg_data = DataLoader.Data(x_name, f"bg_path/{x_name}", size=dum_size)
            neu_data = DataLoader.Data(x_name, f"neu_path/{x_name}", size=dum_size)
            sta_data = DataLoader.Data(x_name, f"sta_path/{x_name}", size=dum_size)
            text_data = DataLoader.Data(x_name, f"text_path/{x_name}", size=dum_size)

            container.addXYPair(x_name, "bg", img_data, bg_data)
            container.addXYPair(x_name, "neu", img_data, neu_data)
            container.addXYPair(x_name, "staff", img_data, sta_data)
            container.addXYPair(x_name, "text", img_data, text_data)

        total_layer_len = len(container.meta['bg']['pending'])
        for layer_name in ["bg", "neu", "staff", "text"]:
            # Load 2 * 10G int 21G RAM
            container.initWorkingList(layer_name)
            pending_len = len(container.meta[layer_name]['pending'])
            working_len = len(container.meta[layer_name]['working'])
            # [2*20G] should be loaded in the working list
            # [2*1G, 2*2G, 2*2G] should stay in the pending list
            self.checkPendingWorking(container, layer_name, working_len, pending_len, total_layer_len, "After Init: ")

            # Trying to lad 10*2 Gb into 11G RAM will raise ValueError
            # Remove 2*20G from the working list, now
            # [2*1G, 2*2G, 2*2G] should stay in the working list
            # [2*20G] should be loaded in the pending list
            for i in range(10):
                container.reloadPendingList(layer_name)
                pending_len = len(container.meta[layer_name]['pending'])
                working_len = len(container.meta[layer_name]['working'])
                self.checkPendingWorking(container, layer_name, working_len, pending_len, total_layer_len, "After Reload: ")

            container.delWorkingList(layer_name)
            self.checkPendingWorking(container, layer_name, 0, total_layer_len, total_layer_len, "After delete: ")

if __name__ == "__main__":
    unittest.main()