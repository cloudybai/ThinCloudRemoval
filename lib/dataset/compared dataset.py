
# class Landsat(Dataset):
#     def __init__(self, txt_path):
#         with open(txt_path, 'r') as f:
#             self.dir_list = f.readlines()
#         self.length = len(self.dir_list)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         multi_dir = self.dir_list[index].strip('\n')
#         f = h5py.File(multi_dir, 'r')
#         name = multi_dir.split('/')[-1].split('.')[0]
#         img_cloud = f['data'][:]
#         img_free = f['label'][:]
#
#         M = np.clip((np.array(img_cloud) - np.array(img_free)).sum(axis=2), 0, 1).astype(np.float32)
#         img = img_cloud.transpose([2, 0, 1]).astype(np.float32) / 255
#         gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
#         img = torch.FloatTensor(img)
#         gt = torch.FloatTensor(gt)
#         return (img, gt, M, name)