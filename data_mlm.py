from data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  SpatialMlmDataset,spatial_mlm_collate)
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from torch.utils.data import DataLoader
from utils.const import IMG_DIM, IMG_LABEL_DIM, BUCKET_SIZE


def create_dataloaders(imgdbpath, datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                     opts.num_bb, opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['db']) == len(dset['img'])
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            img_db = [all_img_dbs[path] for path in dset['img']]
        else:
            assert len(dset['db']) == len(dset['img']) == 1
            img_db = all_img_dbs[dset['img'][0]]

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'

            if is_train:
                LOGGER.info(f"Loading {task} train dataset "
                            f"{dset['db']}, {[img.img_dir for img in img_db]}")
                txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                          for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} validation dataset, "
                            f"{dset['db']}, {img_db.img_dir}")
                txt_db = TxtTokLmdb(dset['db'][0], -1)

            if task.startswith('mlm'):
                dataset = build_mlm_dataset(txt_db, img_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task}')

            LOGGER.info(f"{len(dataset[0])} samples loaded")
            loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                #dataloaders[task] = PrefetchLoader(loader)
                dataloaders[task] = loader
    return dataloaders, all_img_dbs

def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader

def build_mlm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        collate_fn = spatial_mlm_collate
        datasets = [SpatialMlmDataset(t, i) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        collate_fn = spatial_mlm_collate
        dataset = SpatialMlmDataset(txt_db, img_db)

    return dataset, collate_fn