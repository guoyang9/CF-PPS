from tqdm import tqdm


def training_progress(loader, epoch, epochs, loss, debug):
    return tqdm(loader, desc="Running Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                ncols=117, unit=' steps', unit_scale=True,
                postfix={"loss": "{:.3f}".format(float(loss))}) if debug else loader


def testing_progress(loader, epoch, epochs, debug):
    return tqdm(loader, desc="Testing Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                ncols=117, unit=' users', unit_scale=True) if debug else loader


def building_progress(df, debug, desc='building'):
    return tqdm(df.iterrows(), desc=desc, total=len(df),
                ncols=117, unit=' entries', unit_scale=True) if debug else df.iterrows()
