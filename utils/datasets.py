import kagglehub

DATASET_IN_CACHE = False

REAL_VOICE_DF = "mathurinache/the-lj-speech-dataset"
FAKE_VOICE_DF = "andreadiubaldo/wavefake-test"
ASV_DF = "awsaf49/asvpoof-2019-dataset"
def get_datasets_path(in_cache: bool = False):
    if in_cache:
        return (
            f"~/.cache/kagglehub/datasets/{REAL_VOICE_DF}/versions/1",
            f"~/.cache/kagglehub/datasets/{FAKE_VOICE_DF}/versions/1",
        )
    else:
        return (
            kagglehub.dataset_download(REAL_VOICE_DF),
            kagglehub.dataset_download(FAKE_VOICE_DF),
        )
