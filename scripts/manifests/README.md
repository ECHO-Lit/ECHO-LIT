# Dataset selection manifests

These CSVs list *which* clips AudioLens uses from each dataset and carry the
labels the app reads. **No audio is stored here.** The scripts in `scripts/`
read a manifest, get the clips from the official source, and write both into
`Backend/data/`.

| File | Used by | Source dataset | Licence of the source |
|------|---------|----------------|-----------------------|
| `ravdess_subset.csv` (144 rows) | `download_ravdess.py` | [RAVDESS](https://zenodo.org/records/1188976) | CC BY-NC-SA 4.0 |
| `saa_metadata.csv` (150) | `download_saa.py` | [Speech Accent Archive](https://accent.gmu.edu) | CC BY-NC-SA 4.0 |
| `l2_arctic_metadata.csv` (150), `l2_arctic_phone_error_annotations.csv` (891) | `prepare_l2arctic.py` | [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) | CC BY-NC 4.0 |
| `cv_valid_dev.csv` (100) | `prepare_common_voice.py` | [Common Voice (Kaggle v1)](https://www.kaggle.com/datasets/mozillaorg/common-voice) | CC0 per Mozilla (confirm at source) |

The metadata and annotations are derived from those datasets, so they keep the
source licence: attribution always, share-alike for RAVDESS and SAA,
non-commercial for RAVDESS, SAA and L2-ARCTIC. AudioLens's own code is MIT.

Citations
- RAVDESS: Livingstone & Russo (2018), PLoS ONE 13(5): e0196391.
- SAA: Weinberger, S. H. (2015). Speech Accent Archive. George Mason University.
- L2-ARCTIC: Zhao et al. (2018), Interspeech 2018.
- Common Voice: Ardila et al. (2020), LREC 2020.
