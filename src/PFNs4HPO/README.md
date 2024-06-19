# FT-PFN

## Usage

The models used for our final setup can be found in `final_models/`.

```python
import pfns4hpo

ftpfn = pfns4hpo.PFN_MODEL(name="bopfn_broken_unisep_1000curves_10params_2M")
```


## Retrain FT-PFN

Download and unzip training dataset [here](https://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/prior_bopfn_broken.zip) (~40Gb), then run with the correct path of the downloaded data:


```bash
python main.py --epochs=800 --emsize=512 --nlayers=6 --num_borders 1000 --batch_size=25 --subsample=1 --num_gpus 1 --prior hpo_lc_pfn_bopfn_broken --output_file bopfn_broken_1000curves_10params_2M.pt --seq_len 1000 --num_features 12 --border_batch_size 1000 --load_path ${PATH_CHECKPOINT_DATASET_HERE} --no-full_support --linspace_borders
```