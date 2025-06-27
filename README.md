# Data Deduplication

## Environment Setup
```bash
uv init --bare --python 3.10
uv sync --python 3.10
source .venv/bin/activate
uv pip install -r ./requirements.txt
```

## Image Descriptors / Embeddings

Generates a descriptor for the data with the [SSCD](https://github.com/facebookresearch/sscd-copy-detection) Model from FAIR

## Test-Dataset Indexing

To compute the embeddings of a single dataset, run
```bash
python compute_embeddings.py 
    --dataset lmms-lab/VQAv2 
    --split test
```

To compute all embeddings for the selected datasets in `lmms_datasets.py`, run
```bash
python run_all_embeddings.py
```

We index all image test datasets from [LMMS-lab](https://huggingface.co/lmms-lab/datasets?sort=alphabetical), since these are the availible used in lmms-eval. Datasets focused on video understanding are not considert.

This results in the following embeddings:
```
Found embeddings directory: /fsx/luis_wiedmann/data-dedup/embeddings-lmms

Counting image embeddings in embeddings-lmms/
================================================================================
COCO-Caption2017                                  40,670 embeddings (dim: 512)
COCO-Caption                                      40,775 embeddings (dim: 512)
ChartQA                                            2,500 embeddings (dim: 512)
DC100_EN                                             100 embeddings (dim: 512)
DocVQA_DocVQA                                      5,188 embeddings (dim: 512)
DocVQA_InfographicVQA                              3,288 embeddings (dim: 512)
Egothink_Activity                                    100 embeddings (dim: 512)
Egothink_Forecasting                                 100 embeddings (dim: 512)
Egothink_Localization_location                        50 embeddings (dim: 512)
Egothink_Localization_spatial                         50 embeddings (dim: 512)
Egothink_Object_affordance                            50 embeddings (dim: 512)
Egothink_Object_attribute                             50 embeddings (dim: 512)
Egothink_Object_existence                             50 embeddings (dim: 512)
Egothink_Planning_assistance                          50 embeddings (dim: 512)
Egothink_Planning_navigation                          50 embeddings (dim: 512)
Egothink_Reasoning_comparing                          50 embeddings (dim: 512)
Egothink_Reasoning_counting                           50 embeddings (dim: 512)
Egothink_Reasoning_situated                           50 embeddings (dim: 512)
Ferret-Bench                                         120 embeddings (dim: 512)
GQA_test_all_images                                2,993 embeddings (dim: 512)
LLaVA-Bench-Wilder                                   128 embeddings (dim: 512)
LLaVA-NeXT-Interleave-Bench_in_domain             42,462 embeddings (dim: 512)
LLaVA-NeXT-Interleave-Bench_multi_view_in_domain    174,264 embeddings (dim: 512)
LLaVA-NeXT-Interleave-Bench_out_of_domain          9,167 embeddings (dim: 512)
LiveBench_2024-05                                    292 embeddings (dim: 512)
LiveBench_2024-06                                    250 embeddings (dim: 512)
LiveBench_2024-07                                    250 embeddings (dim: 512)
LiveBench_2024-08                                      8 embeddings (dim: 512)
LiveBench_2024-09                                    200 embeddings (dim: 512)
MIA-Bench                                            400 embeddings (dim: 512)
MMBench_EN                                         6,718 embeddings (dim: 512)
MME                                                2,374 embeddings (dim: 512)
MMMU                                              12,141 embeddings (dim: 512)
MMT-Benchmark                                     28,198 embeddings (dim: 512)
MMT_MI-Benchmark                                  43,660 embeddings (dim: 512)
MMVet                                                218 embeddings (dim: 512)
NoCaps                                            10,600 embeddings (dim: 512)
OCRBench-v2                                       10,000 embeddings (dim: 512)
POPE                                               9,000 embeddings (dim: 512)
RealWorldQA                                          765 embeddings (dim: 512)
RefCOCO                                            5,000 embeddings (dim: 512)
SEED-Bench-2                                      66,776 embeddings (dim: 512)
SEED-Bench                                        44,289 embeddings (dim: 512)
ST-VQA                                             4,070 embeddings (dim: 512)
ScienceQA_ScienceQA-FULL                           2,017 embeddings (dim: 512)
ScienceQA_ScienceQA-IMG                            2,017 embeddings (dim: 512)
TextCaps                                           3,289 embeddings (dim: 512)
VQAv2                                             92,916 embeddings (dim: 512)
VisualWebBench_action_ground                         103 embeddings (dim: 512)
VisualWebBench_action_prediction                     281 embeddings (dim: 512)
VisualWebBench_element_ground                        413 embeddings (dim: 512)
VisualWebBench_element_ocr                           245 embeddings (dim: 512)
VisualWebBench_heading_ocr                            46 embeddings (dim: 512)
VisualWebBench_web_caption                           134 embeddings (dim: 512)
VisualWebBench_webqa                                 314 embeddings (dim: 512)
VizWiz-Caps                                        8,000 embeddings (dim: 512)
VizWiz-VQA                                         8,000 embeddings (dim: 512)
ai2d-no-mask                                       3,088 embeddings (dim: 512)
ai2d                                               3,088 embeddings (dim: 512)
continous-benchmark_ESPN                             738 embeddings (dim: 512)
continous-benchmark_arxiv                            221 embeddings (dim: 512)
continous-benchmark_bbc_news                          21 embeddings (dim: 512)
continous-benchmark_wiki_articles                     84 embeddings (dim: 512)
flickr30k                                         31,783 embeddings (dim: 512)
textvqa                                            5,734 embeddings (dim: 512)
vstar-bench                                          191 embeddings (dim: 512)
================================================================================
TOTAL                                            730,287 embeddings

Summary:
  • Total image embeddings: 730,287
  • Number of datasets: 66
  • Embedding files processed: 66

Top 15 datasets by number of embeddings:
------------------------------------------------------------
 1. LLaVA-NeXT-Interleave-Bench_multi_view_in_domain    174,264 (dim: 512)
 2. VQAv2                                   92,916 (dim: 512)
 3. SEED-Bench-2                            66,776 (dim: 512)
 4. SEED-Bench                              44,289 (dim: 512)
 5. MMT_MI-Benchmark                        43,660 (dim: 512)
 6. LLaVA-NeXT-Interleave-Bench_in_domain     42,462 (dim: 512)
 7. COCO-Caption                            40,775 (dim: 512)
 8. COCO-Caption2017                        40,670 (dim: 512)
 9. flickr30k                               31,783 (dim: 512)
10. MMT-Benchmark                           28,198 (dim: 512)
11. MMMU                                    12,141 (dim: 512)
12. NoCaps                                  10,600 (dim: 512)
13. OCRBench-v2                             10,000 (dim: 512)
14. LLaVA-NeXT-Interleave-Bench_out_of_domain      9,167 (dim: 512)
15. POPE                                     9,000 (dim: 512)
... and 51 more datasets                   73,586

Dataset size distribution:
----------------------------------------
Large (>100k)     1 datasets,    174,264 embeddings
Medium (10k-100k)  12 datasets,    464,270 embeddings
Small (1k-10k)   18 datasets,     85,531 embeddings
Tiny (<1k)       35 datasets,      6,222 embeddings
```

## Duplicate Retrival
To check if an image is already present in the indexed test datasets, you simply have to get its embedding and compute the cosine similarity to the cached datasets. For single images, this can be done by
```bash
python find_similar_images.py 
    --images data/image-vqav2.jpg 
    --embeddings embeddings-lmms/lmms-lab-VQAv2_test_embeddings.npy 
    --image_ids embeddings-lmms/lmms-lab-VQAv2_test_image_ids.npy
```