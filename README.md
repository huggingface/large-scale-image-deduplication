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
    --output_dir embeddings
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

To find the duplicates of a whole HF dataset against the precomputed embeddings, run
```bash
python dedup_dataset.py 
  --dataset lmms-lab/RefCOCO  
  --split test
  --precomputed_dir embeddings
```

This results in the following:
```
Step 1: Computing embeddings for new dataset...
Using device: cuda
Computing embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [00:39<00:00,  7.88it/s]
Saved 10000 embeddings to embeddings-tmp/
Embedding shape: (10000, 512)
Total time: 39.71915 seconds
Time per sample: 0.00397 seconds
Model inference time: 2.04610 seconds
Model time per sample: 0.00020 seconds
Total function time: 40.81330 seconds
Step 2: Finding duplicates against precomputed embeddings...
Loaded 10000 new embeddings
Found 66 precomputed embedding files
Comparing against precomputed file 1/66: lmms-lab-Egothink_Localization_spatial_test_embeddings.npy
Comparing against precomputed file 2/66: lmms-lab-LiveBench_2024-07_test_embeddings.npy
Comparing against precomputed file 3/66: lmms-lab-SEED-Bench_test_embeddings.npy
Comparing against precomputed file 4/66: lmms-lab-Egothink_Forecasting_test_embeddings.npy
Comparing against precomputed file 5/66: lmms-lab-DC100_EN_test_embeddings.npy
Comparing against precomputed file 6/66: lmms-lab-MMBench_EN_test_embeddings.npy
Comparing against precomputed file 7/66: lmms-lab-OCRBench-v2_test_embeddings.npy
Comparing against precomputed file 8/66: lmms-lab-LLaVA-NeXT-Interleave-Bench_in_domain_test_embeddings.npy
Comparing against precomputed file 9/66: lmms-lab-ScienceQA_ScienceQA-FULL_test_embeddings.npy
Comparing against precomputed file 10/66: lmms-lab-flickr30k_test_embeddings.npy
Comparing against precomputed file 11/66: lmms-lab-Egothink_Reasoning_comparing_test_embeddings.npy
Comparing against precomputed file 12/66: lmms-lab-continous-benchmark_arxiv_test_embeddings.npy
Comparing against precomputed file 13/66: lmms-lab-VisualWebBench_webqa_test_embeddings.npy
Comparing against precomputed file 14/66: lmms-lab-COCO-Caption_test_embeddings.npy
Comparing against precomputed file 15/66: lmms-lab-LLaVA-Bench-Wilder_test_embeddings.npy
Comparing against precomputed file 16/66: lmms-lab-RefCOCO_test_embeddings.npy
Comparing against precomputed file 17/66: lmms-lab-GQA_test_all_images_test_embeddings.npy
Comparing against precomputed file 18/66: lmms-lab-LiveBench_2024-09_test_embeddings.npy
Comparing against precomputed file 19/66: lmms-lab-MIA-Bench_test_embeddings.npy
Comparing against precomputed file 20/66: lmms-lab-VisualWebBench_element_ocr_test_embeddings.npy
Comparing against precomputed file 21/66: lmms-lab-DocVQA_DocVQA_test_embeddings.npy
Comparing against precomputed file 22/66: lmms-lab-MME_test_embeddings.npy
Comparing against precomputed file 23/66: lmms-lab-POPE_test_embeddings.npy
Comparing against precomputed file 24/66: lmms-lab-ai2d-no-mask_test_embeddings.npy
Comparing against precomputed file 25/66: lmms-lab-ChartQA_test_embeddings.npy
Comparing against precomputed file 26/66: lmms-lab-Egothink_Activity_test_embeddings.npy
Comparing against precomputed file 27/66: lmms-lab-MMMU_test_embeddings.npy
Comparing against precomputed file 28/66: lmms-lab-VisualWebBench_action_ground_test_embeddings.npy
Comparing against precomputed file 29/66: lmms-lab-VQAv2_test_embeddings.npy
Comparing against precomputed file 30/66: lmms-lab-LiveBench_2024-08_test_embeddings.npy
Comparing against precomputed file 31/66: lmms-lab-LLaVA-NeXT-Interleave-Bench_multi_view_in_domain_test_embeddings.npy
Comparing against precomputed file 32/66: lmms-lab-LiveBench_2024-05_test_embeddings.npy
Comparing against precomputed file 33/66: lmms-lab-continous-benchmark_bbc_news_test_embeddings.npy
Comparing against precomputed file 34/66: lmms-lab-MMVet_test_embeddings.npy
Comparing against precomputed file 35/66: lmms-lab-SEED-Bench-2_test_embeddings.npy
Comparing against precomputed file 36/66: lmms-lab-NoCaps_test_embeddings.npy
Comparing against precomputed file 37/66: lmms-lab-continous-benchmark_ESPN_test_embeddings.npy
Comparing against precomputed file 38/66: lmms-lab-MMT-Benchmark_test_embeddings.npy
Comparing against precomputed file 39/66: lmms-lab-Egothink_Object_affordance_test_embeddings.npy
Comparing against precomputed file 40/66: lmms-lab-RealWorldQA_test_embeddings.npy
Comparing against precomputed file 41/66: lmms-lab-ai2d_test_embeddings.npy
Comparing against precomputed file 42/66: lmms-lab-TextCaps_test_embeddings.npy
Comparing against precomputed file 43/66: lmms-lab-Egothink_Planning_navigation_test_embeddings.npy
Comparing against precomputed file 44/66: lmms-lab-Egothink_Localization_location_test_embeddings.npy
Comparing against precomputed file 45/66: lmms-lab-Egothink_Planning_assistance_test_embeddings.npy
Comparing against precomputed file 46/66: lmms-lab-VizWiz-VQA_test_embeddings.npy
Comparing against precomputed file 47/66: lmms-lab-Ferret-Bench_test_embeddings.npy
Comparing against precomputed file 48/66: lmms-lab-Egothink_Reasoning_counting_test_embeddings.npy
Comparing against precomputed file 49/66: lmms-lab-VisualWebBench_action_prediction_test_embeddings.npy
Comparing against precomputed file 50/66: lmms-lab-LiveBench_2024-06_test_embeddings.npy
Comparing against precomputed file 51/66: lmms-lab-Egothink_Object_attribute_test_embeddings.npy
Comparing against precomputed file 52/66: lmms-lab-Egothink_Object_existence_test_embeddings.npy
Comparing against precomputed file 53/66: lmms-lab-Egothink_Reasoning_situated_test_embeddings.npy
Comparing against precomputed file 54/66: lmms-lab-textvqa_test_embeddings.npy
Comparing against precomputed file 55/66: lmms-lab-continous-benchmark_wiki_articles_test_embeddings.npy
Comparing against precomputed file 56/66: lmms-lab-VisualWebBench_web_caption_test_embeddings.npy
Comparing against precomputed file 57/66: lmms-lab-VisualWebBench_heading_ocr_test_embeddings.npy
Comparing against precomputed file 58/66: lmms-lab-MMT_MI-Benchmark_test_embeddings.npy
Comparing against precomputed file 59/66: lmms-lab-DocVQA_InfographicVQA_test_embeddings.npy
Comparing against precomputed file 60/66: lmms-lab-VizWiz-Caps_test_embeddings.npy
Comparing against precomputed file 61/66: lmms-lab-ScienceQA_ScienceQA-IMG_test_embeddings.npy
Comparing against precomputed file 62/66: lmms-lab-COCO-Caption2017_test_embeddings.npy
Comparing against precomputed file 63/66: lmms-lab-LLaVA-NeXT-Interleave-Bench_out_of_domain_test_embeddings.npy
Comparing against precomputed file 64/66: lmms-lab-ST-VQA_test_embeddings.npy
Comparing against precomputed file 65/66: lmms-lab-vstar-bench_test_embeddings.npy
Comparing against precomputed file 66/66: lmms-lab-VisualWebBench_element_ground_test_embeddings.npy

Timing Summary:
Total execution: 65.49147 seconds
Embedding computation: 40.83068 seconds (62.3%)
Duplicate detection: 24.66074 seconds (37.7%)
  - Loading precomputed: 0.35375 seconds (0.5%)
  - Similarity search: 24.25959 seconds (37.0%)

Found 10000 duplicate images out of 10000
Duplicate results saved to: duplicates/duplicates_lmms-lab-OCRBench-v2_test.json
```