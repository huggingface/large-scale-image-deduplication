#!/usr/bin/env python3

from compute_embeddings import compute_embeddings

LMMS_NAMES = {
    'lmms-lab/ai2d': [],
    'lmms-lab/ai2d-no-mask': [],
    'lmms-lab/ChartQA': [],
    'lmms-lab/COCO-Caption': [],
    'lmms-lab/COCO-Caption2017': [],
    'lmms-lab/continous-benchmark': ['ESPN', 'arxiv', 'bbc_news', 'wiki_articles'], #'Wiki_img',
    'lmms-lab/DC100_EN': [],
    'lmms-lab/DocVQA': ['DocVQA', 'InfographicVQA'],
    'lmms-lab/Egothink': ['Activity', 'Forecasting', 'Localization_location', 'Localization_spatial', 'Object_affordance', 'Object_attribute', 'Object_existence', 'Planning_assistance', 'Planning_navigation', 'Reasoning_comparing', 'Reasoning_counting', 'Reasoning_situated'],
    'lmms-lab/Ferret-Bench': [],
    'lmms-lab/flickr30k': [],
    'lmms-lab/GQA': ['test_all_images'],
    'lmms-lab/LiveBench': ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09'],
    'lmms-lab/LLaVA-Bench-Wilder': [],
    'lmms-lab/LLaVA-NeXT-Interleave-Bench': ['in_domain', 'multi_view_in_domain', 'out_of_domain'],
    'lmms-lab/MIA-Bench': [],
    'lmms-lab/MMBench_EN': [],
    'lmms-lab/MME': [],
    'lmms-lab/MMT-Benchmark': [],
    'lmms-lab/MMT_MI-Benchmark': [],
    'lmms-lab/MMMU': [],
    'lmms-lab/MMVet': [],
    'lmms-lab/NoCaps': [],
    'lmms-lab/OCRBench-v2': [],
    'lmms-lab/POPE': [],
    'lmms-lab/RealWorldQA': [],
    'lmms-lab/RefCOCO': [],
    'lmms-lab/ScienceQA': ['ScienceQA-FULL', 'ScienceQA-IMG'],
    'lmms-lab/SEED-Bench': [],
    'lmms-lab/SEED-Bench-2': [],
    'lmms-lab/ST-VQA': [],
    'lmms-lab/TextCaps': [],
    'lmms-lab/textvqa': [],
    'lmms-lab/VisualWebBench': ['action_ground', 'action_prediction', 'element_ground', 'element_ocr', 'heading_ocr', 'web_caption', 'webqa'],
    'lmms-lab/VizWiz-Caps': [],
    'lmms-lab/VizWiz-VQA':[],
    'lmms-lab/VQAv2': [],
    'lmms-lab/vstar-bench': []
}

def main():
    for dataset_name, subset_names in LMMS_NAMES.items():
        if subset_names:  # Dataset has subsets
            for subset_name in subset_names:
                print(f"Processing {dataset_name} with subset {subset_name}")
                compute_embeddings(dataset_name, name=subset_name)
        else:  # Dataset has no subsets
            print(f"Processing {dataset_name}")
            compute_embeddings(dataset_name)

if __name__ == "__main__":
    main() 