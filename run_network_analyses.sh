#!/bin/bash

#base_dir="/Users/Hajer/unity/PhD/Adult/KCL-HYPE/mind/DK"

templates=("fsaverage5") #"fsaverage" "fsaverage5"
features=("5feat") #"4feat""
atlases=("aparc" "aparc.a2009s" ) #"aparc" "aparc.a2009s" "aparc.DKTatlas" 
paths=("supersynth") #"supersynth" 
label_pairs=(
    "GE HFC"
    "GE HFE"
    "HFC HFE"
)
methods=("mind")
#Add command line arguments to specify the template, feature, and label pair to process
project="$1"
#method="$2" #mind or MSN
tool="$2" #recon-any or freesurfer
#path="$4" #mrr or supersynth
#templates="$2"
#features="$3"
#subjects_file="$4"
for path in "${paths[@]}"; do
    for method in "${methods[@]}"; do
        base_dir="/data/project/unity/Hajer/PhD/${tool}/${project}/${path}/${method}"
        subjects_file="/data/project/unity/Hajer/PhD/${tool}/${project}/subjects.txt"
        if [ "$project" = "KCL-HYPE" ]; then
        echo "$project"
        elif [ "$project" = "UCL-paired" ]; then
        echo "$project"
        label_pairs=(
            #"3T Hyperfine"
            "3T 1.5T"
            #"1.5T Hyperfine"
        )
        else
        echo "Invalid project name"
        exit
        fi
        #
        for template in "${templates[@]}"; do
            mkdir -p "${base_dir}/${template}/results"
            for atlas in "${atlases[@]}"; do
                for feature in "${features[@]}"; do
                    for pair in "${label_pairs[@]}"; do

                        read -r hf lf <<< "$pair"
                        mkdir "${base_dir}/${template}/results"
                        python network_analyses.py \
                            --mind-dir "${base_dir}/${template}/${atlas}/${feature}" \
                            --hf-label "$hf" \
                            --lf-label "$lf" \
                            --output-dir "${base_dir}/${template}/results" \
                            --suffix "${atlas}_${feature}" --template ${template}

                    done
                done
            done
        done
    done
done