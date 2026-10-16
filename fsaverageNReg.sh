
FEATURES="thickness area curv sulc volume"
HEMIS="lh rh"
SESSIONS="Hyperfine" #1.5T 3T 

project="$1"
seg="$2"
template="$3"
#templates="$2"
#features="$3"
#subjects_file="$4
SURF_ROOT=/data/project/unity/Hajer/PhD/${seg}/${project}/mrr
SCRIPT_DIR='/data/project/unity/Hajer/PhD/scripts'
subjects_file=/data/project/unity/Hajer/PhD/${seg}/${project}/subjects.txt

if [ "$project" = "KCL-HYPE" ]; then
  echo "$project"
  SESSIONS="ses-HFC ses-HFE" #ses-GE 
  echo "$SESSIONS"
elif [ "$project" = "UCL-paired" ]; then
  echo "$project"
  echo "$SESSIONS"
else
  echo "Invalid project name"
  exit
fi



while IFS= read -r SUBJ; do

  for SES in $SESSIONS; do
    export SUBJECTS_DIR=/data/project/unity/Hajer/PhD/${seg}/${project}/mrr/$SES
    cd $SUBJECTS_DIR
    #ln -s $FREESURFER_HOME/subjects/${template} ${template}

    mv $SUBJECTS_DIR/${SUBJ}_${SES}/surf/${HEMI}.${FEAT}.${template} \
    $SUBJECTS_DIR/${SUBJ}_${SES}/surf/${HEMI}.${FEAT}.${template}.bak 2>/dev/null


    echo "Processing $SUBJ $SES"

    for HEMI in $HEMIS; do
      for FEAT in $FEATURES; do
        INFILE=$SUBJECTS_DIR/${SUBJ}_${SES}/surf/${HEMI}.${FEAT}
        OUTFILE=$SUBJECTS_DIR/${SUBJ}_${SES}/surf/${HEMI}.${FEAT}.${template}

        if [ -f "$INFILE" ]; then
          echo "  $HEMI $FEAT -> ${template}"
          mri_surf2surf \
            --srcsubject ${SUBJ}_${SES} \
            --trgsubject ${template} \
            --hemi $HEMI \
            --sval $INFILE \
            --tval $OUTFILE \
            --src_type curv \
            --trg_type curv 
            
          # mris_fwhm \
          #   --s fsaverage5 \
          #   --hemi $HEMI \
          #   --cortex \
          #   --i $SUBJECTS_DIR/${SUBJ}_${SES}/surf/$HEMI.$FEAT.fsaverage5 \
          #   --o $SUBJECTS_DIR/${SUBJ}_${SES}/surf/$HEMI.$FEAT.fsaverage5.sm10 \
          #   --fwhm 10 \

          

        else
          echo "  WARNING: missing $INFILE for $SUBJ ($HEMI)"
        fi

      done
    done

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc" \
      --fsaverage5

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc"  \
      --thickness --fsaverage5

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc.a2009s"  \
      --fsaverage5

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc.a2009s"  \
      --thickness --fsaverage5


    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc" 

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc" \
      --thickness

    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc.a2009s"  \


    python ${SCRIPT_DIR}/MIND-master/run.py \
      --subject ${SUBJ} \
      --session ${SES} \
      --surf_root ${SURF_ROOT} \
      --parc "aparc.a2009s"  \
      --thickness

  done
done < "$subjects_file"
