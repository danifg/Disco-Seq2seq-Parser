set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
# Argument handling
config=$1
checkpoint=$2
results_path=$3
HELP="\ntest.sh <config> <model_checkpoint> [<results_path>]\n"
[ -z "$config" ] && echo -e $HELP && exit 1
[ -z "$checkpoint" ] && echo -e $HELP && exit 1
[ -z "$results_path" ] && results_path=""
set -o nounset 

# Load config
. "$config"

# Default path
if [ "$results_path" == "" ];then
    # fix for ensembles
    single_checkpoint=$(echo $checkpoint | sed 's@\.pt:.*@@')
    results_path=$(dirname $single_checkpoint)/$TEST_TAG/valid
fi
mkdir -p $(dirname $results_path)

# to profile decoder
# decorate target function with @profile
# test_command="kernprof -o generate.lprof -l fairseq/generate.py"
# python -m line_profiler generate.py.lprof
test_command=fairseq-generate

# Decode to get predicted action sequence
if [ ! -f "${results_path}.actions" ];then
    echo "$test_command $FAIRSEQ_GENERATE_ARGS --path $checkpoint --results-path ${results_path}"
    $test_command $FAIRSEQ_GENERATE_ARGS \
        --path $checkpoint \
        --results-path ${results_path} 
fi


# FOR EACH TASK EVALUATE FOR EACH OF THE SUB TASKS INVOLVED e.g. AMR+NER
for single_task  in $(python -c "print(' '.join('$TASK_TAG'.split('+')))");do

    if [ "$single_task" == "AMR" ];then
    
        # AMR (Smatch)
        # Create the AMR from the model obtained actions
        if [ "$BLINK_CACHE_PATH" == "" ];then
    
            # Smatch evaluation without wiki
            # Compute score in the background
            python smatch/smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE \
                 ${results_path}.amr \
                 -r 10 \
                 > ${results_path}.smatch
            # plot score
            cat ${results_path}.smatch
            
        else
    
            # Smatch evaluation with wiki
            # add wiki
            python scripts/retyper.py \
                --inputfile ${results_path}.amr \
                --outputfile ${results_path}.wiki.amr \
                --skipretyper \
                --wikify \
                --blinkcachepath $BLINK_CACHE_PATH \
                --blinkthreshold 0.0

            # Compute score in the background
            smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE_WIKI \
                 ${results_path}.wiki.amr \
                 -r 10 \
                 > ${results_path}.wiki.smatch
            # plot score
            cat ${results_path}.wiki.smatch
        
        fi

    elif [ "$single_task" == "dep-parsing" ];then
	python delin_inoSWAP.py ${results_path}.actions $ORACLE_FOLDER/dev.source $ORACLE_FOLDER/dev.pos > ${results_path}.discbracket

	discodop eval $ORACLE_FOLDER/dev.discbracket ${results_path}.discbracket $ORACLE_FOLDER/proper.prm --fmt=discbracket | tail -n 3 | head -n 1 | awk  '{print "LAS: " $4}' > ${results_path}.las
	
	cat ${results_path}.las
	

	
    else
 
        echo "Unsupported task $single_task"
        exit 1

    fi
 
done
