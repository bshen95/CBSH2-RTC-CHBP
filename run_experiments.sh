#!/bin/bash
f="$(basename -- $1)";
trimmed=$(echo "$f" | cut -f 1 -d '.')

#./run_experiments.sh $map_name $agent_number $timeOut $flags

#even scen
echo "testing even case:"
for i in {1..25}
do
scen_file="dataset/scen-even/"$trimmed"-even-"$i".scen";
output_file=results/$trimmed/even/agent-$2-$3-$4.csv;
echo ./cbs -m $1 -a $scen_file -o $output_file -k $2 -t $3 --cluster_heuristics=$4
./cbs -m $1 -a $scen_file -o $output_file -k $2 -t $3 --cluster_heuristics=$4
done

#random scen
echo "testing random case:"
for i in {1..25}
do
  scen_file="dataset/scen-random/"$trimmed"-random-"$i".scen";
  output_file=results/$trimmed/random/agent-$2-$3-$4.csv;
  echo ./cbs -m $1 -a $scen_file -o $output_file -k $2 -t $3 --cluster_heuristics=$4
  ./cbs -m $1 -a $scen_file -o $output_file -k $2 -t $3 --cluster_heuristics=$4
done