#!/bin/bash

python3 cricket_states.py --balls 15 --runs 30 > ./data/cricket/state_list

rm result1a.txt
touch result1a.txt

for i in {0..9}
do
    for j in {0..9}
    do
        python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.$i$j > out.txt
        python3 planner.py --mdp out.txt > out2.txt
        python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
        head -1 out3.txt >> result1a.txt
    done
done

rm result1b.txt
touch result1b.txt

for i in {0..9}
do
    for j in {0..9}
    do
        python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.$i$j > out.txt
        python3 planner.py --mdp out.txt --policy ./data/cricket/rand_pol.txt > out2.txt
        python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
        head -1 out3.txt >> result1b.txt
    done
done

rm result2a.txt
touch result2a.txt

for i in {1..20}
do
    python3 cricket_states.py --balls 10 --runs $i > ./data/cricket/state_list.txt
    python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.25 > out.txt
    python3 planner.py --mdp out.txt > out2.txt
    python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
    head -1 out3.txt >> result2a.txt
done

rm result2b.txt
touch result2b.txt

for i in {1..20}
do
    python3 cricket_states.py --balls 10 --runs $i > ./data/cricket/state_list.txt
    python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.25 > out.txt
    python3 planner.py --mdp out.txt --policy ./data/cricket/rand_pol.txt > out2.txt
    python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
    head -1 out3.txt >> result2b.txt
done

rm result3a.txt
touch result3a.txt

for i in {1..15}
do
    python3 cricket_states.py --balls $i --runs 10 > ./data/cricket/state_list.txt
    python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.25 > out.txt
    python3 planner.py --mdp out.txt > out2.txt
    python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
    head -1 out3.txt >> result3a.txt
done

rm result3b.txt
touch result3b.txt

for i in {1..15}
do
    python3 cricket_states.py --balls $i --runs 10 > ./data/cricket/state_list.txt
    python3 encoder.py --states ./data/cricket/state_list.txt --parameters ./data/cricket/sample-p1.txt --q 0.25 > out.txt
    python3 planner.py --mdp out.txt --policy ./data/cricket/rand_pol.txt > out2.txt
    python3 decoder.py --states ./data/cricket/state_list.txt --value-policy out2.txt > out3.txt
    head -1 out3.txt >> result3b.txt
done

