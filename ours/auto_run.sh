#./out -t 0.1 -e 100 -c -s 2000 ../data/yeast_nnode10_k50_coldstart/d.dimas ../data/yeast_nnode10_k50_coldstart/query/

for dataset in yeast human cora citeseer pubmed wordnet
# for dataset in wordnet
do
	for node in 10 15 20 25 30
	do
		prefix="../data/${dataset}_nnode${node}_k50_coldstart"
		./origin -t 0.01 -e 100 -c -s 20000 ${prefix}/d.dimas ${prefix}/query/ > ${dataset}_nnode${node}_coldstart.txt&
		./origin -t 0.01 -e 100 -c -s 20000 ${prefix}/d.dimas ${prefix}/query/ ${prefix}/query/ > ${dataset}_nnode${node}.txt&
		# for k in 5 30 60 90 120 150 180
		# do
		# 	taskset -c 28-47 ./faiss_index_building -t 0.9 -e 100 -c -s 200000 -k $k ${prefix}/d.dimas ${prefix}/query/ ${prefix}/query/
		# done
	done
	wait
done

