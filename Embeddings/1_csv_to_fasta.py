input_file = '../Benchmarking_Datasets/Benchmark_20_pos.csv' # Specify the input file path
output_file = 'Benchmark_20_pos.fasta' # Specify the output file path

fp_write = open(output_file,'w')

count = 1
with open(input_file) as fp:
	for line in fp:
		line = line.replace('\n','')
		tabs = line.split(',')
		fp_write.write('>sp|' + str(count) + '|\n')
		fp_write.write(tabs[1])
		fp_write.write('\n')
		count = count + 1
fp_write.close()