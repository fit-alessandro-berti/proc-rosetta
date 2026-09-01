#/bin/sh
python3 test.py --curriculum simple --conformance-method footprints >results_easy.txt
python3 test.py --curriculum medium --conformance-method footprints >results_medium.txt
python3 test.py --curriculum complex --conformance-method footprints >results_complex.txt
