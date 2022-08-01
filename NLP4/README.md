"# nlp_ass4"
In this project I implemented a model from the SNLI competition.
Details of the results are in the report.pdf.
The model uses the snli data. https://nlp.stanford.edu/projects/snli/

Instructions about running the project:
To run the best model, run the file run_test.py. 
Need to run on gpu
To run the file, the following files need to be in the folder:
    data files:
glove.840B.300d.txt - glove word embedding 

snli_1.0/snli_1.0_train.txt - train file from snli dataset

snli_1.0/snli_1.0_dev.txt - dev file from snli dataset

snli_1.0/snli_1.0_test.txt - test file from snli dataset

    python files:
model.py
utils.py
snli_data.py
    Best model file:
distance_based_saved_model
To run trained_model on train, dev and test datasets: 
python run_test.py 

results will be available in results.txt as well as printed to screen

To run the main to train model again:
Note1: This will overwrite distance_based_saved_model
Note2: Recommended to use gpu for this part 
run the file main.py same as running run_test.py
The main.py file can run also on cpu.
run with "cpu argument": main.py --device "cpu"
results will be available in final as well as printed to screen

