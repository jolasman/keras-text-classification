import os, pickle, sys
from tqdm import tqdm



def removeFile(filename):
    """Removes the filename if it exists"""
    if os.path.exists(filename):
        print("\n\nRemoving the existing ", filename, "file!")
        os.remove(filename)


def loadFile(filename):
    "returns the filename data"
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        print("There are no enough files to start predicting; please train the model again")
        sys.exit()


def countingWords(training_content):
    """Returns the number of words inside the trainning dataset"""
    num_words = []
    pbar = tqdm(range(len(training_content)))
    for step in pbar:
        num_words.extend(training_content[step].split())
        pbar.set_description("Counting trhe number of words")

    number_words_dataset = len(num_words) + 1
    print("Number of words: ", number_words_dataset)
    return number_words_dataset