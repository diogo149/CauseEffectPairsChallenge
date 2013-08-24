import data_io


def main():
    print("Reading the valid pairs")
    valid = data_io.read_valid_pairs()

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")
    predictions = classifier.predict(valid)
    predictions = predictions.flatten()

    print("Writing predictions to file")
    data_io.write_submission(predictions)

if __name__ == "__main__":
    main()
