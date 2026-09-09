evaluation_dataset = [
    {
        "question": "What dataset was used in this project?",
        "ground_truth": "The MNIST dataset was used.",
        "expected_page": 1
    },
    {
        "question": "How many training images are in the MNIST dataset?",
        "ground_truth": "There are 60,000 training images.",
        "expected_page": 1
    },
    {
        "question": "How many test images are in the MNIST dataset?",
        "ground_truth": "There are 10,000 test images.",
        "expected_page": 1
    },
    {
        "question": "Why were the pixel values divided by 255?",
        "ground_truth": "The pixel values were divided by 255 to normalize them to the range 0 to 1, which improves convergence and training stability.",
        "expected_page": 1
    },
    {
        "question": "What is the architecture of the ANN?",
        "ground_truth": "The ANN uses a Flatten layer, a Dense layer with 128 neurons and ReLU activation, and a Dense output layer with 10 neurons and Softmax activation.",
        "expected_page": 2
    },
    {
        "question": "What optimizer was used to compile the model?",
        "ground_truth": "The Adam optimizer was used.",
        "expected_page": 2
    },
    {
        "question": "What loss function was used?",
        "ground_truth": "Sparse categorical crossentropy was used.",
        "expected_page": 2
    },
    {
        "question": "How was the ANN trained?",
        "ground_truth": "The ANN was trained using model.fit with epochs=10, batch_size=32, and validation_split=0.2.",
        "expected_page": 2
    },
    {
        "question": "How were predictions converted into class labels?",
        "ground_truth": "The argmax operation was used to obtain the predicted class label.",
        "expected_page": 3
    },
    {
        "question": "What filename was used to save the trained model?",
        "ground_truth": "The trained model was saved as mnist_ann_model.keras.",
        "expected_page": 3
    },

    # Negative test:
    # The answer should NOT be found in the document.
    {
        "question": "What accuracy did the ANN achieve on the test set?",
        "ground_truth": None,
        "expected_page": None
    }
]
if __name__ == "__main__":
    for i, item in enumerate(evaluation_dataset, start=1):
        print(f"{i}. {item['question']}")