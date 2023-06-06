# Parkinson's Disease Detection using Voice and Spiral Handwriting

This project focuses on detecting Parkinson's disease using voice and spiral handwriting data. It combines a Support Vector Machine (SVM) model for voice-based detection and a Convolutional Neural Network (CNN) model for spiral handwritten image-based detection. The models are then fused using the late fusion technique, with a Logistic Regression model to provide the final output.

## Installation

Install the required dependencies:

```python
pip install -r requirements.txt
```

## Usage

1. Prepare the input data:
- Record the audio of the patient using the provided voice recording functionality.
- Capture the image of the spiral handwriting using the integrated image capture feature.

2. Run the inference:
- Provide the recorded audio and captured image as input to the pre-trained models.
- The SVM model analyzes the voice data to detect Parkinson's disease.
- The CNN model processes the spiral handwriting image for Parkinson's detection.
- Apply the late fusion technique to combine the outputs of the SVM and CNN models.
- Use the Logistic Regression model to generate the final output.

## Configuration

- No additional configuration is required for the pre-trained models.
- Adjust the threshold or decision criteria in the code based on the project's specific requirements.

## Contributing

We welcome contributions from the community. If you'd like to contribute to the project, please follow these guidelines:
- Fork the repository and create a new branch for your feature or bug fix.
- Ensure your code follows the project's coding standards.
- Submit a pull request, describing the changes you've made and their purpose.

## Testing

To run the tests, use the following command:
```
python tests.py
```
Make sure the dependencies are installed before running the tests.

## Troubleshooting

If you encounter any issues or errors, please refer to the [troubleshooting guide](docs/troubleshooting.md) for possible solutions.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to acknowledge the following individuals and resources for their contributions to this project:

- [Reference Paper on Late Fusion Technique](https://www.example.com/paper)
- [Scikit-learn Library](https://scikit-learn.org/)
- [Keras Library](https://keras.io/)

## Contact Information

For any questions or inquiries, please contact us at [email@example.com](mailto:email@example.com).
