# GrapgRAG Experiments

[![Python Application Test with Github Actions](https://github.com/BobZhang26/Bob_PythonTemplate1/actions/workflows/cicd.yml/badge.svg)](https://github.com/BobZhang26/Bob_PythonTemplate1/actions/workflows/cicd.yml)

## Overview

This project is focused on experiments with Graph-based Retrieval-Augmented Generation (GrapgRAG). It includes various scripts and notebooks for connecting to Google Sheets, interacting with Neo4j, and generating responses using OpenAI's models.

## Features

- **Google Sheets Integration**: Connect and retrieve data from Google Sheets.
- **Neo4j Integration**: Create and query vector indexes in a Neo4j graph database.
- **OpenAI Integration**: Generate responses using OpenAI's models.
- **Data Processing**: Use pandas for data manipulation and analysis.
- **Jupyter Notebooks**: Interactive notebooks for running experiments and visualizing results.
- **CI/CD**: Continuous integration and deployment using GitHub Actions.

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/BobZhang26/GrapgRAG_experiments.git
    cd GrapgRAG_experiments
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a [.env](http://_vscodecontentref_/0) file in the root directory and add the following variables:
        ```plaintext
        GOOGLE_SHEET_CREDS=path/to/your/json/keyfile.json
        NEO4J_URL=bolt://localhost:7687
        NEO4J_USERNAME=your_username
        NEO4J_PASSWORD=your_password
        OPENAI_API_KEY=your_openai_api_key
        ```

## Usage

### Running Jupyter Notebooks

1. **Start Jupyter Notebook**:
    ```sh
    jupyter notebook
    ```

2. **Open and run the notebooks**:
    - [annotation.ipynb](http://_vscodecontentref_/1): Annotate and process data from Google Sheets.
    - [retrieve.ipynb](http://_vscodecontentref_/2): Retrieve and process data from Neo4j and generate responses using OpenAI.

### Running Scripts

1. **Connect to Google Sheets**:
    ```python
    from conn import connect2Googlesheet
    spreadsheet = connect2Googlesheet()
    ```

2. **Generate Responses**:
    ```python
    from libs import generate_response
    response, context = generate_response(graph, query, method, model)
    ```

## Included Tools and Libraries

- **Makefile**: Simplify common tasks.
- **Pytest**: Testing framework.
- **Pandas**: Data manipulation and analysis.
- **Pylint**: Code linting.
- **Dockerfile**: Containerize the application.
- **GitHub Copilot**: AI pair programmer.
- **Jupyter and IPython**: Interactive computing.
- **GitHub Actions**: Continuous integration and deployment.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

