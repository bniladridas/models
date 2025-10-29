#!/bin/bash

# Main script for GPT-2 training project

# Check if setup is done
if [ ! -d "venv" ]; then
    echo "Setup not detected. Running setup..."

    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv venv

    # Activate the virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate

    # Install dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt

    # Install pre-commit
    echo "Installing pre-commit..."
    pip install pre-commit

    # Install pre-commit hooks
    echo "Installing pre-commit hooks..."
    pre-commit install

    # Install the project in editable mode
    echo "Installing project in editable mode..."
    pip install -e .

    echo "Setup complete!"
else
    # Activate venv
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Check if dataset is prepared
if [ ! -d "data/tokenized_wikitext" ]; then
    echo -e "${YELLOW}Dataset not found. Preparing dataset...${NC}"
    python src/data.py
    echo -e "${GREEN}Dataset prepared!${NC}"
fi

# Activate venv
source venv/bin/activate

while true; do
    # Choose action
    echo -e "${GREEN}Choose action:${NC}"
    echo "1) Train model"
    echo "2) Run tests"
    echo "3) Evaluate model"
    echo "4) Exit"
    read -p "Enter choice (1-4): " action

    case $action in
        1)
            # Choose model type
            echo -e "${GREEN}Choose model type:${NC}"
            echo "1) GPT-2"
            echo "2) GPT-Neo"
            read -p "Enter choice (1 or 2): " model_choice

            case $model_choice in
                1)
                    MODEL_TYPE="gpt2"
                    ;;
                2)
                    MODEL_TYPE="gpt-neo"
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Skipping.${NC}"
                    continue
                    ;;
            esac

            # Choose number of epochs
            echo -e "${GREEN}Enter number of epochs (e.g., 3):${NC}"
            read -p "Epochs: " NUM_EPOCHS

            # Now choose training
            echo -e "${GREEN}Choose training method:${NC}"
            echo "1) Local (CPU/GPU)"
            echo "2) Docker (CPU only)"
            read -p "Enter choice (1 or 2): " choice

            case $choice in
                1)
                    echo -e "${BLUE}Running local training with $MODEL_TYPE for $NUM_EPOCHS epochs...${NC}"
                    MODEL_TYPE=$MODEL_TYPE NUM_EPOCHS=$NUM_EPOCHS python src/train.py
                    ;;
                2)
                    echo -e "${BLUE}Running Docker training with $MODEL_TYPE for $NUM_EPOCHS epochs...${NC}"
                    docker build --build-arg MODEL_TYPE=$MODEL_TYPE --build-arg NUM_EPOCHS=$NUM_EPOCHS -t gpt2-trainer .
                    docker run --rm -e MODEL_TYPE=$MODEL_TYPE -e NUM_EPOCHS=$NUM_EPOCHS gpt2-trainer
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Skipping.${NC}"
                    continue
                    ;;
            esac
            ;;
        2)
            echo -e "${BLUE}Running tests...${NC}"
            pytest tests/ -v
            ;;
        3)
            echo -e "${BLUE}Running evaluation...${NC}"
            python src/evaluate.py
            ;;
        4)
            echo -e "${GREEN}Exiting. Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Try again.${NC}"
            ;;
    esac
done

# Choose number of epochs
echo -e "${GREEN}Enter number of epochs (e.g., 3):${NC}"
read -p "Epochs: " NUM_EPOCHS

# Now choose training
echo -e "${GREEN}Choose training method:${NC}"
echo "1) Local (CPU/GPU)"
echo "2) Docker (CPU only)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo -e "${BLUE}Running local training with $MODEL_TYPE for $NUM_EPOCHS epochs...${NC}"
        MODEL_TYPE=$MODEL_TYPE NUM_EPOCHS=$NUM_EPOCHS python src/train.py
        ;;
    2)
        echo -e "${BLUE}Running Docker training with $MODEL_TYPE for $NUM_EPOCHS epochs...${NC}"
        docker build --build-arg MODEL_TYPE=$MODEL_TYPE --build-arg NUM_EPOCHS=$NUM_EPOCHS -t gpt2-trainer .
        docker run --rm -e MODEL_TYPE=$MODEL_TYPE -e NUM_EPOCHS=$NUM_EPOCHS gpt2-trainer
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# After training, rename the model directory with timestamp
if [ -d "models/trained_model" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    NEW_NAME="models/trained_model_${MODEL_TYPE}_${TIMESTAMP}"
    mv models/trained_model "$NEW_NAME"
    echo -e "${GREEN}Model saved as $NEW_NAME${NC}"
else
    echo -e "${YELLOW}No models/trained_model directory found.${NC}"
fi
