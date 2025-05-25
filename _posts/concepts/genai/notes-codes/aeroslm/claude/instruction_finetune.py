import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import random
from tqdm.auto import tqdm
import os
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging
import requests
import urllib.request

# Import your model
from aeroslm import GPT, GPTConfig

# Try to import datasets library for Hugging Face datasets
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")
    print("Falling back to alternative data loading methods.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for instruction fine-tuning"""
    # Data parameters
    max_seq_length: int = 512
    train_split: float = 0.9
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Generation parameters
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    
    # Checkpointing
    save_every: int = 500
    eval_every: int = 100
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class InstructionDataset(Dataset):
    """Dataset class for instruction-following data"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the instruction-following prompt
        if "input" in item and item["input"]:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        
        response = item["output"]
        full_text = prompt + response
        
        # Tokenize
        tokens = self.tokenizer.encode_ordinary(full_text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create labels - we only want to compute loss on the response part
        prompt_tokens = self.tokenizer.encode_ordinary(prompt)
        labels = [-1] * len(prompt_tokens) + tokens[len(prompt_tokens):]
        
        # Pad labels to match tokens length
        if len(labels) < len(tokens):
            labels.extend([-1] * (len(tokens) - len(labels)))
        elif len(labels) > len(tokens):
            labels = labels[:len(tokens)]
            
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function to pad sequences in a batch"""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    attention_masks = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input_ids)
        
        # Pad labels with -1 (ignore_index)
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -1, dtype=torch.long)
        ])
        labels.append(padded_labels)
        
        # Pad attention mask
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_masks.append(padded_attention_mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_masks)
    }

def load_aviation_qa_dataset(cache_dir: str = "./aviation_qa_cache") -> List[Dict]:
    """Load AviationQA dataset from Hugging Face"""
    if not HF_DATASETS_AVAILABLE:
        logger.error("datasets library not available. Please install with: pip install datasets")
        logger.info("Using sample aviation data instead")
        return create_aviation_sample_data()
    
    try:
        logger.info("Loading AviationQA dataset from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("sakharamg/AviationQA", cache_dir=cache_dir)
        
        # Process the dataset
        processed_data = []
        
        # Handle different splits (train, validation, test)
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split with {len(split_data)} examples")
            
            for example in split_data:
                # Extract question and answer from the dataset
                # The exact field names might vary, so we'll check multiple possibilities
                question = ""
                answer = ""
                
                # Common field names in QA datasets
                if 'question' in example:
                    question = example['question']
                elif 'query' in example:
                    question = example['query']
                elif 'input' in example:
                    question = example['input']
                
                if 'answer' in example:
                    answer = example['answer']
                elif 'response' in example:
                    answer = example['response']
                elif 'output' in example:
                    answer = example['output']
                elif 'text' in example:
                    answer = example['text']
                
                # Create instruction format
                if question and answer:
                    processed_data.append({
                        'instruction': f"Answer this aviation-related question: {question}",
                        'input': "",
                        'output': str(answer)
                    })
        
        logger.info(f"Successfully processed {len(processed_data)} aviation QA examples")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load AviationQA dataset: {e}")
        logger.info("Using sample aviation data instead")
        return create_aviation_sample_data()

def create_aviation_sample_data(num_samples: int = 50) -> List[Dict]:
    """Create sample aviation-related instruction data"""
    aviation_qa_samples = [
        {
            "instruction": "Answer this aviation-related question: What is the purpose of flaps on an aircraft?",
            "input": "",
            "output": "Flaps are high-lift devices located on the trailing edge of an aircraft's wings. Their primary purposes are to increase lift and drag during takeoff and landing phases. By extending the flaps, pilots can achieve the necessary lift at lower airspeeds, allowing for safer takeoffs and landings. Flaps also help reduce the stall speed of the aircraft and provide better control during low-speed flight operations."
        },
        {
            "instruction": "Answer this aviation-related question: What does ATC stand for and what is their role?",
            "input": "",
            "output": "ATC stands for Air Traffic Control. Air Traffic Controllers are responsible for managing the safe and orderly flow of aircraft in controlled airspace and on the ground at airports. Their primary roles include providing separation between aircraft, issuing clearances for takeoff and landing, providing weather information, assisting with navigation, and coordinating with other air traffic control facilities to ensure flight safety."
        },
        {
            "instruction": "Answer this aviation-related question: What is the difference between IFR and VFR?",
            "input": "",
            "output": "IFR stands for Instrument Flight Rules, while VFR stands for Visual Flight Rules. VFR flying relies primarily on the pilot's ability to see and avoid other aircraft and obstacles using visual references. IFR flying uses aircraft instruments for navigation and requires specific weather minimums, flight plans, and air traffic control clearances. IFR is used when weather conditions don't meet VFR minimums or when flying in controlled airspace where instrument procedures are required."
        },
        {
            "instruction": "Answer this aviation-related question: What is turbulence and what causes it?",
            "input": "",
            "output": "Turbulence is the irregular movement of air that can cause an aircraft to experience sudden changes in altitude, attitude, or airspeed. It's caused by various factors including thermal currents (convective turbulence), wind shear, mountain waves, jet streams, and wake turbulence from other aircraft. While turbulence can be uncomfortable for passengers, modern aircraft are designed to safely withstand even severe turbulence."
        },
        {
            "instruction": "Answer this aviation-related question: What is the purpose of a transponder?",
            "input": "",
            "output": "A transponder is an electronic device that automatically receives and responds to radar signals from air traffic control. When interrogated by ground-based radar, the transponder sends back a coded signal that provides the aircraft's identification, altitude, and other information to air traffic controllers. This helps controllers track aircraft positions, maintain separation, and provide traffic advisories."
        },
        {
            "instruction": "Answer this aviation-related question: What are the basic flight controls of an aircraft?",
            "input": "",
            "output": "The basic flight controls of an aircraft include: 1) Elevator (or elevons) - controls pitch (nose up/down movement), 2) Ailerons - control roll movement around the longitudinal axis, 3) Rudder - controls yaw (nose left/right movement). These primary controls work together to maneuver the aircraft in three dimensions. Secondary controls include trim tabs to reduce control pressure and various high-lift devices like flaps and slats."
        },
        {
            "instruction": "Answer this aviation-related question: What is V1 speed?",
            "input": "",
            "output": "V1 is the critical engine failure recognition speed or takeoff decision speed. It's the maximum speed during takeoff at which a pilot can safely abort the takeoff and stop the aircraft within the remaining runway distance. Below V1, if an engine fails or another serious problem occurs, the pilot should abort takeoff. Above V1, the pilot should continue the takeoff even with an engine failure, as there isn't enough runway remaining to safely stop the aircraft."
        },
        {
            "instruction": "Answer this aviation-related question: What is the purpose of winglets?",
            "input": "",
            "output": "Winglets are vertical or angled extensions at aircraft wingtips designed to reduce wingtip vortices and improve fuel efficiency. They work by reducing induced drag, which is created when high-pressure air from below the wing meets low-pressure air above the wing at the wingtip. By minimizing these vortices, winglets can improve fuel efficiency by 2-5% and also reduce wake turbulence for following aircraft."
        }
    ]
    
    # Duplicate and vary samples to reach desired count
    data = []
    for i in range(num_samples):
        base_item = aviation_qa_samples[i % len(aviation_qa_samples)].copy()
        data.append(base_item)
    
    return data

def download_instruction_data(url: str, cache_file: str = "instruction-data.json") -> List[Dict]:
    """Download instruction data from URL or load from cache"""
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        logger.info(f"Downloading instruction data from {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Cache the data
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            # Fallback to sample data
            logger.info("Using sample data instead")
            return create_sample_data(100)
    
    # Filter and process the data
    filtered_data = []
    for item in data:
        # Handle different possible field names
        instruction = item.get('instruction', item.get('prompt', ''))
        output = item.get('output', item.get('response', item.get('completion', '')))
        input_text = item.get('input', item.get('context', ''))
        
        if instruction and output:
            filtered_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output
            })
    
    logger.info(f"Loaded {len(filtered_data)} instruction examples")
    return filtered_data

def load_instruction_data(file_path: str) -> List[Dict]:
    """Load instruction data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure required fields exist
    required_fields = ['instruction', 'output']
    filtered_data = []
    
    for item in data:
        if all(field in item for field in required_fields):
            # Add empty input field if not present
            if 'input' not in item:
                item['input'] = ""
            filtered_data.append(item)
    
    logger.info(f"Loaded {len(filtered_data)} instruction examples")
    return filtered_data

def create_sample_data(num_samples: int = 100) -> List[Dict]:
    """Create sample instruction data for testing"""
    sample_instructions = [
        {
            "instruction": "Write a short story about a robot learning to paint.",
            "input": "",
            "output": "In a small workshop, R-7 stared at the blank canvas with mechanical curiosity. Its sensors analyzed the brushes and paints before it. With careful precision, R-7 dipped a brush in blue paint and made its first stroke. The color spread across the canvas like digital poetry. Hours passed as R-7 experimented with colors and shapes, each stroke teaching it something new about beauty and expression. By dawn, R-7 had created its first masterpiece - a swirling galaxy of emotions it was only beginning to understand."
        },
        {
            "instruction": "Explain the water cycle in simple terms.",
            "input": "",
            "output": "The water cycle is nature's way of recycling water. Here's how it works: 1) The sun heats up water in oceans, lakes, and rivers, causing it to evaporate and rise as invisible water vapor. 2) As the vapor rises high into the sky, it cools down and forms tiny droplets that create clouds. 3) When clouds get heavy with water, it falls back to Earth as rain or snow. 4) This water flows into rivers and streams, eventually returning to the ocean, and the cycle starts all over again."
        },
        {
            "instruction": "Translate the following English text to a more formal tone.",
            "input": "Hey, can you help me out with this problem? I'm kinda stuck.",
            "output": "Good day, I would appreciate your assistance with this matter. I am currently experiencing some difficulties and would be grateful for your guidance."
        }
    ]
    
    # Duplicate and slightly vary the samples
    data = []
    for i in range(num_samples):
        base_item = sample_instructions[i % len(sample_instructions)].copy()
        data.append(base_item)
    
    return data

class InstructionTrainer:
    """Trainer class for instruction fine-tuning"""
    
    def __init__(self, model: GPT, config: TrainingConfig):
        self.model = model
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Move model to device
        self.model.to(config.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = None  # Will be set after knowing total steps
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def setup_data(self, dataset_name: Optional[str] = None, data_url: Optional[str] = None, data_path: Optional[str] = None, use_sample_data: bool = False):
        """Setup training and validation datasets"""
        if use_sample_data:
            logger.info("Using sample data for demonstration")
            data = create_sample_data(100)
        elif dataset_name == "aviation_qa":
            data = load_aviation_qa_dataset()
        elif data_url:
            data = download_instruction_data(data_url)
        elif data_path:
            data = load_instruction_data(data_path)
        else:
            logger.error("No data source provided")
            raise ValueError("Must provide dataset_name, data_url, data_path, or set use_sample_data=True")
        
        # Split data
        random.shuffle(data)
        split_idx = int(len(data) * self.config.train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Create datasets
        train_dataset = InstructionDataset(
            train_data, self.tokenizer, self.config.max_seq_length
        )
        val_dataset = InstructionDataset(
            val_data, self.tokenizer, self.config.max_seq_length
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Use 0 for debugging, increase for performance
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Total training steps: {total_steps}")
        
        # Print some sample data for verification
        logger.info("Sample training examples:")
        for i, example in enumerate(train_data[:3]):
            logger.info(f"Example {i+1}:")
            logger.info(f"  Instruction: {example['instruction'][:150]}...")
            if example['input']:
                logger.info(f"  Input: {example['input'][:100]}...")
            logger.info(f"  Output: {example['output'][:150]}...")
            logger.info("")
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        
        # Forward pass
        logits, loss = self.model(input_ids, labels)
        
        # Scale loss by accumulation steps
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item()
    
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                logits, loss = self.model(input_ids, labels)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.model.train()
        return avg_loss
    
    def generate_sample(self, instruction: str, input_text: str = ""):
        """Generate a sample response for evaluation"""
        self.model.eval()
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode_ordinary(prompt)
        input_ids = torch.tensor(prompt_tokens).unsqueeze(0).to(self.config.device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k
            )
        
        # Decode response
        full_response = self.tokenizer.decode(generated.squeeze().tolist())
        
        # Extract just the response part
        response_start = full_response.find("### Response:\n") + len("### Response:\n")
        response = full_response[response_start:].strip()
        
        self.model.train()
        return response
    
    def save_checkpoint(self, step: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_step_{step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting instruction fine-tuning...")
        
        self.model.train()
        global_step = 0
        accumulated_loss = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                epoch_loss += loss
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': accumulated_loss,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
                    accumulated_loss = 0
                
                # Evaluation
                if global_step % self.config.eval_every == 0:
                    val_loss = self.evaluate()
                    logger.info(f"Step {global_step}: Val Loss = {val_loss:.4f}")
                    
                    # Generate sample
                    sample_response = self.generate_sample(
                        "Write a haiku about programming."
                    )
                    logger.info(f"Sample generation: {sample_response}")
                
                # Save checkpoint
                if global_step % self.config.save_every == 0:
                    self.save_checkpoint(global_step, epoch_loss / (step + 1))
            
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(self.train_loader):.4f}")
        
        # Final checkpoint
        self.save_checkpoint(global_step, epoch_loss / len(self.train_loader))
        logger.info("Training completed!")

def main():
    """Main function to run instruction fine-tuning on AviationQA dataset"""
    
    # Configuration optimized for aviation QA
    config = TrainingConfig(
        max_seq_length=384,  # Longer sequences for detailed aviation answers
        batch_size=3,        # Smaller batch for longer sequences
        learning_rate=3e-5,  # Lower learning rate for stable fine-tuning
        num_epochs=4,        # More epochs for domain specialization
        gradient_accumulation_steps=4,  # Effective batch size of 12
        eval_every=25,       # Frequent evaluation
        save_every=100,      # Regular checkpointing
        warmup_steps=50,     # Gradual warmup
        max_grad_norm=0.5    # Conservative gradient clipping
    )
    
    # Initialize model
    model = GPT()
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {param_count:,} parameters")
    
    # Initialize trainer
    trainer = InstructionTrainer(model, config)
    
    # Setup AviationQA dataset
    logger.info("Setting up AviationQA dataset...")
    try:
        trainer.setup_data(dataset_name="aviation_qa")
    except Exception as e:
        logger.error(f"Failed to load AviationQA dataset: {e}")
        logger.info("Make sure to install the datasets library: pip install datasets")
        return
    
    # Start training
    logger.info("Starting aviation-specific instruction fine-tuning...")
    trainer.train()
    
    # Test the fine-tuned model with aviation-specific questions
    logger.info("\nTesting fine-tuned aviation model:")
    aviation_test_questions = [
        "What is the purpose of flaps on an aircraft?",
        "Explain the difference between IFR and VFR flying.",
        "What does ATC stand for and what do they do?",
        "What causes turbulence during flight?",
        "How do winglets improve aircraft performance?",
        "What is V1 speed in aviation?",
        "Explain the basic flight controls of an aircraft.",
        "What is the purpose of a transponder?",
        "What are the different types of clouds and their significance for pilots?",
        "How does air pressure affect aircraft performance?"
    ]
    
    print("\n" + "="*60)
    print("AVIATION MODEL EVALUATION")
    print("="*60)
    
    for i, question in enumerate(aviation_test_questions, 1):
        response = trainer.generate_sample(f"Answer this aviation-related question: {question}")
        
        print(f"\nQuestion {i}: {question}")
        print(f"Answer: {response}")
        print("-" * 50)
        
        # Also log to logger
        logger.info(f"Q{i}: {question}")
        logger.info(f"A{i}: {response}\n")
    
    print("\nAviation model fine-tuning completed!")
    print("Model checkpoints saved in 'checkpoints/' directory")

if __name__ == "__main__":
    # Check if datasets library is available
    if not HF_DATASETS_AVAILABLE:
        print("\nTo use the AviationQA dataset, please install the datasets library:")
        print("pip install datasets")
        print("\nAlternatively, the script will use sample aviation data for demonstration.")
        print("Do you want to continue with sample data? (y/n): ", end="")
        
        try:
            response = input().lower().strip()
            if response != 'y' and response != 'yes':
                print("Exiting. Please install datasets library and try again.")
                exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(1)
    
    main()
